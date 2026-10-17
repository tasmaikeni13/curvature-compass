"""Optimizer dispatch and state placement for the common training runner."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from .competitors import BaselineConfig, empty_baseline_state, make_baseline_update, make_sophia_refresh
from .optim import TomographyState, direction, empty_state, make_refresh, make_update
from .runtime import host_array


class OptimizerExecution:
    def __init__(self, decoder, config, mesh, replicated):
        self.decoder, self.config = decoder, config
        self.tomography = not isinstance(config.optimizer, BaselineConfig)
        self.needs_prior = self.tomography and not config.optimizer.adaptive_alignment
        self.replicated = replicated
        # Shard basis and prior columns across data mesh to conserve device memory.
        columns = NamedSharding(mesh, P(None, "data"))
        if self.tomography and decoder.parameter_count % mesh.size:
            raise ValueError("Tomography parameter dimension must divide the basis mesh")
        self.state_sharding = TomographyState(columns, replicated, replicated, replicated) if self.tomography else replicated
        self.prior_sharding = columns if self.needs_prior else replicated
        self.empty = jax.jit(lambda: empty_state(decoder.parameter_count, config.optimizer.cap)
                             if self.tomography else empty_baseline_state(decoder, config.optimizer),
                             out_shardings=self.state_sharding)
        self.state_shapes = jax.eval_shape(self.empty)

    def put_prior(self, frame):
        if not self.needs_prior or self.config.optimizer.adaptive_alignment:
            return jax.device_put(np.zeros((1, 1), np.float32), self.replicated)
        return jax.make_array_from_callback(frame.shape, self.prior_sharding, lambda index: np.asarray(frame[index]))

    def should_refresh(self, step, state):
        if self.tomography:
            return step % self.config.optimizer.refresh_interval == 0 or int(host_array(state.refresh_step)) < 0
        if self.config.optimizer.name == "sophia_g":
            return step % self.config.optimizer.hessian_interval == 0 or int(host_array(state.hessian_step)) < 0
        return False

    def age(self, step, state):
        last = int(host_array(state.refresh_step if self.tomography else state.hessian_step))
        return step - last if last >= 0 else None

    def refresh_function(self):
        if self.tomography:
            refresh = make_refresh(self.decoder, self.config)
            return lambda p, g, b, prior, step, state: refresh(p, g, b, prior, step)
        if self.config.optimizer.name == "sophia_g":
            refresh = make_sophia_refresh(self.decoder, self.config)
            return lambda p, g, b, prior, step, state: refresh(p, b, state, step)
        return None

    def update_function(self):
        if not self.tomography:
            return make_baseline_update(self.decoder, self.config)
        # Avoid constructing/capturing an oversized parameter-dimension decay constant.
        def tensor_update(parameters, gradient, state, rate):
            delta, stats = direction(gradient, state, self.config.optimizer)
            values = []
            for (_, shape), (start, stop) in zip(self.decoder.layout, self.decoder.offsets):
                decay = self.config.weight_decay if len(shape) >= 2 else 0.
                values.append(parameters[start:stop] * (1 - rate * decay) + rate * delta[start:stop])
            parameters = jnp.concatenate(values)
            stats["parameters_finite"] = jnp.all(jnp.isfinite(parameters))
            return parameters, state, stats

        return tensor_update

    def restore(self, arrays, metadata):
        if metadata.get("optimizer_state_format") != "leaves-v1":
            if not self.tomography:
                raise ValueError("Checkpoint optimizer state format differs")
            tree = TomographyState(*(arrays[n] for n in ("basis", "inverse", "count", "refresh_step")))
            leaves, structure = jax.tree.flatten(tree)
        else:
            shapes, structure = jax.tree.flatten(self.state_shapes)
            if metadata["optimizer_leaf_count"] != len(shapes):
                raise ValueError("Checkpoint optimizer leaf count mismatch")
            leaves = []
            shardings = jax.tree.broadcast(self.state_sharding, self.state_shapes)
            specs = jax.tree.leaves(shardings)
            for i, (shape, sharding) in enumerate(zip(shapes, specs)):
                name = f"optimizer_{i:04d}"
                sharded = metadata.get("sharded_arrays", {}).get(name)
                if sharded is None:
                    leaves.append(arrays[name])
                    continue
                if tuple(sharded["shape"]) != shape.shape or np.dtype(sharded["dtype"]) != shape.dtype:
                    raise ValueError("Checkpoint sharded optimizer shape/dtype mismatch")
                pieces = {json.dumps(s["index"]): arrays[s["array"]] for s in sharded["shards"]}

                def callback(index, pieces=pieces, dimensions=shape.shape):
                    from .checkpoint import encode_index
                    key = json.dumps(encode_index(index, dimensions))
                    if key not in pieces:
                        raise ValueError("Checkpoint does not contain this worker's optimizer shard")
                    return pieces[key]

                leaves.append(jax.make_array_from_callback(shape.shape, sharding, callback))
        expected, _ = jax.tree.flatten(self.state_shapes)
        if len(expected) != len(leaves) or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(leaves, expected)):
            raise ValueError("Checkpoint optimizer tensor shape/dtype mismatch")
        tree = jax.tree.unflatten(structure, leaves)
        return jax.device_put(tree, self.state_sharding)
