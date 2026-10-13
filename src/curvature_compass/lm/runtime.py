"""Explicit multihost startup and globally sharded batches."""
import os
from pathlib import Path
import numpy as np


def initialize(distributed=False, coordinator=None, processes=None, process_id=None):
    import jax
    if distributed:
        if coordinator is None:
            jax.distributed.initialize(initialization_timeout=120)
        else:
            if processes is None or process_id is None:
                raise ValueError("Explicit coordinator requires process count and process id")
            jax.distributed.initialize(coordinator_address=coordinator, num_processes=processes,
                                       process_id=process_id, initialization_timeout=120)
    if not os.environ.get("COMPASS_DISABLE_JAX_CACHE"):
        jax.config.update("jax_compilation_cache_dir", str(Path(".jax_cache").resolve()))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    return jax


def mesh_and_shardings():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.array(jax.devices()), ("data",))
    return mesh, NamedSharding(mesh, P()), NamedSharding(mesh, P("data", None))


def global_batch(reader, cursor, size, sharding, prefix_length=None):
    import jax
    if size % jax.device_count() or size % jax.process_count():
        raise ValueError("Global batch size must divide the full data mesh")
    local_size = size // jax.process_count()
    local = reader.batch(cursor, size, jax.process_index() * local_size, local_size, prefix_length)
    shape = (size, local[0].shape[1])
    return tuple(jax.make_array_from_process_local_data(sharding, a, global_shape=shape) for a in local)


def host_array(value):
    """Only for replicated outputs; never gather a sharded global minibatch."""
    import jax
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        if not value.is_fully_replicated:
            raise ValueError("Cannot fetch non-addressable, non-replicated data")
        return np.asarray(value.addressable_shards[0].data)
    return np.asarray(value)


def host_stats(stats):
    result = {}
    for key, value in stats.items():
        array = host_array(value)
        result[key] = array.tolist() if array.ndim else array.item()
    return result


def sync(label):
    import jax
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(label)


def any_host_stop(requested):
    import jax
    if jax.process_count() == 1:
        return requested
    from jax.experimental import multihost_utils
    return bool(np.any(multihost_utils.process_allgather(np.array(requested, dtype=np.int32))))
