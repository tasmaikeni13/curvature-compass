from dataclasses import replace
import json
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pytest
from curvature_compass.lm import checkpoint
from curvature_compass.lm.artifacts import sha256
from curvature_compass.lm.competitors import BASELINES, BaselineConfig
from curvature_compass.lm.config import ModelConfig, RunConfig
from curvature_compass.lm.train import execute


def test_phase4_tensor_decay_matches_independent_dense_mask_rule():
    import jax
    import jax.numpy as jnp
    from curvature_compass.lm.config import OptimizerConfig
    from curvature_compass.lm.experiments import SweepRunConfig
    from curvature_compass.lm.execution import OptimizerExecution
    from curvature_compass.lm.model import Decoder
    from curvature_compass.lm.optim import acquire, direction
    from curvature_compass.lm.runtime import mesh_and_shardings
    config = SweepRunConfig(model=ModelConfig(vocab_size=11, padded_vocab_size=16, width=8, layers=1,
        heads=2, mlp_width=16, sequence_length=8), optimizer=OptimizerConfig(configuration="D", cap=2),
        global_batch_size=2, curvature_batch_size=2, curvature_sequence_length=4,
        token_budget=40000, compute_dtype="bfloat16", weight_decay=.17)
    decoder = Decoder(config.model)
    mesh, rep, _ = mesh_and_shardings()
    execution = OptimizerExecution(decoder, config, mesh, rep)
    gradient = np.random.default_rng(8).normal(size=decoder.parameter_count).astype(np.float32)
    state, _ = acquire(lambda v: 2 * v, jnp.asarray(gradient), jnp.zeros((1, 1)), config.optimizer)
    initial = decoder.initialize(2)
    delta, _ = direction(jnp.asarray(gradient), state, config.optimizer)
    expected = initial * (1 - .003 * config.weight_decay * decoder.decay_mask()) + .003 * np.asarray(delta)
    actual, _, stats = jax.jit(execution.update_function())(jnp.asarray(initial), jnp.asarray(gradient), state, .003)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
    assert bool(stats["parameters_finite"]) and bool(stats["direction_finite_and_descent"])


def test_600m_update_traces_without_allocating_a_parameter_sized_decay_mask(monkeypatch):
    import jax
    import jax.numpy as jnp
    from curvature_compass.lm.experiments import load_run_config
    from curvature_compass.lm.execution import OptimizerExecution
    from curvature_compass.lm.model import Decoder
    from curvature_compass.lm.runtime import mesh_and_shardings
    config = load_run_config("configs/lm600m_tomography.json")
    decoder = Decoder(config.model)
    monkeypatch.setattr(decoder, "decay_mask", lambda: (_ for _ in ()).throw(AssertionError("600M mask allocated")))
    mesh, rep, _ = mesh_and_shardings()
    execution = OptimizerExecution(decoder, config, mesh, rep)
    vector = jax.ShapeDtypeStruct((decoder.parameter_count,), jnp.float32)
    result = jax.eval_shape(execution.update_function(), vector, vector, execution.state_shapes,
                            jax.ShapeDtypeStruct((), jnp.float32))
    assert result[0].shape == (596993280,) and result[0].dtype == jnp.float32


@pytest.mark.parametrize("method", BASELINES)
def test_baseline_training_bitwise_recovery(tmp_path, fixture_data, method):
    config = RunConfig(model=ModelConfig(width=8, layers=1, heads=2, mlp_width=16, sequence_length=8),
        optimizer=BaselineConfig(name=method, precondition_frequency=1, hessian_interval=2),
        seed=41001, data_seed=42001, global_batch_size=2, curvature_batch_size=1, curvature_sequence_length=4,
        token_budget=49, validation_tokens=17, learning_rate=.0003, warmup_steps=1, eval_interval=2, checkpoint_interval=2,
        compute_dtype="bfloat16")
    whole, split = tmp_path / "whole", tmp_path / "split"
    first = execute(config, fixture_data, None, whole, smoke_steps=4)
    execute(config, fixture_data, None, split, smoke_steps=2)
    second = execute(config, fixture_data, None, split, smoke_steps=4, resume=True)
    args = (config.fingerprint, sha256(fixture_data / "manifest.json"), None)
    a, _, _ = checkpoint.load(whole / "checkpoints", *args)
    b, _, _ = checkpoint.load(split / "checkpoints", *args)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert first["tokens"] == second["tokens"] == 49 and first["complete_budget"] and second["complete_budget"]
    assert first["total_hvps"] == second["total_hvps"] == 0
    rows = [json.loads(line) for line in (split / "metrics-worker-0.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert all(r["parameters_finite"] and r["optimizer_state_finite"] for r in rows)
    assert all(e["validation_tokens"] == 17 for e in second["evaluations"])
    if method == "sophia_g":
        assert [r["sampled_label_gradient_count"] for r in rows if r["refresh"]] == [1, 1]
        assert rows[0]["ggn_predicted_tokens"] == 4
    if method == "soap":
        assert rows[0]["soap_initialization_batch"] and not rows[1]["soap_initialization_batch"]
        assert not any(r["refresh"] for r in rows)


def test_sharded_tomography_checkpoint_roundtrip(tmp_path):
    # A separate runtime gives an actual multi-device partition even on CPU.
    code = '''
import json
from pathlib import Path
import jax
import numpy as np
from dataclasses import replace
from curvature_compass.lm.artifacts import source_hashes
from curvature_compass.lm.config import ModelConfig, OptimizerConfig
from curvature_compass.lm.experiments import SweepRunConfig
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.execution import OptimizerExecution
from curvature_compass.lm.runtime import mesh_and_shardings
from curvature_compass.lm import checkpoint
config = SweepRunConfig(model=ModelConfig(vocab_size=11,padded_vocab_size=16,width=8,layers=1,heads=2,mlp_width=16,sequence_length=8),
    optimizer=OptimizerConfig(configuration="D",cap=2),global_batch_size=2,curvature_batch_size=2,curvature_sequence_length=4,
    token_budget=40000,compute_dtype="bfloat16")
config.validate()
decoder = Decoder(config.model)
mesh, rep, _ = mesh_and_shardings()
assert mesh.size == 2
execution = OptimizerExecution(decoder,config,mesh,rep)
state = execution.empty()._replace(basis=jax.make_array_from_callback((2,decoder.parameter_count),execution.state_sharding.basis,
    lambda i: np.arange(2*decoder.parameter_count,dtype=np.float32).reshape(2,-1)[i]))
assert not state.basis.is_fully_replicated
metadata = {"step":1,"config_hash":config.fingerprint,"data_hash":"d","prior_hash":None,"source_sha256":source_hashes()}
checkpoint.save(Path(__import__('sys').argv[1]),jax.device_put(decoder.initialize(3),rep),state,metadata)
arrays, meta, _ = checkpoint.load(Path(__import__('sys').argv[1]),config.fingerprint,"d",None,source_hashes())
assert meta["optimizer_state_format"] == "leaves-v1" and meta["sharded_arrays"]
restored = execution.restore(arrays,meta)
for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(restored)):
    np.testing.assert_array_equal(a,b)
assert restored.basis.sharding == state.basis.sharding
'''
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
