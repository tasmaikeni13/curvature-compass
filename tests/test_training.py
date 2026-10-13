from dataclasses import replace
import json
import numpy as np
import pytest
from curvature_compass.lm import checkpoint
from curvature_compass.lm.artifacts import atomic_json, sha256
from curvature_compass.lm.calibrate import fit_pca
from curvature_compass.lm.config import ModelConfig, OptimizerConfig, RunConfig
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.train import execute


def controls_and_prior(tmp_path, fixture_data, cell="D"):
    config = RunConfig(model=ModelConfig(width=8, layers=1, heads=2, mlp_width=16, sequence_length=8),
                       optimizer=OptimizerConfig(configuration=cell, cap=2, refresh_interval=2),
                       global_batch_size=2, curvature_batch_size=1, curvature_sequence_length=4,
                       token_budget=49, validation_tokens=17, warmup_steps=1,
                       eval_interval=2, checkpoint_interval=2)
    model = Decoder(config.model)
    rows = np.random.default_rng(15).normal(size=(4, model.parameter_count)).astype(np.float32)
    basis, _ = fit_pca(rows, 2)
    path = tmp_path / "prior.npy"; np.save(path, basis)
    atomic_json(path.with_suffix(".json"), {"schema": 1, "model_fingerprint": model.fingerprint, "rank": 2,
                "data_manifest_sha256": sha256(fixture_data / "manifest.json"), "basis_sha256": sha256(path),
                "seed": 30900, "method": "Synthetic orthonormal test frame; not a research prior"})
    return config, path


def test_bitwise_resumed_training_and_exact_partial_final_budget(tmp_path, fixture_data):
    config, prior = controls_and_prior(tmp_path, fixture_data)
    complete = tmp_path / "complete"
    resumed = tmp_path / "resumed"
    first = execute(config, fixture_data, prior, complete, smoke_steps=4)
    execute(config, fixture_data, prior, resumed, smoke_steps=2)
    second = execute(config, fixture_data, prior, resumed, smoke_steps=4, resume=True)
    args = (config.fingerprint, sha256(fixture_data / "manifest.json"), sha256(prior))
    a, _, _ = checkpoint.load(complete / "checkpoints", *args)
    b, _, _ = checkpoint.load(resumed / "checkpoints", *args)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert first["tokens"] == second["tokens"] == 49
    assert first["complete_budget"] and second["complete_budget"]
    assert first["total_hvps"] == second["total_hvps"]
    assert [r["step"] for r in first["evaluations"]] == [r["step"] for r in second["evaluations"]]
    rows = [json.loads(line) for line in (resumed / "metrics-worker-0.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert sum(r["predicted_tokens"] for r in rows) == 49
    assert all(r.get("evaluation", {}).get("validation_tokens", 17) == 17 for r in rows)


@pytest.mark.parametrize("cell", list("ABCD"))
def test_all_cells_execute_real_model_hvps_and_cached_updates(tmp_path, fixture_data, cell):
    config, prior = controls_and_prior(tmp_path, fixture_data, cell)
    config = replace(config, token_budget=64)
    summary = execute(config, fixture_data, prior, tmp_path / cell, smoke_steps=2)
    assert summary["status"] == "finished" and summary["tokens"] == 32
    rows = [json.loads(line) for line in (tmp_path / cell / "metrics-worker-0.jsonl").read_text().splitlines()]
    assert rows[0]["refresh"] and not rows[1]["refresh"]
    assert rows[0]["hvp_count"] >= 1 and rows[1]["hvp_count"] == 0
    assert all(r["parameters_finite"] and r["direction_finite_and_descent"] for r in rows)
    if cell in "AC":
        assert not rows[0]["rank_exhausted"]


def test_full_run_and_smoke_bound_guard(tmp_path, fixture_data):
    config, prior = controls_and_prior(tmp_path, fixture_data)
    with pytest.raises(ValueError, match="Full training requires"):
        execute(config, fixture_data, prior, tmp_path / "forbidden")
    with pytest.raises(ValueError, match="limited"):
        execute(config, fixture_data, prior, tmp_path / "too-long", smoke_steps=9)
