import json
from pathlib import Path
import numpy as np
import pytest

from curvature_compass.lm.config import ModelConfig, RunConfig
from curvature_compass.lm.competitors import BaselineConfig
from curvature_compass.lm.experiments import load_run_config
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.train import execute


def test_125m_parameter_count_matches_canonical():
    """Verify default 125M LLaMA configuration produces exactly 124,275,456 parameters."""
    config = ModelConfig()
    decoder = Decoder(config)
    assert decoder.parameter_count == 124_275_456


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon", "soap", "sophia_g"])
def test_smoke_steps_verifies_initialization_gradients_and_loss_descent(tmp_path, fixture_data, optimizer_name):
    """
    Verify --smoke-steps runs bounded steps (2 steps), verifies parameter initialization,
    gradient calculation, and checks loss descent on CPU.
    """
    model_cfg = ModelConfig(vocab_size=32000, padded_vocab_size=32768, width=16, layers=2, heads=2, mlp_width=32, sequence_length=8)
    opt_cfg = BaselineConfig(name=optimizer_name, precondition_frequency=1, hessian_interval=2)
    run_cfg = RunConfig(
        model=model_cfg,
        optimizer=opt_cfg,
        seed=41001,
        data_seed=42001,
        global_batch_size=2,
        curvature_batch_size=2,
        curvature_sequence_length=8,
        token_budget=64,
        validation_tokens=16,
        learning_rate=0.01,
        warmup_steps=1,
        eval_interval=2,
        checkpoint_interval=2,
        compute_dtype="float32",
        attention="reference"
    )

    out_dir = tmp_path / optimizer_name
    summary = execute(run_cfg, fixture_data, None, out_dir, smoke_steps=2, smoke_validation_tokens=16)

    assert summary["status"] == "finished"
    assert summary["step"] == 2
    assert summary["smoke_verification"]["parameter_initialization_finite"]
    assert summary["smoke_verification"]["bias_free"]
    assert summary["smoke_verification"]["rmsnorm_unit_initialized"]
    assert summary["smoke_verification"]["gradient_norm_finite"]
    assert summary["smoke_verification"]["gradient_norm_positive"]
    assert summary["smoke_verification"]["loss_finite"]
    assert summary["smoke_verification"]["loss_descent"]
    assert summary["smoke_verification"]["loss_drop"] > 0

    metrics = [json.loads(line) for line in (out_dir / "metrics-worker-0.jsonl").read_text().splitlines()]
    assert len(metrics) == 2
    assert all(r["parameters_finite"] for r in metrics)


def test_smoke_steps_bound_guard(tmp_path, fixture_data):
    """Verify smoke_steps outside [1, 8] is strictly rejected."""
    model_cfg = ModelConfig(vocab_size=32000, padded_vocab_size=32768, width=16, layers=1, heads=2, mlp_width=32, sequence_length=8)
    run_cfg = RunConfig(model=model_cfg, token_budget=64)
    with pytest.raises(ValueError, match="limited to 1–8 steps"):
        execute(run_cfg, fixture_data, None, tmp_path / "zero", smoke_steps=0)
    with pytest.raises(ValueError, match="limited to 1–8 steps"):
        execute(run_cfg, fixture_data, None, tmp_path / "nine", smoke_steps=9)


def test_full_run_without_gate_rejected(tmp_path, fixture_data):
    """Verify full comparison run without gate is rejected."""
    model_cfg = ModelConfig(vocab_size=32000, padded_vocab_size=32768, width=16, layers=1, heads=2, mlp_width=32, sequence_length=8)
    run_cfg = RunConfig(model=model_cfg, token_budget=64)
    with pytest.raises(ValueError, match="Full training requires"):
        execute(run_cfg, fixture_data, None, tmp_path / "no_gate", smoke_steps=None, gate=None)


def test_cpu_attention_fallback_in_smoke_mode(tmp_path, fixture_data):
    """Verify that specifying attention='flash' or 'pallas' on CPU gracefully falls back without crashing."""
    model_cfg = ModelConfig(vocab_size=32000, padded_vocab_size=32768, width=16, layers=1, heads=2, mlp_width=32, sequence_length=8)
    run_cfg = RunConfig(
        model=model_cfg, optimizer=BaselineConfig(name="adamw"), attention="flash", global_batch_size=2,
        curvature_batch_size=2, curvature_sequence_length=8,
        token_budget=32, validation_tokens=16
    )
    out_dir = tmp_path / "flash_fallback"
    summary = execute(run_cfg, fixture_data, None, out_dir, smoke_steps=1, smoke_validation_tokens=16)
    assert summary["status"] == "finished"

