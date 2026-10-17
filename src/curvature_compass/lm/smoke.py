"""Bounded small-model, real-data update and bitwise recovery checks."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from . import checkpoint
from .artifacts import atomic_json, environment, sha256, source_hashes
from .config import ModelConfig
from .experiments import load_run_config
from .runtime import initialize, sync
from .sweep import METHODS
from .train import execute


def smoke(data, output, evidence=None):
    import jax
    if Path(output).exists() and any(Path(output).iterdir()):
        raise ValueError("Smoke checks require a fresh output directory")
    batch = max(2, jax.device_count())
    model = ModelConfig(width=16, layers=1, heads=2, mlp_width=32, sequence_length=8)
    results = {}
    for method in METHODS:
        template = load_run_config(f"configs/lm125m_{method}.json")
        opt = template.optimizer
        if method == "tomography":
            opt = replace(opt, configuration="D", cap=2, refresh_interval=2)
        else:
            opt = replace(opt, precondition_frequency=1, hessian_interval=2)
        config = replace(template, model=model, optimizer=opt, global_batch_size=batch, curvature_batch_size=batch,
            curvature_sequence_length=8, token_budget=2500 * batch * 8, validation_tokens=128,
            checkpoint_interval=2, eval_interval=2, learning_rate=.0003, warmup_steps=1)
        directory = Path(output) / method
        reference = execute(config, data, None, directory / "reference", smoke_steps=3, smoke_validation_tokens=128)
        execute(config, data, None, directory / "resumed", smoke_steps=1, smoke_validation_tokens=128)
        resumed = execute(config, data, None, directory / "resumed", smoke_steps=3, resume=True, smoke_validation_tokens=128)
        data_hash = sha256(Path(data) / "manifest.json")
        a, _, _ = checkpoint.load(directory / "reference/checkpoints", config.fingerprint, data_hash, None, source_hashes())
        b, _, _ = checkpoint.load(directory / "resumed/checkpoints", config.fingerprint, data_hash, None, source_hashes())
        if a.keys() != b.keys() or any(not np.array_equal(a[key], b[key]) for key in a):
            raise ValueError("Bounded recovery was not bitwise identical: " + method)
        if reference["tokens"] != resumed["tokens"] or reference["total_hvps"] != resumed["total_hvps"]:
            raise ValueError("Bounded recovery changed token/HVP accounting")
        results[method] = {"passed": True, "bitwise_checkpoint_continuation": True,
            "parameter_count": reference["parameter_count"], "step": reference["step"],
            "tokens": reference["tokens"], "total_hvps": reference["total_hvps"],
            "compiled_memory": json.loads((directory / "reference" / f"run-worker-{jax.process_index()}.json").read_text())["compilation"],
            "reference_checkpoint_sha256": sha256(directory / "reference/checkpoints/step-00000003/state.npz"),
            "resumed_checkpoint_sha256": sha256(directory / "resumed/checkpoints/step-00000003/state.npz")}
        print(json.dumps({"optimizer": method, "smoke_and_bitwise_recovery": "passed"}), flush=True)
    record = {"schema": 1, "status": "passed", "mode": "small_model_smoke",
        "checks": results, "source_sha256": source_hashes(), "environment": environment(),
        "data_manifest_sha256": sha256(Path(data) / "manifest.json"),
        "scope": "Five tiny-model real-data JAX/XLA update/validation/checkpoint runs and bitwise continuation. No 600M compilation, performance, sweep or full comparison claim."}
    path = Path(evidence) if evidence else Path(output) / "evidence"
    atomic_json(path.with_name(path.name + f"-worker-{jax.process_index()}.json"), record)
    sync("smoke-checks-complete")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    args = parser.parse_args()
    initialize(args.distributed, args.coordinator, args.processes, args.process_id)
    smoke(args.data, args.output, args.evidence)


if __name__ == "__main__":
    main()
