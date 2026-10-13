"""Bind a future full-run launch to actual infrastructure validation artifacts."""
import json
from pathlib import Path
from .artifacts import sha256, source_hashes

REQUIRED_CHECKS = ("data_integrity", "four_controls", "cpu_tests", "tpu_numerics",
                   "full_shape_benchmarks", "all_four_implementations", "checkpoint_recovery",
                   "cost_accounting", "selected_acceleration")


def validate_gate(path, config, data_hash, prior_hash, environment):
    gate = json.loads(Path(path).read_text())
    if gate.get("schema") != 1 or gate.get("status") != "passed":
        raise ValueError("Infrastructure has not passed its correctness/performance gate")
    if any(gate.get("checks", {}).get(name) is not True for name in REQUIRED_CHECKS) or not gate.get("evidence_sha256"):
        raise ValueError("Infrastructure validation evidence/check coverage is incomplete")
    if config.fingerprint not in gate.get("validated_config_hashes", []):
        raise ValueError("This experiment configuration was not validated")
    if gate["data_manifest_sha256"] != data_hash or gate["prior_sha256"] != prior_hash:
        raise ValueError("Data/prior changed since infrastructure validation")
    if gate["source_sha256"] != source_hashes():
        raise ValueError("Source/config changed since infrastructure validation")
    for name in ("backend", "device_kind", "process_count", "local_device_count", "jax", "jaxlib"):
        if gate["environment"][name] != environment[name]:
            raise ValueError(f"Execution environment differs from validated {name}")
    for name, digest in gate["evidence_sha256"].items():
        if sha256(name) != digest:
            raise ValueError(f"Validation evidence changed: {name}")
    return gate
