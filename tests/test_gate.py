import json
from dataclasses import replace
import pytest
from curvature_compass.lm.artifacts import atomic_json, sha256, source_hashes
from curvature_compass.lm.config import RunConfig
from curvature_compass.lm.gate import REQUIRED_CHECKS, validate_gate


@pytest.mark.parametrize("mutation", ["status", "coverage", "data", "source", "environment", "evidence", "config"])
def test_full_run_gate_rejects_unvalidated_or_changed_experiments(tmp_path, mutation):
    config = RunConfig()
    evidence = tmp_path / "evidence.json"; evidence.write_text('{"fixture_only": true}\n')
    env = {"backend": "tpu", "device_kind": "TPU v4", "process_count": 4, "local_device_count": 4,
           "jax": "0.6.2", "jaxlib": "0.6.2"}
    gate = {"schema": 1, "status": "passed", "checks": dict.fromkeys(REQUIRED_CHECKS, True),
            "validated_config_hashes": [config.fingerprint], "data_manifest_sha256": "data",
            "prior_sha256": "prior", "source_sha256": source_hashes(), "environment": env.copy(),
            "evidence_sha256": {str(evidence): sha256(evidence)}}
    path = tmp_path / "gate.json"; atomic_json(path, gate)
    validate_gate(path, config, "data", "prior", env)
    if mutation == "status": gate["status"] = "pending"
    elif mutation == "coverage": gate["checks"].pop("full_shape_benchmarks")
    elif mutation == "data": gate["data_manifest_sha256"] = "other-data"
    elif mutation == "source": gate["source_sha256"] = {}
    elif mutation == "environment": gate["environment"]["process_count"] = 1
    elif mutation == "evidence": evidence.write_text("changed")
    elif mutation == "config": config = replace(config, seed=1)
    atomic_json(path, gate)
    with pytest.raises(ValueError):
        validate_gate(path, config, "data", "prior", env)
