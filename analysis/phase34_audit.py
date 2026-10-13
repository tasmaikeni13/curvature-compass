"""Audit implementation delivery evidence without launching any workload."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from curvature_compass.lm.artifacts import atomic_json, sha256, source_hashes
from curvature_compass.lm.config import load_config
from curvature_compass.lm.experiments import load_run_config
from curvature_compass.lm.gate import validate_gate
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.sweep import METHODS, SEEDS


def audit():
    current, evidence, checks = source_hashes(), {}, {}
    for cell in "ABCD":
        name = f"configs/lm60m_{cell}.json"
        # The original frozen controls are available in the delivery's base commit.
        frozen = subprocess.check_output(["git", "show", "5caa86e:" + name])
        assert Path(name).read_bytes() == frozen, "60M controls changed"
    checks["frozen_phase3_controls"] = True
    for method in METHODS:
        config = load_run_config(f"configs/lm600m_{method}.json")
        assert Decoder(config.model).parameter_count == 596_993_280
        assert config.training_steps == 2500 and config.token_budget == 163_840_000
        assert config.curvature_sequence_length == config.model.sequence_length == 1024
    assert len(SEEDS) == 3 and len(set(SEEDS)) == 3
    checks["five_600m_templates_and_exact_budgets"] = True

    xml = Path("analysis/results/phase23/cpu_tests.xml")
    tree = ET.parse(xml)
    names = [case.attrib["name"] for case in tree.findall(".//testcase")]
    assert len(names) == 53 and not any(tree.findall(".//" + tag) for tag in ("failure", "error", "skipped"))
    required = ("test_all_cells_execute_real_model_hvps_and_cached_updates", "test_baseline_training_bitwise_recovery",
        "test_sharded_tomography_checkpoint_roundtrip", "test_phase4_tensor_decay_matches_independent_dense_mask_rule",
        "test_600m_update_traces_without_allocating_a_parameter_sized_decay_mask",
        "test_sophia_full_context_measures_every_position_and_rejects_short_sweep_context",
        "test_sweep_plan_controls_and_three_seed_coverage", "test_selection_rejects_smoke_and_incomplete_trials",
        "test_sweep_coordinator_collects_all_worker_summaries_without_checkpoints")
    assert all(any(name.startswith(prefix) for name in names) for prefix in required)
    evidence[str(xml)] = sha256(xml)
    checks["cpu_53_checks_with_required_coverage"] = True

    data_hashes = set()
    for worker in range(4):
        path = Path(f"analysis/results/phase34/tpu_smoke-worker-{worker}.json")
        record = json.loads(path.read_text())
        assert record["status"] == "passed" and record["mode"] == "small_model_smoke"
        assert record["source_sha256"] == current, "Final TPU smoke source differs from delivery"
        assert record["environment"]["backend"] == "tpu" and record["environment"]["process_count"] == 4
        assert record["environment"]["local_device_count"] == 4 and len(record["environment"]["devices"]) == 16
        assert set(record["checks"]) == set(METHODS)
        for case in record["checks"].values():
            assert case["passed"] and case["bitwise_checkpoint_continuation"]
            assert case["parameter_count"] == 807_248 and case["step"] == 3 and case["tokens"] == 384
        data_hashes.add(record["data_manifest_sha256"])
        evidence[str(path)] = sha256(path)
    assert len(data_hashes) == 1
    checks["all_five_actual_tpu_smokes_and_bitwise_recovery_on_four_workers"] = True

    gate_path = Path("analysis/results/phase23/gate.json")
    gate = json.loads(gate_path.read_text())
    assert gate["data_manifest_sha256"] in data_hashes
    for cell in "ABCD":
        validate_gate(gate_path, load_config(f"configs/lm60m_{cell}.json"), gate["data_manifest_sha256"], gate["prior_sha256"], gate["environment"])
    evidence[str(gate_path)] = sha256(gate_path)
    checks["phase3_preparation_gate_current_and_bound"] = True
    for path in Path("runs").glob("**/run-worker-*.json"):
        run = json.loads(path.read_text())
        assert run["mode"] == "smoke" and run["parameter_count"] < 1_000_000
    checks["workspace_runs_are_bounded_tiny_smokes_only"] = True
    for path in (Path("analysis/phase34_protocol.md"), Path("analysis/phase34_results.md"),
                 Path("analysis/phase34_launch.md"), Path("analysis/phase34_audit.py")):
        evidence[str(path)] = sha256(path)
    for path in Path("analysis/results/phase34").glob("*logs/worker-*.log"):
        evidence[str(path)] = sha256(path)
    result = {"schema": 1, "status": "passed", "scope": "Phase 3/4 software implementation and bounded smoke delivery only; not completion of training/sweeps or validation of 600M execution.",
        "checks": checks, "source_sha256": current, "evidence_sha256": evidence,
        "not_executed": ["60M comparisons", "600M full-shape smokes", "2,500-step sweeps", "125M main runs"],
        "remaining_for_user": "Execute actual 600M-shape smoke/recovery, create the separate sweep gate, then run the registered experiments."}
    atomic_json("analysis/results/phase34/delivery_audit.json", result)
    print(json.dumps({"status": "passed", "checks": checks}, indent=2))
    return result


if __name__ == "__main__":
    audit()
