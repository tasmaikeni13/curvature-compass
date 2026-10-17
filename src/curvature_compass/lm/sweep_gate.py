"""Bind launches to a frozen plan and real full-shape smokes."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import jax
from .artifacts import atomic_json, sha256, source_hashes
from .calibrate import load_prior
from .competitors import BaselineConfig, empty_baseline_state
from .config import digest
from .data import verify_data
from .experiments import optimizer_name, run_from_dict
from .model import Decoder
from .optim import empty_state

ENVIRONMENT_KEYS = ("backend", "device_kind", "process_count", "local_device_count", "jax", "jaxlib")
REQUIRED_CPU_TESTS = ("test_adamw_matches_independent_bias_corrected_recurrence",
    "test_muon_matches_numpy_matrix_oracle_and_auxiliary_adam",
    "test_soap_projected_adam_and_reference_initialization",
    "test_bartlett_diagonal_matches_exact_ggn_in_expectation",
    "test_baseline_training_bitwise_recovery", "test_sharded_tomography_checkpoint_roundtrip",
    "test_sweep_plan_controls_and_three_seed_coverage", "test_selection_rejects_smoke_and_incomplete_trials")


def state_shape_key(config):
    decoder = Decoder(config.model)
    state = jax.eval_shape(lambda: empty_baseline_state(decoder, config.optimizer)
        if isinstance(config.optimizer, BaselineConfig) else empty_state(decoder.parameter_count, config.optimizer.cap))
    return digest({"optimizer": optimizer_name(config), "model": config.to_dict()["model"],
        "state": [(a.shape, str(a.dtype)) for a in jax.tree.leaves(state)],
        "batch": config.global_batch_size, "accumulation": config.accumulation_steps,
        "curvature_batch": config.curvature_batch_size, "curvature_length": config.curvature_sequence_length,
        "ns_steps": getattr(config.optimizer, "ns_steps", None)})


def requirements(plan_path):
    from .sweep import load_plan
    plan = load_plan(plan_path, current_source=True)
    unique = {}
    for entry in plan["entries"]:
        key = state_shape_key(run_from_dict(entry["config"]))
        if key not in unique:
            unique[key] = {"state_shape_key": key, "trial_id": entry["id"], "optimizer": entry["optimizer"],
                "config": str(Path(plan_path).parent / "configs" / (entry["id"] + ".json"))}
    return list(unique.values())


def create(plan_path, data, smokes, cpu_tests, output, prior=None):
    from .sweep import load_plan
    plan_path, cpu_tests = Path(plan_path), Path(cpu_tests)
    plan = load_plan(plan_path, current_source=True)
    tree = ET.parse(cpu_tests)
    names = [t.attrib["name"] for t in tree.findall(".//testcase")]
    if any(tree.findall(".//" + tag) for tag in ("failure", "error", "skipped")) or any(not any(n.startswith(required) for n in names) for required in REQUIRED_CPU_TESTS):
        raise ValueError("CPU correctness/recovery/workflow coverage is missing or failed")
    manifest = verify_data(data, full=True)
    if not manifest["complete_source_scan"] or manifest["predicted_tokens"]["train"] < plan["entries"][0]["config"]["token_budget"]:
        raise ValueError("Gate requires the real prepared corpus with sufficient nonrepeated tokens")
    data_hash = sha256(Path(data) / "manifest.json")
    evidence = {str(plan_path): sha256(plan_path), str(cpu_tests): sha256(cpu_tests)}
    covered, env, prior_hashes = set(), None, {}
    configs = {e["config_hash"]: e for e in plan["entries"]}
    for directory in map(Path, smokes):
        run_path, summary_path = directory / "run-worker-0.json", directory / "summary-worker-0.json"
        run, summary = json.loads(run_path.read_text()), json.loads(summary_path.read_text())
        config = run_from_dict(run["config"])
        if config.fingerprint not in configs or summary["config_hash"] != config.fingerprint:
            raise ValueError("Full-shape smoke is not a registered sweep config")
        if run["source_sha256"] != plan["source_sha256"] or summary["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Full-shape smoke numerical source differs from the plan")
        if run["mode"] != "smoke" or summary["mode"] != "smoke" or summary["status"] != "finished":
            raise ValueError("Gate requires successful bounded smoke outcomes")
        if summary["parameter_count"] != 596_993_280 or not 2 <= summary["step"] <= 8 or run["recovery"] is None:
            raise ValueError("Full-shape smoke must exercise at least two steps and compatible checkpoint recovery")
        if summary["tokens"] != summary["step"] * config.global_batch_size * config.model.sequence_length:
            raise ValueError("Full-shape smoke token accounting differs")
        if run["data_hash"] != data_hash or summary["data_manifest_sha256"] != data_hash:
            raise ValueError("Full-shape smoke data differs")
        fixed = optimizer_name(config) == "tomography" and not config.optimizer.adaptive_alignment
        prior_hash = None
        if fixed:
            if prior is None:
                raise ValueError("Fixed-alignment gate requires the actual 600M PCA prior")
            frame, record = load_prior(prior, Decoder(config.model), config.optimizer.cap, data_hash)
            if record["seed"] in [e["config"]["seed"] for e in plan["entries"]] or record.get("data_split") != "calibration":
                raise ValueError("600M PCA prior is not independent calibration evidence")
            del frame
            prior_hash = sha256(prior)
        if run["prior_hash"] != prior_hash or summary["prior_sha256"] != prior_hash:
            raise ValueError("Full-shape smoke prior differs")
        prior_hashes[optimizer_name(config)] = prior_hash
        runtime = {k: run["environment"][k] for k in ENVIRONMENT_KEYS}
        if runtime["backend"] != "tpu" or runtime["process_count"] != 4 or runtime["local_device_count"] != 4:
            raise ValueError("600M pod gate requires full-shape evidence on the intended four-worker TPU runtime")
        if env is None:
            env = runtime
        elif env != runtime:
            raise ValueError("Full-shape smoke environments differ")
        for worker in range(runtime["process_count"]):
            path = directory / f"summary-worker-{worker}.json"
            worker_summary = json.loads(path.read_text())
            for key in ("mode", "status", "config_hash", "step", "tokens", "parameter_count", "source_sha256", "data_manifest_sha256", "prior_sha256"):
                if worker_summary[key] != summary[key]:
                    raise ValueError("Full-shape smoke workers disagree")
            evidence[str(path)] = sha256(path)
        metrics = directory / "metrics-worker-0.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if len(rows) != summary["step"] or any(not r["parameters_finite"] or not r.get("finite", True) for r in rows):
            raise ValueError("Full-shape smoke logs are missing or nonfinite")
        if not summary["evaluations"] or summary["evaluations"][-1]["step"] != summary["step"]:
            raise ValueError("Full-shape smoke did not exercise validation")
        for path in (run_path, metrics):
            evidence[str(path)] = sha256(path)
        covered.add(state_shape_key(config))
    needed = {r["state_shape_key"] for r in requirements(plan_path)}
    if not needed <= covered:
        raise ValueError("Missing full-shape smoke/recovery evidence for optimizer state shapes: " + str(len(needed - covered)))
    record = {"schema": 1, "status": "passed", "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path), "validated_config_hashes": sorted(configs),
        "source_sha256": source_hashes(), "data_manifest_sha256": data_hash,
        "prior_sha256_by_optimizer": prior_hashes, "environment": env, "covered_state_shapes": sorted(covered),
        "evidence_sha256": evidence,
        "scope": "CPU algorithm/recovery correctness and bounded actual 600M-shape TPU smoke/recovery for every state shape in this exact plan. This is not sweep or long-run stability evidence."}
    atomic_json(output, record)
    return record


def validate_sweep_gate(path, config, data_hash, prior_hash, environment):
    from .sweep import load_plan
    gate = json.loads(Path(path).read_text())
    if gate.get("schema") != 1 or gate.get("status") != "passed":
        raise ValueError("Gate requires a successful full-shape smoke validation")
    plan = load_plan(gate["plan_path"], current_source=True)
    if sha256(gate["plan_path"]) != gate["plan_sha256"] or config.fingerprint not in gate["validated_config_hashes"]:
        raise ValueError("Sweep controls differ from the gated plan")
    if config.fingerprint not in {e["config_hash"] for e in plan["entries"]}:
        raise ValueError("Configuration is absent from the gated sweep plan")
    if gate["source_sha256"] != source_hashes() or gate["data_manifest_sha256"] != data_hash:
        raise ValueError("Sweep source or data changed since smoke validation")
    if gate["prior_sha256_by_optimizer"].get(optimizer_name(config), "missing") != prior_hash:
        raise ValueError("Sweep prior differs from validated inputs")
    if state_shape_key(config) not in gate.get("covered_state_shapes", []):
        raise ValueError("This optimizer state shape has no full-shape validation")
    if any(gate["environment"][k] != environment[k] for k in ENVIRONMENT_KEYS):
        raise ValueError("Sweep runtime differs from the validated topology/numerical environment")
    if not gate.get("evidence_sha256"):
        raise ValueError("Sweep gate is missing its validation evidence")
    for name, expected in gate["evidence_sha256"].items():
        if sha256(name) != expected:
            raise ValueError("Sweep validation evidence changed: " + name)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    required = commands.add_parser("requirements")
    required.add_argument("--plan", required=True, type=Path)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--plan", required=True, type=Path)
    create_parser.add_argument("--data", required=True, type=Path)
    create_parser.add_argument("--prior", type=Path)
    create_parser.add_argument("--smokes", required=True, nargs="+", type=Path)
    create_parser.add_argument("--cpu-tests", type=Path, default=Path("analysis/results/cpu_tests.xml"))
    create_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "requirements":
        print(json.dumps(requirements(args.plan), indent=2))
    else:
        gate = create(args.plan, args.data, args.smokes, args.cpu_tests, args.output, args.prior)
        print(json.dumps({"status": gate["status"], "scope": gate["scope"]}, indent=2))


if __name__ == "__main__":
    main()
