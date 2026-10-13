from copy import deepcopy
from dataclasses import replace
import json
import fnmatch
from pathlib import Path
import numpy as np
import pytest
from curvature_compass.lm import compare, sweep, sweep_gate
from curvature_compass.lm.artifacts import atomic_json, sha256, source_hashes
from curvature_compass.lm.config import digest, load_config
from curvature_compass.lm.experiments import load_run_config, run_from_dict
from curvature_compass.lm.run_results import read_run


def test_sweep_coordinator_collects_all_worker_summaries_without_checkpoints(tmp_path, monkeypatch):
    calls = []
    ranks = {"w1": 2, "w2": 3, "w3": 1}  # Actual pod evidence showed this ordering.

    def fake_scp(argv, **kwargs):
        calls.append((argv, kwargs))
        host, remote = argv[-2].split(":", 1)
        name = f"summary-worker-{ranks[host]}.json"
        assert fnmatch.fnmatch(name, Path(remote).name)
        atomic_json(Path(argv[-1]) / name, {"fixture_only": True, "worker": ranks[host]})

    monkeypatch.setattr(sweep.subprocess, "run", fake_scp)
    sweep.collect_worker_summaries(tmp_path)
    assert len(calls) == 3
    for worker, (argv, options) in enumerate(calls, start=1):
        assert argv[-2] == f"w{worker}:{tmp_path}/summary-worker-*.json"
        assert argv[-1] == str(tmp_path)
        assert options["check"] and "checkpoints" not in " ".join(argv)
        assert (tmp_path / f"summary-worker-{worker}.json").exists()


@pytest.fixture
def registered_plan(tmp_path, monkeypatch):
    # This fixture isolates plan generation/selection logic; it is not training evidence.
    winner = {"schema": 1, "phase": 3, "status": "frozen", "winner": "C",
              "winner_optimizer": load_config("configs/lm60m_C.json").to_dict()["optimizer"]}
    path = tmp_path / "synthetic-winner.json"
    atomic_json(path, {**winner, "fixture_only": True})
    monkeypatch.setattr(compare, "validate_selection", lambda _: winner)
    root = tmp_path / "plan"
    plan = sweep.generate(path, root, settings=2)
    return root / "plan.json", plan


def test_sweep_plan_controls_and_three_seed_coverage(registered_plan, tmp_path, monkeypatch):
    path, plan = registered_plan
    assert plan["trial_count"] == 30 and plan["total_training_steps"] == 75000
    assert {e["optimizer"] for e in plan["entries"]} == set(sweep.METHODS)
    for method in sweep.METHODS:
        for setting in range(2):
            rows = [e for e in plan["entries"] if e["optimizer"] == method and e["setting"] == setting]
            assert {(e["config"]["seed"], e["config"]["data_seed"]) for e in rows} == set(sweep.SEEDS)
            for e in rows:
                config = load_run_config(path.parent / "configs" / (e["id"] + ".json"))
                assert config.token_budget == 163840000 and config.training_steps == 2500
                if method == "tomography":
                    assert config.optimizer.configuration == "C"
    assert sweep.load_plan(path) == json.loads(path.read_text())
    for mutation in ("missing_seed", "architecture", "winning_cell", "steps", "unmatched_seed"):
        changed = deepcopy(plan)
        if mutation == "missing_seed":
            changed["entries"].pop()
        else:
            entry = changed["entries"][0]
            if mutation == "architecture":
                entry["config"]["model"]["layers"] -= 1
            elif mutation == "winning_cell":
                entry["config"]["optimizer"]["configuration"] = "D"
            elif mutation == "steps":
                entry["config"]["training_steps"] = 2499
            else:
                entry["config"]["data_seed"] += 1
            entry["config_hash"] = digest(entry["config"])
        changed["fingerprint"] = digest({k: v for k, v in changed.items() if k != "fingerprint"})
        with pytest.raises(ValueError):
            sweep.validate_plan(changed)


def test_selection_rejects_smoke_and_incomplete_trials(registered_plan, tmp_path):
    path, plan = registered_plan
    directory = tmp_path / "smoke"
    atomic_json(directory / "run-worker-0.json", {"mode": "smoke", "config": plan["entries"][0]["config"]})
    atomic_json(directory / "summary-worker-0.json", {"mode": "smoke"})
    with pytest.raises(ValueError, match="Smoke"):
        read_run(directory)
    result = sweep.report(path, tmp_path / "no-runs", tmp_path / "progress.json")
    assert not result["terminal"] and not result["selected"]
    assert len(result["trials"]) == 30
    with pytest.raises(ValueError, match="Cannot freeze"):
        sweep.report(path, tmp_path / "no-runs", tmp_path / "forbidden.json", freeze=True)
    assert not (tmp_path / "forbidden.json").exists()


def test_sweep_selection_uses_all_three_seeds_and_preserves_failures(registered_plan, tmp_path, monkeypatch):
    path, plan = registered_plan
    by_hash = {e["config_hash"]: e for e in plan["entries"]}

    def fixture_run(directory, config):
        entry = by_hash[config.fingerprint]
        if entry["optimizer"] == "soap" and entry["setting"] == 1 and entry["seed_index"] == 0:
            raise ValueError("Explicit synthetic stability failure")
        loss = 4. + .002 * entry["setting"] + .01 * (entry["seed_index"] - 1)
        return {"run": {"source_sha256": plan["source_sha256"]}, "final_loss": loss, "perplexity": float(np.exp(loss)),
            "elapsed_seconds": 100. - 10 * entry["setting"], "costs": {},
            "summary": {"total_hvps": 8, "device_memory_stats": []}, "evidence_sha256": {}}

    monkeypatch.setattr(sweep, "read_run", fixture_run)
    failed = tmp_path / "runs/soap-001-0"
    atomic_json(failed / "failure-worker-0.json", {"fixture_only": True, "error_type": "FloatingPointError"})
    result = sweep.report(path, tmp_path / "runs", tmp_path / "frozen.json", freeze=True)
    assert result["status"] == "frozen" and result["terminal"]
    assert len(result["trials"]) == 30 and len(result["settings"]) == 10
    assert result["selected"]["adamw"]["setting"] == 1  # .002 nat tie favors measured time.
    assert result["selected"]["soap"]["setting"] == 0
    assert result["selected"]["adamw"]["loss_std"] == pytest.approx(.01)
    assert result["selected"]["adamw"]["loss_standard_error"] == pytest.approx(.01 / np.sqrt(3))
    assert len(result["selected"]["adamw"]["seed_losses"]) == 3
    assert any(r["status"] == "failed" for r in result["trials"])
    assert str(failed / "failure-worker-0.json") in result["evidence_sha256"]


def test_sweep_gate_rejects_changed_controls_source_inputs_runtime_and_evidence(registered_plan, tmp_path):
    path, plan = registered_plan
    config = run_from_dict(next(e["config"] for e in plan["entries"] if e["optimizer"] == "adamw"))
    proof = tmp_path / "fixture-proof.json"; atomic_json(proof, {"fixture_only": True})
    env = {"backend": "tpu", "device_kind": "TPU v4", "process_count": 4, "local_device_count": 4, "jax": "0.6.2", "jaxlib": "0.6.2"}
    gate = {"schema": 1, "phase": 4, "status": "passed", "plan_path": str(path), "plan_sha256": sha256(path),
        "source_sha256": source_hashes(), "validated_config_hashes": [config.fingerprint], "data_manifest_sha256": "data",
        "prior_sha256_by_optimizer": {"adamw": None}, "environment": env, "covered_state_shapes": [sweep_gate.state_shape_key(config)],
        "evidence_sha256": {str(proof): sha256(proof)}}
    gate_path = tmp_path / "fixture-gate.json"; atomic_json(gate_path, gate)
    sweep_gate.validate_sweep_gate(gate_path, config, "data", None, env)
    for mutation in ("source", "data", "prior", "shape", "coverage", "runtime", "plan", "phase"):
        changed = deepcopy(gate)
        if mutation == "source": changed["source_sha256"] = {}
        elif mutation == "data": changed["data_manifest_sha256"] = "changed"
        elif mutation == "prior": changed["prior_sha256_by_optimizer"]["adamw"] = "unused-prior"
        elif mutation == "shape": changed["covered_state_shapes"] = []
        elif mutation == "coverage": changed["validated_config_hashes"] = []
        elif mutation == "runtime": changed["environment"]["process_count"] = 1
        elif mutation == "plan": changed["plan_sha256"] = "changed"
        else: changed["phase"] = 3
        atomic_json(gate_path, changed)
        with pytest.raises(ValueError):
            sweep_gate.validate_sweep_gate(gate_path, config, "data", None, env)
    atomic_json(gate_path, gate)
    proof.write_text("modified proof")
    with pytest.raises(ValueError, match="evidence changed"):
        sweep_gate.validate_sweep_gate(gate_path, config, "data", None, env)


def test_tiny_smoke_is_rejected_as_full_shape_sweep_evidence(registered_plan, tmp_path, fixture_data):
    path, _ = registered_plan
    cases = "".join(f'<testcase name="{name}"/>' for name in sweep_gate.REQUIRED_CPU_TESTS)
    xml = tmp_path / "fixture.xml"; xml.write_text("<testsuites><testsuite>" + cases + "</testsuite></testsuites>")
    with pytest.raises(ValueError, match="real prepared corpus"):
        sweep_gate.create(path, fixture_data, [], xml, tmp_path / "forbidden-gate.json")
    assert not (tmp_path / "forbidden-gate.json").exists()


def test_phase3_selection_criterion_and_seed_confounding_note(tmp_path, monkeypatch):
    losses, times = {"A": 4., "B": 3.994, "C": 4.02, "D": 4.01}, {"A": 80., "B": 100., "C": 70., "D": 50.}

    def fixture_run(directory, config):
        cell = config.optimizer.configuration
        evaluation = {"validation_loss": losses[cell], "elapsed_seconds": times[cell]}
        row = {"step": 1, "evaluation": evaluation, "elapsed_seconds": times[cell], "refresh": True,
            "hvp_count": 8, "residual_ratio": .3, "stopped_by_policy": cell in "CD", "rank_exhausted": False,
            "clipped_eigenvalues": 0, "curvature_age": 0}
        return {"config": config, "run": {"source_sha256": {}, "data_hash": "same", "prior_hash": "same", "compilation": {}},
            "summary": {"parameter_count": 60957184, "tokens": config.token_budget, "total_hvps": 8, "device_memory_stats": []},
            "rows": [row], "final_loss": losses[cell], "perplexity": float(np.exp(losses[cell])), "elapsed_seconds": times[cell],
            "costs": {}, "prior_attempt_failures": [], "evidence_sha256": {}}

    monkeypatch.setattr(compare, "read_run", fixture_run)
    result = compare.comparison([tmp_path / c for c in "ABCD"], tmp_path / "fixture-comparison.json")
    assert result["winner"] == "A" and "confounds seed" in result["limitations"]
    assert result["exploratory_loss_contrasts"]["count_fixed_alignment_C_minus_A"] == pytest.approx(.02)
    with pytest.raises(ValueError, match="four"):
        compare.comparison([tmp_path], None)
