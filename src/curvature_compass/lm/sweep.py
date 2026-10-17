"""Generate, launch, report and freeze the 600M FineWeb-Edu hyperparameter sweep."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .artifacts import atomic_json, sha256, source_hashes
from .config import digest
from .experiments import load_run_config, optimizer_name, run_from_dict
from .model import Decoder
from .run_results import pareto_frontier, read_run

METHODS = ("adamw", "muon", "soap", "sophia_g", "tomography")
SEEDS = ((41001, 42001), (41002, 42002), (41003, 42003))
SPACES = {
    "adamw": {
        "learning_rate": ("log", 0.0001, 0.003),
        "optimizer.beta1": ("choice", 0.9, 0.95),
        "optimizer.beta2": ("choice", 0.95, 0.98, 0.99, 0.999),
        "optimizer.epsilon": ("log", 1e-10, 1e-6),
        "weight_decay": ("choice", 0.0, 0.01, 0.1),
    },
    "muon": {
        "learning_rate": ("log", 0.003, 0.1),
        "optimizer.momentum": ("choice", 0.9, 0.95, 0.98),
        "optimizer.ns_steps": ("choice", 5, 8),
        "optimizer.auxiliary_lr_ratio": ("log", 0.005, 0.1),
        "optimizer.beta2": ("choice", 0.95, 0.99),
        "weight_decay": ("choice", 0.0, 0.01, 0.1),
    },
    "soap": {
        "learning_rate": ("log", 0.0001, 0.005),
        "optimizer.beta1": ("choice", 0.9, 0.95),
        "optimizer.beta2": ("choice", 0.95, 0.99),
        "optimizer.shampoo_beta": ("choice", 0.9, 0.95, 0.99),
        "optimizer.precondition_frequency": ("choice", 5, 10, 20),
        "optimizer.max_precondition_dimension": ("choice", 768, 2560),
        "weight_decay": ("choice", 0.0, 0.01, 0.1),
    },
    "sophia_g": {
        "learning_rate": ("log", 0.0001, 0.003),
        "optimizer.beta1": ("choice", 0.9, 0.965, 0.98),
        "optimizer.beta2": ("choice", 0.95, 0.99, 0.995),
        "optimizer.rho": ("log", 0.01, 0.1),
        "optimizer.hessian_interval": ("choice", 5, 10, 20),
        "weight_decay": ("choice", 0.0, 0.01, 0.1),
    },
    "tomography": {
        "learning_rate": ("log", 0.005, 0.15),
        "optimizer.damping": ("log", 0.1, 10.0),
        "optimizer.tolerance": ("choice", 0.3, 0.5, 0.7, 0.9),
        "optimizer.cap": ("choice", 4, 8),
        "optimizer.refresh_interval": ("choice", 4, 8, 16, 32),
        "weight_decay": ("choice", 0.0, 0.01, 0.1),
    },
}


def generate(
    output,
    methods=METHODS,
    settings=8,
    token_budget_per_optimizer=600_000_000,
    plan_seed=43000,
    config_dir="configs",
):
    if settings < 2:
        raise ValueError("A systematic sweep requires at least two settings per optimizer")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an already registered sweep plan")

    methods = list(methods)
    num_seeds = len(SEEDS)
    trials_per_optimizer = settings * num_seeds
    token_budget_per_trial = token_budget_per_optimizer // trials_per_optimizer
    steps_per_trial = token_budget_per_trial // (64 * 1024)
    actual_trial_tokens = steps_per_trial * 64 * 1024

    rng, entries = np.random.default_rng(plan_seed), []
    for method in methods:
        base = load_run_config(f"{config_dir}/lm125m_{method}.json")
        space = dict(SPACES[method])
        if method == "tomography" and not base.optimizer.adaptive_count:
            space.pop("optimizer.tolerance", None)
        strata = {key: (rng.permutation(settings - 1) + rng.random(settings - 1)) / (settings - 1) for key in space}
        for setting in range(settings):
            controls = base.to_dict()
            controls["token_budget"] = actual_trial_tokens
            controls["training_steps"] = steps_per_trial
            if setting:
                for key, definition in space.items():
                    quantile = strata[key][setting - 1]
                    if definition[0] == "log":
                        value = math.exp(math.log(definition[1]) + quantile * math.log(definition[2] / definition[1]))
                    else:
                        choices = definition[1:]
                        value = choices[min(len(choices) - 1, int(quantile * len(choices)))]
                    if key.startswith("optimizer."):
                        controls["optimizer"][key.split(".", 1)[1]] = value
                    else:
                        controls[key] = value
            for seed_index, (seed, data_seed) in enumerate(SEEDS):
                config = run_from_dict({**controls, "seed": seed, "data_seed": data_seed})
                entries.append({
                    "id": f"{method}-{setting:03d}-{seed_index}",
                    "optimizer": method,
                    "setting": setting,
                    "seed_index": seed_index,
                    "config": config.to_dict(),
                    "config_hash": config.fingerprint,
                })
    plan = {
        "schema": 1,
        "status": "registered",
        "settings_per_optimizer": settings,
        "plan_seed": plan_seed,
        "seed_pairs": SEEDS,
        "search_spaces": {m: SPACES[m] for m in methods},
        "methods": methods,
        "source_sha256": source_hashes(),
        "entries": entries,
        "trial_count": len(entries),
        "steps_per_trial": steps_per_trial,
        "total_training_steps": steps_per_trial * len(entries),
        "token_budget_per_optimizer": token_budget_per_optimizer,
        "total_predicted_training_tokens": sum(e["config"]["token_budget"] for e in entries),
        "selection_rule": "Three stable completed seeds per setting; minimum mean final validation loss, within <.005 nat prefer mean elapsed training time. All planned trials must be terminal.",
        "scale_transfer": "Hyperparameter tuning conducted directly on the nominal 125M LLaMA architecture.",
    }
    plan["fingerprint"] = digest(plan)
    validate_plan(plan)
    for entry in entries:
        atomic_json(output / "configs" / (entry["id"] + ".json"), entry["config"])
    atomic_json(output / "plan.json", plan)
    return plan


def validate_plan(plan, current_source=False):
    body = {k: v for k, v in plan.items() if k != "fingerprint"}
    if plan.get("schema") != 1 or plan.get("status") != "registered" or digest(body) != plan.get("fingerprint"):
        raise ValueError("Sweep plan schema/status/fingerprint mismatch")
    if current_source and plan["source_sha256"] != source_hashes():
        raise ValueError("Sweep source changed; register and validate a new plan")
    methods = plan.get("methods", METHODS)
    if plan["settings_per_optimizer"] < 2 or digest(plan["seed_pairs"]) != digest(SEEDS):
        raise ValueError("Sweep registered search/seed controls differ from the protocol")
    expected_search_spaces = {m: SPACES[m] for m in methods}
    if digest(plan.get("search_spaces", {})) != digest(expected_search_spaces):
        raise ValueError("Sweep registered search/seed controls differ from the protocol")
    if len(plan["entries"]) != plan["trial_count"] or plan["trial_count"] != len(methods) * 3 * plan["settings_per_optimizer"]:
        raise ValueError("Sweep trial count mismatch with methods and seeds")
    groups, common = {}, None
    ids = set()
    for entry in plan["entries"]:
        config = run_from_dict(entry["config"])
        if entry["id"] in ids or entry["config_hash"] != config.fingerprint or optimizer_name(config) != entry["optimizer"]:
            raise ValueError("Sweep trial identity/config mismatch")
        ids.add(entry["id"])
        if (config.seed, config.data_seed) != SEEDS[entry["seed_index"]]:
            raise ValueError("Sweep seeds are not matched across optimizers")
        if Decoder(config.model).parameter_count != 124_275_456:
            raise ValueError("Sweep requires nominal 125M model with exactly 124,275,456 parameters")
        shared = config.to_dict()
        for name in ("optimizer", "seed", "data_seed", "learning_rate", "weight_decay"):
            shared.pop(name, None)
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError("Sweep has uncontrolled architecture/training differences")
        key = (entry["optimizer"], entry["setting"])
        groups.setdefault(key, []).append(entry)
    expected = {(method, i) for method in methods for i in range(plan["settings_per_optimizer"])}
    if set(groups) != expected:
        raise ValueError("Sweep setting coverage is incomplete")
    for entries in groups.values():
        if sorted(e["seed_index"] for e in entries) != [0, 1, 2]:
            raise ValueError("Every sweep setting must have all three seeds")
        controls = [{k: v for k, v in e["config"].items() if k not in ("seed", "data_seed")} for e in entries]
        if any(v != controls[0] for v in controls):
            raise ValueError("Hyperparameters changed between seeds of a setting")
    steps_per_trial = plan.get("steps_per_trial", 381)
    if plan["total_training_steps"] != steps_per_trial * len(plan["entries"]) or plan["total_predicted_training_tokens"] != sum(e["config"]["token_budget"] for e in plan["entries"]):
        raise ValueError("Sweep resource accounting differs from its actual trial budgets")
    return plan


def load_plan(path, current_source=False):
    return validate_plan(json.loads(Path(path).read_text()), current_source)


def collect_worker_summaries(directory):
    """Collect only small terminal records; parameters/checkpoints stay local."""
    directory = Path(directory).resolve()
    for worker in range(1, 4):
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        f"w{worker}:{directory}/summary-worker-*.json", str(directory)], check=True)


def launch(plan_path, data, output, gate, prior=None, trial_ids=None, resume=False, timeout=86400):
    from .sweep_gate import validate_sweep_gate
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = load_plan(plan_path, current_source=True)
    gate_record = json.loads(Path(gate).read_text())
    if gate_record.get("plan_sha256") != sha256(plan_path):
        raise ValueError("Sweep gate binds a different registered plan")
    selected = plan["entries"] if trial_ids is None else [e for e in plan["entries"] if e["id"] in trial_ids]
    if trial_ids is not None and {e["id"] for e in selected} != set(trial_ids):
        raise ValueError("Unknown sweep trial ID")
    from .calibrate import load_prior
    from .data import verify_data
    manifest = verify_data(data, full=True)
    if not manifest["complete_source_scan"]:
        raise ValueError("Sweep requires the prepared pinned corpus, not a synthetic fixture")
    data_hash = sha256(Path(data) / "manifest.json")
    env = gate_record["environment"]
    for entry in selected:
        config = run_from_dict(entry["config"])
        fixed = entry["optimizer"] == "tomography" and not config.optimizer.adaptive_alignment
        prior_hash = None
        if fixed:
            if prior is None:
                raise ValueError("The selected fixed-alignment cell requires its 125M PCA frame")
            frame, _ = load_prior(prior, Decoder(config.model), config.optimizer.cap, data_hash)
            del frame
            prior_hash = sha256(prior)
        validate_sweep_gate(gate, config, data_hash, prior_hash, env)
        path = plan_path.parent / "configs" / (entry["id"] + ".json")
        if load_run_config(path).fingerprint != config.fingerprint:
            raise ValueError("Generated trial config changed since registration")
    root = Path(__file__).resolve().parents[3]
    for entry in selected:
        directory = output / entry["id"]
        if (directory / "summary-worker-0.json").exists():
            try:
                if any(not (directory / f"summary-worker-{i}.json").exists() for i in range(1, 4)):
                    collect_worker_summaries(directory)
                read_run(directory, run_from_dict(entry["config"]))
                continue
            except ValueError:
                if not resume:
                    raise ValueError("Existing incomplete/failed trial requires --resume or a new run directory")
        command = [sys.executable, str(root / "scripts/pod.py"), "--timeout", str(timeout),
            "--logs", str(directory / "launcher"), "curvature_compass.lm.train", "--config",
            str(plan_path.parent / "configs" / (entry["id"] + ".json")), "--data", str(Path(data).resolve()),
            "--output", str(directory), "--full-run", "--gate", str(Path(gate).resolve())]
        if entry["optimizer"] == "tomography" and prior is not None:
            command.extend(["--prior", str(Path(prior).resolve())])
        if resume and (directory / "run-worker-0.json").exists():
            command.append("--resume")
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=root, check=False)
        attempt_dir = directory / "attempts"
        attempt = len(list(attempt_dir.glob("*.json"))) if attempt_dir.exists() else 0
        infrastructure_failure = completed.returncode != 0 and not (directory / "run-worker-0.json").exists()
        atomic_json(attempt_dir / f"{attempt:04d}.json", {"trial": entry["id"], "config_hash": entry["config_hash"],
            "returncode": completed.returncode, "launcher_seconds": time.perf_counter() - started,
            "status": "infrastructure_failure" if infrastructure_failure else "finished" if completed.returncode == 0 else "failed",
            "command": command, "plan_sha256": sha256(plan_path)})
        if infrastructure_failure:
            raise RuntimeError("Pod launch failed before training started; leave busy workers untouched and retry later")
        if completed.returncode == 0:
            collect_worker_summaries(directory)


def report(plan_path, runs, output, freeze=False):
    plan_path, runs = Path(plan_path), Path(runs)
    plan = load_plan(plan_path)
    methods = plan.get("methods", METHODS)
    trials, evidence, groups = [], {str(plan_path): sha256(plan_path)}, {}
    for entry in plan["entries"]:
        directory = runs / entry["id"]
        row = {k: entry[k] for k in ("id", "optimizer", "setting", "seed_index", "config_hash")}
        attempts = []
        for path in sorted((directory / "attempts").glob("*.json")):
            attempts.append(json.loads(path.read_text()))
            evidence[str(path)] = sha256(path)
        row["attempts"] = attempts
        try:
            result = read_run(directory, run_from_dict(entry["config"]))
            if result["run"]["source_sha256"] != plan["source_sha256"]:
                raise ValueError("Trial numerical source differs from the frozen sweep plan")
            row.update(status="complete", final_loss=result["final_loss"], perplexity=result["perplexity"],
                       elapsed_seconds=result["elapsed_seconds"], costs=result["costs"],
                       total_hvps=result["summary"]["total_hvps"], device_memory_stats=result["summary"]["device_memory_stats"])
            evidence.update(result["evidence_sha256"])
        except (OSError, ValueError, KeyError) as error:
            failure_paths = list(directory.glob("failure-worker-*.json"))
            for path in failure_paths:
                evidence[str(path)] = sha256(path)
            failed_launch = attempts and attempts[-1]["returncode"] != 0 and attempts[-1].get("status") == "failed"
            row.update(status="failed" if failure_paths or failed_launch else "missing_or_incomplete", reason=str(error))
        trials.append(row)
        groups.setdefault((entry["optimizer"], entry["setting"]), []).append(row)
    settings = []
    for (method, setting), rows in groups.items():
        record = {"optimizer": method, "setting": setting, "trials": [r["id"] for r in rows],
                  "complete_seeds": sum(r["status"] == "complete" for r in rows),
                  "eligible": all(r["status"] == "complete" for r in rows),
                  "attempt_launcher_seconds": sum(a["launcher_seconds"] for r in rows for a in r["attempts"])}
        if record["eligible"]:
            losses, times = [r["final_loss"] for r in rows], [r["elapsed_seconds"] for r in rows]
            sd = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
            record.update(mean_loss=float(np.mean(losses)), loss_std=sd, loss_standard_error=sd / math.sqrt(len(losses)),
                          loss_range=[min(losses), max(losses)], seed_losses=losses, mean_elapsed_seconds=float(np.mean(times)),
                          elapsed_std=float(np.std(times, ddof=1)) if len(times) > 1 else 0.0)
        settings.append(record)
    selected, frontiers = {}, {}
    for method in methods:
        eligible = [s for s in settings if s["optimizer"] == method and s["eligible"]]
        frontiers[method] = [s["setting"] for s in pareto_frontier(eligible, "mean_loss", "mean_elapsed_seconds")]
        if eligible:
            minimum = min(s["mean_loss"] for s in eligible)
            best = min((s for s in eligible if s["mean_loss"] - minimum < .005),
                       key=lambda s: (s["mean_elapsed_seconds"], s["mean_loss"], s["setting"]))
            config = next(e["config"] for e in plan["entries"] if e["optimizer"] == method and e["setting"] == best["setting"] and e["seed_index"] == 0)
            selected[method] = {**best, "config": config, "config_hash": run_from_dict(config).fingerprint}
    terminal = all(t["status"] != "missing_or_incomplete" for t in trials)
    if freeze and (not terminal or set(selected) != set(methods)):
        raise ValueError("Cannot freeze: every planned trial must be terminal and each optimizer needs a stable three-seed setting")
    result = {"schema": 1, "status": "frozen" if freeze else "progress", "plan_sha256": sha256(plan_path),
        "trial_count": len(trials), "terminal": terminal, "trials": trials, "settings": settings,
        "selected": selected, "pareto_settings": frontiers, "selection_rule": plan["selection_rule"],
        "scale_transfer": plan["scale_transfer"], "evidence_sha256": evidence}
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("plan")
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--settings", type=int, default=8)
    generate_parser.add_argument("--token-budget", type=int, default=600_000_000)
    generate_parser.add_argument("--optimizers", nargs="+", choices=["adamw", "muon", "soap", "sophia_g", "tomography"], default=["adamw", "muon", "soap", "sophia_g", "tomography"])
    generate_parser.add_argument("--plan-seed", type=int, default=43000)
    generate_parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", required=True, type=Path)
    run_parser.add_argument("--data", required=True, type=Path)
    run_parser.add_argument("--prior", type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--gate", required=True, type=Path)
    run_parser.add_argument("--full-run", required=True, action="store_true")
    run_parser.add_argument("--trials", nargs="+")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--timeout", type=int, default=86400)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--plan", required=True, type=Path)
    report_parser.add_argument("--runs", required=True, type=Path)
    report_parser.add_argument("--output", required=True, type=Path)
    report_parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = generate(args.output, args.optimizers, args.settings, args.token_budget, args.plan_seed, args.config_dir)
        print(json.dumps({k: result[k] for k in ("trial_count", "total_training_steps", "total_predicted_training_tokens", "fingerprint")}, indent=2))
    elif args.command == "run":
        launch(args.plan, args.data, args.output, args.gate, args.prior, args.trials, args.resume, args.timeout)
    else:
        result = report(args.plan, args.runs, args.output, args.freeze)
        print(json.dumps({"status": result["status"], "terminal": result["terminal"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
