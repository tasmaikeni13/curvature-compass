"""Validate completed training artifacts before any optimizer selection."""
import json
import math
from pathlib import Path
import numpy as np
from .artifacts import sha256
from .experiments import run_from_dict


def read_run(directory, expected_config=None):
    directory = Path(directory)
    run_path, summary_path = directory / "run-worker-0.json", directory / "summary-worker-0.json"
    run = json.loads(run_path.read_text())
    summary = json.loads(summary_path.read_text())
    if run["mode"] != "comparison" or summary["mode"] != "comparison":
        raise ValueError("Smoke outcomes cannot participate in optimizer selection")
    config = run_from_dict(run["config"])
    if expected_config is not None and config.fingerprint != expected_config.fingerprint:
        raise ValueError("Run controls differ from the registered experiment")
    if run["config_hash"] != config.fingerprint or summary["config_hash"] != config.fingerprint:
        raise ValueError("Run/summary controls disagree")
    if summary["status"] != "finished" or not summary["complete_budget"] or summary["tokens"] != config.token_budget:
        raise ValueError("Run is not a completed exact-token comparison")
    expected_steps = math.ceil(config.token_budget / (config.global_batch_size * config.model.sequence_length))
    if summary["step"] != expected_steps:
        raise ValueError("Run has an incorrect training-step count")
    evidence = {}
    summaries = []
    for worker in range(run["environment"]["process_count"]):
        path = directory / f"summary-worker-{worker}.json"
        worker_summary = json.loads(path.read_text())
        for field in ("status", "mode", "config_hash", "parameter_count", "step", "tokens", "target_tokens",
                      "complete_budget", "total_hvps", "data_manifest_sha256", "prior_sha256", "source_sha256"):
            if worker_summary[field] != summary[field]:
                raise ValueError(f"Worker outcomes disagree: {field}")
        summaries.append(worker_summary)
        evidence[str(path)] = sha256(path)
    for field, run_field in (("source_sha256", "source_sha256"), ("data_manifest_sha256", "data_hash"), ("prior_sha256", "prior_hash")):
        if summary[field] != run[run_field]:
            raise ValueError("Run/summary provenance disagrees")
    metrics_path = directory / "metrics-worker-0.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    if len(rows) != expected_steps or [r["step"] for r in rows] != list(range(1, expected_steps + 1)):
        raise ValueError("Training logs have missing or duplicate steps")
    total_tokens, total_hvps = 0, 0
    for row in rows:
        total_tokens += row["predicted_tokens"]
        total_hvps += row["hvp_count"]
        if row["tokens"] != total_tokens or row["total_hvps"] != total_hvps or row["predicted_tokens"] <= 0:
            raise ValueError("Training logs violate exact token/HVP accounting")
        if not row["parameters_finite"] or not row.get("finite", True):
            raise ValueError("A nonfinite update/acquisition was recorded")
        if "direction_finite_and_descent" in row and not row["direction_finite_and_descent"]:
            raise ValueError("Tomography logged an invalid direction")
        if not np.isfinite(row["training_loss"]):
            raise ValueError("Training loss is nonfinite")
    if total_tokens != config.token_budget or total_hvps != summary["total_hvps"]:
        raise ValueError("Summary disagrees with the actual logged token/HVP counts")
    evaluations = summary["evaluations"]
    if not evaluations or evaluations[-1]["step"] != expected_steps or evaluations[-1]["tokens"] != config.token_budget:
        raise ValueError("No final held-out evaluation at the complete budget")
    if any(e["validation_tokens"] != config.validation_tokens or not np.isfinite(e["validation_loss"]) for e in evaluations):
        raise ValueError("Held-out evaluation controls changed or loss is nonfinite")
    if [r["evaluation"] for r in rows if "evaluation" in r] != evaluations:
        raise ValueError("Summary evaluations disagree with the training log")
    evidence.update({str(run_path): sha256(run_path), str(metrics_path): sha256(metrics_path)})
    failures = []
    for path in sorted(directory.glob("failure-worker-*.json")):
        failure = json.loads(path.read_text())
        failures.append(failure)
        evidence[str(path)] = sha256(path)
    if any(f.get("error_type") == "FloatingPointError" or "FloatingPointError" in f.get("traceback", "") for f in failures):
        raise ValueError("Numerical failure disqualifies this run from stable selection")
    costs = {name: sum(float(row.get(name, 0)) for row in rows)
             for name in ("gradient_seconds", "curvature_seconds", "update_seconds", "optimizer_seconds",
                          "data_seconds", "compute_step_seconds", "checkpoint_seconds")}
    costs["evaluation_seconds"] = sum(e["seconds"] for e in evaluations)
    costs["checkpoint_seconds"] += summary["final_checkpoint_seconds"]
    costs["compile_seconds"] = max(s["compile_seconds"] for s in summaries)
    elapsed = max(s["elapsed_seconds"] for s in summaries)
    return {"config": config, "run": run, "summary": summary, "rows": rows, "evidence_sha256": evidence,
            "final_loss": evaluations[-1]["validation_loss"], "perplexity": evaluations[-1]["perplexity"],
            "elapsed_seconds": elapsed, "costs": costs, "prior_attempt_failures": failures}


def pareto_frontier(records, loss="final_loss", time="elapsed_seconds"):
    return [r for r in records if not any(s[loss] <= r[loss] and s[time] <= r[time]
            and (s[loss] < r[loss] or s[time] < r[time]) for s in records)]


def checked_evidence(record):
    if not record.get("evidence_sha256"):
        raise ValueError("Selection is missing its underlying evidence")
    for path, expected in record["evidence_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Selection evidence changed: {path}")
