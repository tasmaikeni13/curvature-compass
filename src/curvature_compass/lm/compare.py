"""Report the 125M / 2.5B FineWeb-Edu optimizer comparison across 5 matched seeds."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .artifacts import atomic_json
from .experiments import load_run_config
from .run_results import pareto_frontier, read_run

BENCHMARK_OPTIMIZERS = ("adamw", "muon", "soap", "sophia_g")
METHODS = BENCHMARK_OPTIMIZERS
SEEDS_COUNT = 5
SEEDS = (51001, 51002, 51003, 51004, 51005)


def comparison(directories, output=None):
    if len(directories) != len(BENCHMARK_OPTIMIZERS) * SEEDS_COUNT:
        raise ValueError(
            f"Exactly {len(BENCHMARK_OPTIMIZERS) * SEEDS_COUNT} run directories required "
            f"({len(BENCHMARK_OPTIMIZERS)} optimizers x {SEEDS_COUNT} matched seeds)"
        )
    directories = [Path(p).resolve() for p in directories]
    records = []
    evidence = {}
    for directory in directories:
        run_file = directory / "run-worker-0.json"
        if not run_file.exists():
            raise ValueError(f"Run directory missing run-worker-0.json: {directory}")
        run_meta = json.loads(run_file.read_text())
        opt_name = run_meta.get("optimizer") or run_meta.get("config", {}).get("optimizer", {}).get("name")
        expected_config = run_meta.get("config")
        expected = load_run_config(directory / "run-worker-0.json") if expected_config is None else None
        result = read_run(directory, expected)
        evidence.update(result["evidence_sha256"])
        records.append({"directory": directory, "optimizer": opt_name, "result": result})

    # Validate model parameter count and provenance consistency
    first_run = records[0]["result"]["run"]
    for record in records:
        run = record["result"]["run"]
        summary = record["result"]["summary"]
        if summary.get("parameter_count") != 124_275_456:
            raise ValueError(
                f"125M comparison requires the registered 124,275,456 parameter LLaMA model, "
                f"got {summary.get('parameter_count')}"
            )
        if any(run.get(k) != first_run.get(k) for k in ("source_sha256", "data_hash")):
            raise ValueError("All comparison runs must share numerical source and prepared inputs")

    # Group by optimizer
    by_opt = {}
    for record in records:
        opt = record["optimizer"]
        if opt not in BENCHMARK_OPTIMIZERS:
            raise ValueError(f"Unexpected optimizer in comparison: {opt}. Expected one of {BENCHMARK_OPTIMIZERS}")
        by_opt.setdefault(opt, []).append(record)

    for opt in BENCHMARK_OPTIMIZERS:
        if len(by_opt.get(opt, [])) != SEEDS_COUNT:
            raise ValueError(f"Optimizer {opt} must have exactly {SEEDS_COUNT} runs, got {len(by_opt.get(opt, []))}")

    # Sort each optimizer's runs by seed
    for opt in BENCHMARK_OPTIMIZERS:
        by_opt[opt].sort(key=lambda r: (r["result"]["config"].seed, r["result"]["config"].data_seed))

    # Validate matched seeds across all optimizers
    adamw_seeds = [(r["result"]["config"].seed, r["result"]["config"].data_seed) for r in by_opt["adamw"]]
    for opt in BENCHMARK_OPTIMIZERS:
        opt_seeds = [(r["result"]["config"].seed, r["result"]["config"].data_seed) for r in by_opt[opt]]
        if opt_seeds != adamw_seeds:
            raise ValueError(f"Optimizer {opt} seeds {opt_seeds} do not match AdamW seeds {adamw_seeds}")

    # Compute per-optimizer metrics
    optimizer_summaries = []
    losses_by_opt = {}
    for opt in BENCHMARK_OPTIMIZERS:
        runs = by_opt[opt]
        losses = [r["result"]["final_loss"] for r in runs]
        losses_by_opt[opt] = losses
        times = [r["result"]["elapsed_seconds"] for r in runs]
        tokens = [r["result"]["summary"]["tokens"] for r in runs]
        throughputs = [tok / max(t, 1e-30) for tok, t in zip(tokens, times)]
        mean_l = float(np.mean(losses))
        std_l = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
        se_l = std_l / math.sqrt(len(losses))
        mean_t = float(np.mean(times))
        std_t = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        mean_tp = float(np.mean(throughputs))
        optimizer_summaries.append({
            "optimizer": opt,
            "runs": [str(r["directory"]) for r in runs],
            "seed_losses": losses,
            "mean_loss": mean_l,
            "loss_std": std_l,
            "loss_standard_error": se_l,
            "loss_range": [min(losses), max(losses)],
            "mean_elapsed_seconds": mean_t,
            "elapsed_std": std_t,
            "mean_tokens_per_second": mean_tp,
            "final_perplexity": float(np.exp(mean_l)),
        })

    # Paired seed contrasts relative to AdamW: Delta L = L_opt - L_adamw
    paired_contrasts = {}
    for opt in BENCHMARK_OPTIMIZERS:
        if opt == "adamw":
            continue
        deltas = [l_opt - l_ref for l_opt, l_ref in zip(losses_by_opt[opt], losses_by_opt["adamw"])]
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        se_d = std_d / math.sqrt(len(deltas))
        paired_contrasts[f"{opt}_minus_adamw"] = {
            "seed_deltas": deltas,
            "mean_delta": mean_d,
            "delta_std": std_d,
            "delta_standard_error": se_d,
            "ci_95": [mean_d - 1.96 * se_d, mean_d + 1.96 * se_d],
        }

    # Winner selection: lowest mean final loss, within <0.01 nat prefer mean elapsed training time
    minimum_loss = min(s["mean_loss"] for s in optimizer_summaries)
    eligible = [s for s in optimizer_summaries if s["mean_loss"] - minimum_loss < 0.01]
    winner = min(eligible, key=lambda s: (s["mean_elapsed_seconds"], s["mean_loss"], s["optimizer"]))

    pareto = pareto_frontier(optimizer_summaries, "mean_loss", "mean_elapsed_seconds")

    report = {
        "schema": 1,
        "pipeline": "125M_2.5B_comparison",
        "status": "frozen",
        "winner": winner["optimizer"],
        "run_directories": [str(p) for p in directories],
        "parameter_count": 124_275_456,
        "token_budget": 2_500_000_000,
        "optimizers": optimizer_summaries,
        "pareto_optimizers": [s["optimizer"] for s in pareto],
        "paired_contrasts": paired_contrasts,
        "criterion": "Stable exact-token final validation loss over 5 matched seeds; within <.01 nat of minimum, prefer measured lower mean elapsed training time.",
        "evidence_sha256": evidence,
    }
    if output is not None:
        atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="20 run directories (4 optimizers x 5 matched seeds)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = comparison(args.runs, args.output)
    print(json.dumps({"winner": report["winner"], "optimizers": report["optimizers"]}, indent=2))


if __name__ == "__main__":
    main()
