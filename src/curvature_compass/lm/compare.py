"""Report the registered exploratory A/B/C/D screen, without running training."""
import argparse
import json
from pathlib import Path
import numpy as np
from .artifacts import atomic_json
from .config import load_config
from .run_results import pareto_frontier, read_run


def comparison(directories, output):
    if len(directories) != 4:
        raise ValueError("Exactly four A/B/C/D run directories are required")
    directories = [Path(p).resolve() for p in directories]
    records = []
    for cell, directory in zip("ABCD", directories):
        expected = load_config(f"configs/lm60m_{cell}.json")
        result = read_run(directory, expected)
        records.append({"cell": cell, "result": result})
    if len(records) != 4:
        raise ValueError("Exactly four A/B/C/D run directories are required")
    first = records[0]["result"]["run"]
    for record in records:
        run = record["result"]["run"]
        if any(run[key] != first[key] for key in ("source_sha256", "data_hash", "prior_hash")):
            raise ValueError("A/B/C/D do not share numerical source and prepared inputs")
        if record["result"]["summary"]["parameter_count"] != 60_957_184:
            raise ValueError("Phase 3 requires the registered approximately 60M decoder")
    common_time = min(r["result"]["elapsed_seconds"] for r in records)
    evidence, cells = {}, []
    for record in records:
        cell, result = record["cell"], record["result"]
        evidence.update(result["evidence_sha256"])
        rows = result["rows"]
        timed = [(r.get("evaluation", {}).get("elapsed_seconds", r["elapsed_seconds"]), r["evaluation"]["validation_loss"])
                 for r in rows if "evaluation" in r]
        crossings = {}
        for threshold in (8., 7., 6., 5., 4.):
            found = next((i for i, (_, loss) in enumerate(timed) if loss <= threshold), None)
            if found is None:
                crossings[str(threshold)] = None
            elif found == 0:
                crossings[str(threshold)] = {"observed_by_seconds": timed[0][0], "interpolated_seconds": None}
            else:
                t0, l0 = timed[found - 1]; t1, l1 = timed[found]
                crossings[str(threshold)] = {"observed_by_seconds": t1,
                    "interpolated_seconds": t0 + (t1 - t0) * (l0 - threshold) / (l0 - l1)}
        refreshes = [r for r in rows if r["refresh"]]
        cells.append({"cell": cell, "seed": result["config"].seed, "config_hash": result["config"].fingerprint,
            "final_loss": result["final_loss"], "perplexity": result["perplexity"],
            "tokens": result["summary"]["tokens"], "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result["summary"]["tokens"] / result["elapsed_seconds"],
            "costs": result["costs"], "total_hvps": result["summary"]["total_hvps"],
            "mean_hvps_per_refresh": float(np.mean([r["hvp_count"] for r in refreshes])),
            "mean_residual_ratio": float(np.mean([r["residual_ratio"] for r in refreshes])),
            "policy_stops": sum(r["stopped_by_policy"] for r in refreshes),
            "rank_exhaustions": sum(r["rank_exhausted"] for r in refreshes),
            "clipped_eigenvalues": sum(r["clipped_eigenvalues"] for r in refreshes),
            "mean_curvature_age": float(np.mean([r["curvature_age"] for r in rows])),
            "device_memory_stats": result["summary"]["device_memory_stats"],
            "compiled_memory": {k: v["memory"] for k, v in result["run"]["compilation"].items()},
            "validation_thresholds": crossings,
            "equal_time_loss_estimate": float(np.interp(common_time, *zip(*timed))) if timed[0][0] <= common_time <= timed[-1][0] else None,
            "prior_attempt_failures": result["prior_attempt_failures"]})
    minimum = min(c["final_loss"] for c in cells)
    eligible = [c for c in cells if c["final_loss"] - minimum < .01]
    winner = min(eligible, key=lambda c: (c["elapsed_seconds"], c["final_loss"], c["cell"]))
    losses = {c["cell"]: c["final_loss"] for c in cells}
    contrast = {"alignment_fixed_count_B_minus_A": losses["B"] - losses["A"],
                "alignment_adaptive_count_D_minus_C": losses["D"] - losses["C"],
                "count_fixed_alignment_C_minus_A": losses["C"] - losses["A"],
                "count_adaptive_alignment_D_minus_B": losses["D"] - losses["B"],
                "interaction_D_minus_C_minus_B_plus_A": losses["D"] - losses["C"] - losses["B"] + losses["A"]}
    report = {"schema": 1, "phase": 3, "status": "frozen", "winner": winner["cell"],
        "run_directories": [str(Path(p).resolve()) for p in directories],
        "criterion": "Stable exact-token final validation loss; within <.01 nat of minimum, prefer measured lower elapsed training time.",
        "cells": cells, "pareto_cells": [c["cell"] for c in pareto_frontier(cells)],
        "common_elapsed_seconds": common_time, "exploratory_loss_contrasts": contrast,
        "limitations": "One independently initialized seed per cell confounds seed and optimizer. Contrasts are exploratory, not causal estimates. Interpolated timings/losses are estimates. Stochastic clipped residuals are not certified Newton bounds.",
        "winner_optimizer": next(r["result"]["config"].to_dict()["optimizer"] for r in records if r["cell"] == winner["cell"]),
        "evidence_sha256": evidence}
    if output is not None:
        atomic_json(output, report)
    return report


def validate_selection(path):
    from .run_results import checked_evidence
    report = json.loads(Path(path).read_text())
    if report.get("phase") != 3 or report.get("status") != "frozen":
        raise ValueError("Phase 4 requires a frozen Phase 3 selection")
    checked_evidence(report)
    regenerated = comparison(report["run_directories"], None)
    if report != regenerated:
        raise ValueError("Phase 3 selection contradicts its controlled training evidence")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, required=True, type=Path, metavar="RUN", help="A B C D, in that order")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = comparison(args.runs, args.output)
    print(json.dumps({"winner": report["winner"], "cells": report["cells"]}, indent=2))


if __name__ == "__main__":
    main()
