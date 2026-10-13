"""Export measured acceleration tradeoffs and freeze an execution choice."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from curvature_compass.lm.artifacts import atomic_json

ROOT = Path("analysis/results/phase23")


def report(freeze=False, reference_name="benchmark_reference"):
    records = {}
    for path in sorted(ROOT.glob("benchmark_*-worker-0.json")):
        record = json.loads(path.read_text())
        records[path.stem.removesuffix("-worker-0")] = record
    reference = records[reference_name]
    rows = []
    for name, record in records.items():
        medians = record["medians"]
        row = {"artifact": name, "case": record["case"], "status": record["status"],
               "gradient_seconds": medians["gradient_seconds"], "curvature_seconds": medians["curvature_seconds"],
               "update_seconds": medians["update_seconds"], "refresh_step_seconds": medians["refresh_step_seconds"],
               "cached_step_seconds": medians["cached_step_seconds"],
               "gradient_relative_error": record["gradient_relative_error"], "direction_relative_error": record["direction_relative_error"],
               "refresh_speedup_vs_fp32": reference["medians"]["refresh_step_seconds"] / medians["refresh_step_seconds"],
               "cached_speedup_vs_fp32": reference["medians"]["cached_step_seconds"] / medians["cached_step_seconds"]}
        rows.append(row)
    with (ROOT / "acceleration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 4.5))
    names = [r["case"] for r in rows]
    x = np.arange(len(rows))
    gradient = np.array([r["gradient_seconds"] for r in rows])
    curvature = np.array([r["curvature_seconds"] for r in rows])
    update = np.array([r["update_seconds"] for r in rows])
    axis.bar(x, gradient, label="Training gradient")
    axis.bar(x, curvature, bottom=gradient, label="Eight HVPs and acquisition")
    axis.bar(x, update, bottom=gradient + curvature, label="Parameter update")
    low = [min(record["refresh_step_seconds"] for record in records[r["artifact"]]["raw"]) for r in rows]
    high = [max(record["refresh_step_seconds"] for record in records[r["artifact"]]["raw"]) for r in rows]
    totals = gradient + curvature + update
    axis.errorbar(x, totals, yerr=[np.maximum(0, totals - low), np.maximum(0, high - totals)], fmt="none", ecolor="black", capsize=3)
    axis.set_xticks(x, names, rotation=20); axis.set_ylabel("Seconds per measured refresh step")
    axis.set_title("60.96M parameters · v4-32 / 16 JAX devices · batch 64 × 1024\nThree steady-state repeats; bars show median stage times, whiskers observed range")
    axis.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(ROOT / "acceleration.png", dpi=180); plt.close(fig)
    if freeze:
        # Preserve rejected measurements, but ship only standard XLA paths.
        accepted = [(name, record) for name, record in records.items()
                    if record["status"] == "passed" and record["case"] in ("reference", "bf16", "sdpa")]
        xla_name, xla = reference_name, reference
        def meaningful(candidate, baseline):
            return max(baseline["medians"]["refresh_step_seconds"] / candidate["medians"]["refresh_step_seconds"],
                       baseline["medians"]["cached_step_seconds"] / candidate["medians"]["cached_step_seconds"]) >= 1.05
        # Isolate each added component against its appropriate reference.
        # A tiny attention gain cannot inherit bf16's larger speedup.
        for desired in ("bf16", "sdpa"):
            candidates = [(name, record) for name, record in accepted if record["case"] == desired]
            if candidates:
                name, record = min(candidates, key=lambda pair: pair[1]["medians"]["amortized_step_seconds_estimate"])
                if meaningful(record, xla) and record["medians"]["amortized_step_seconds_estimate"] < xla["medians"]["amortized_step_seconds_estimate"]:
                    xla_name, xla = name, record
        selected_name, selected = xla_name, xla
        improvement = max(reference["medians"]["refresh_step_seconds"] / selected["medians"]["refresh_step_seconds"],
                          reference["medians"]["cached_step_seconds"] / selected["medians"]["cached_step_seconds"])
        if improvement < 1.05:
            selected_name, selected = reference_name, reference
        options = {key: selected["config"][key] for key in ("compute_dtype", "attention", "rematerialize", "reuse_linearization")}
        selection = {"schema": 1, "status": "frozen",
                     "criterion": "correctness and >=5% median refresh or cached step improvement; otherwise standard XLA",
                     "selected_benchmark": selected_name, "reference_benchmark": reference_name,
                     "selected_options": options, "custom_kernel_code_removed": True,
                     "rejected_candidates": {"algebra": "Approximately 1.5% refresh gain; removed at user request.",
                                             "splash": "Failed compilation for head size and automatic pod partitioning; removed.",
                                             "sdpa": "Below 5% incremental gain over bf16 reference attention.",
                                             "remat": "Slower; measured memory already fits.",
                                             "recompute": "Slower acquisition than reused linearization."},
                     "measured_rows": rows,
                     "scope": "Execution choice, not an optimizer winner. Amortized stage cost is an estimate; no full-token training outcome is inferred."}
        atomic_json(ROOT / "acceleration_selection.json", selection)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--reference", default="benchmark_reference")
    args = parser.parse_args()
    report(args.freeze, args.reference)
