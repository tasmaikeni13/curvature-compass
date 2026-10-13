"""Read live structured training logs without accessing the TPU runtime."""
import argparse
import json
from pathlib import Path
import time


def inspect(directory):
    rows = []
    for path in sorted(directory.glob("metrics-worker-*.jsonl")):
        lines = path.read_text().splitlines()
        complete = None
        for line in reversed(lines):
            try:
                complete = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if complete is not None:
            rows.append({"worker_log": path.name, "log_age_seconds": time.time() - path.stat().st_mtime,
                         **{k: complete.get(k) for k in ("step", "tokens", "training_loss", "training_tokens_per_second",
                                 "total_hvps", "curvature_age", "optimizer_seconds", "evaluation")}})
    terminal = []
    for pattern in ("summary-worker-*.json", "failure-worker-*.json"):
        for path in sorted(directory.glob(pattern)):
            value = json.loads(path.read_text())
            terminal.append({"artifact": path.name, "status": value.get("status", "failed"),
                             "step": value.get("step"), "tokens": value.get("tokens"), "error": value.get("error")})
    return {"directory": str(directory), "latest_metrics": rows, "terminal_artifacts": terminal,
            "note": "Log age is observational; a quiet log alone does not prove that a process stopped."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.)
    args = parser.parse_args()
    while True:
        print(json.dumps(inspect(args.directory), indent=2), flush=True)
        if not args.watch:
            break
        time.sleep(max(1., args.interval))


if __name__ == "__main__":
    main()
