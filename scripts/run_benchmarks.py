"""Run a finite list of independent compilation/steady-state candidates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", help="Benchmark candidate cases (e.g. reference, bf16, sdpa)")
    parser.add_argument("--config", type=Path, default=Path("configs/lm125m_muon.json"),
                        help="Model configuration file")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/results/benchmarks"),
                        help="Output directory for benchmark logs and summaries")
    parser.add_argument("--data", type=Path, default=Path("data/fineweb"),
                        help="Path to tokenized dataset directory")
    args = parser.parse_args()

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        worker_out = root / f"benchmark_{case}-worker-0.json"
        if worker_out.exists():
            raise ValueError(f"Candidate output already exists: {case}; review it before rerunning")

        command = [
            sys.executable, "scripts/pod.py",
            "--logs", str(root / f"benchmark_{case}_logs"),
            "--timeout", "900",
            "curvature_compass.lm.benchmark",
            "--config", str(args.config),
            "--data", str(args.data),
            "--output", str(root / f"benchmark_{case}.json"),
            "--case", case,
            "--repeats", "3"
        ]
        started = time.perf_counter()
        log_path = root / f"launcher_{case}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        record = {
            "case": case,
            "command": command,
            "exit_code": result.returncode,
            "launcher_seconds": time.perf_counter() - started
        }
        (root / f"launcher_{case}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if result.returncode:
            print(f"Candidate {case} failed; retained complete worker/launcher logs", flush=True)


if __name__ == "__main__":
    main()
