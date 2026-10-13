"""Run a finite list of independent compilation/steady-state candidates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    root = Path("analysis/results/phase23")
    for case in args.cases:
        if (root / f"benchmark_{case}-worker-0.json").exists():
            raise ValueError(f"Candidate output already exists: {case}; review it before rerunning")
        command = [sys.executable, "scripts/pod.py", "--logs", str(root / f"benchmark_{case}_logs"),
                   "--timeout", "900", "curvature_compass.lm.benchmark", "--config", "configs/lm60m_B.json",
                   "--data", "data/fineweb", "--prior", "artifacts/lm60m_prior.npy", "--output",
                   str(root / f"benchmark_{case}.json"), "--case", case, "--repeats", "3"]
        started = time.perf_counter()
        with (root / f"launcher_{case}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        record = {"case": case, "command": command, "exit_code": result.returncode,
                  "launcher_seconds": time.perf_counter() - started}
        (root / f"launcher_{case}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if result.returncode:
            print(f"Candidate {case} failed; retained complete worker/launcher logs", flush=True)


if __name__ == "__main__":
    main()
