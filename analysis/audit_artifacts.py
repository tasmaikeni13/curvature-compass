"""Verify checked-in run bytes and source references without rerunning studies."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis/results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    records = []
    for stage in ("calibration", "evaluation"):
        manifest = json.loads((RESULTS / f"{stage}_manifest.json").read_text())
        data = RESULTS / f"{stage}.csv.gz"
        assert sha(data) == manifest["raw_sha256"], stage
        for source, digest in manifest["source_sha256"].items():
            assert sha(ROOT / source) == digest, (stage, source)
        with gzip.open(data, "rt") as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
        assert row_count == manifest["rows"], (stage, row_count, manifest["rows"])
        records.append({"stage": stage, "rows": row_count, "raw_sha256": sha(data)})

    for stage in ("reference", "iterative", "noise", "drift", "nonconvex",
                  "accuracy", "scalar", "scaling"):
        manifest = json.loads((RESULTS / f"{stage}_manifest.json").read_text())
        data = RESULTS / f"{stage}.jsonl.gz"
        assert manifest["status"] == "completed", stage
        assert sha(data) == manifest["raw_sha256"], stage
        for source, digest in manifest["source_sha256"].items():
            assert sha(ROOT / source) == digest, (stage, source)
        with gzip.open(data, "rt") as stream:
            row_count = sum(1 for _ in stream)
        assert row_count == manifest["records"], (stage, row_count, manifest["records"])
        records.append({"stage": stage, "rows": row_count, "raw_sha256": sha(data)})

    retrieved = json.loads((ROOT / "analysis/literature/retrieval.json").read_text())
    sources = json.loads((ROOT / "analysis/literature/sources.json").read_text())
    assert len(retrieved["records"]) == len(sources) == 30
    assert all(item["status"] == "downloaded_and_extracted" for item in retrieved["records"])
    lean_log = (RESULTS / "lean_build.log").read_text()
    assert "Build completed successfully" in lean_log
    assert "sorryAx" not in lean_log and "error:" not in lean_log
    assert lean_log.count("depends on axioms:") == 15
    result = {"status": "passed", "runs": records,
              "primary_sources": len(sources), "lean_theorems": 15}
    (RESULTS / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
