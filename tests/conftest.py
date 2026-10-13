import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import pytest
from curvature_compass.lm.artifacts import atomic_json, sha256
from curvature_compass.lm.data import DATA_REVISION, TOKENIZER_REVISION, document_key, split_for


@pytest.fixture
def fixture_data(tmp_path):
    """Explicitly synthetic unit-test data; real smoke runs use pinned FineWeb."""
    root = tmp_path / "data"
    root.mkdir()
    files, counts = {}, {}
    for split, size in (("train", 1025), ("validation", 129), ("calibration", 513)):
        np.asarray(np.arange(size) % 100 + 1, dtype="<u2").tofile(root / (split + ".bin"))
        index = 0
        while split_for(document_key(f"test fixture {index}")) != split:
            index += 1
        (root / (split + ".documents.sha256")).write_bytes(document_key(f"test fixture {index}"))
        counts[split] = size - 1
        for suffix in (".bin", ".documents.sha256"):
            p = root / (split + suffix)
            files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    atomic_json(root / "manifest.json", {"schema": 1, "dataset_revision": DATA_REVISION,
                "tokenizer_revision": TOKENIZER_REVISION, "vocab_size": 50257,
                "files": files, "predicted_tokens": counts, "complete_source_scan": False,
                "fixture_only": True})
    return root
