"""Streaming hashes and atomic local artifacts; no unsafe pickle checkpoints."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "src/curvature_compass/lm").glob("*.py"))
    paths += list((root / "configs").glob("*.json"))
    paths += list((root / "scripts").glob("*.py")) + list((root / "tests").glob("*.py"))
    paths += [root / "pyproject.toml"]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def environment():
    import jax
    import jaxlib
    import numpy
    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": numpy.__version__,
              "devices": [str(d) for d in jax.devices()], "device_kind": jax.devices()[0].device_kind,
              "process_count": jax.process_count(), "local_device_count": jax.local_device_count(),
              "backend": jax.default_backend()}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result
