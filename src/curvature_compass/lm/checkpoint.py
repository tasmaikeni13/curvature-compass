"""Committed checkpoint generations with integrity-checked rollback."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
from .artifacts import atomic_json, sha256
from .runtime import host_array


def save(directory, parameters, state, metadata, keep=2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{metadata['step']:08d}"
    final, temporary = directory / name, directory / (name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    arrays = {"parameters": host_array(parameters), "basis": host_array(state.basis),
              "inverse": host_array(state.inverse), "count": host_array(state.count),
              "refresh_step": host_array(state.refresh_step)}
    with (temporary / "state.npz").open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    metadata = {**metadata, "schema": 1, "state_sha256": sha256(temporary / "state.npz")}
    atomic_json(temporary / "manifest.json", metadata)
    if final.exists():
        # Re-saving an interruption at the same completed step is safe only
        # when the previously committed arrays agree.
        old = json.loads((final / "manifest.json").read_text())
        if old["state_sha256"] != metadata["state_sha256"]:
            raise ValueError("Conflicting checkpoint at the same step")
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, final)
    atomic_json(directory / "latest.json", {"checkpoint": name})
    completed = sorted(p for p in directory.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
    for old in completed[:-keep]:
        shutil.rmtree(old)
    return final


def load(directory, config_hash, data_hash, prior_hash, source_hashes=None):
    directory = Path(directory)
    candidates = sorted((p for p in directory.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp")), reverse=True)
    errors = []
    for path in candidates:
        try:
            metadata = json.loads((path / "manifest.json").read_text())
            if metadata["schema"] != 1 or sha256(path / "state.npz") != metadata["state_sha256"]:
                raise ValueError("Checkpoint hash/schema mismatch")
            for key, expected in (("config_hash", config_hash), ("data_hash", data_hash), ("prior_hash", prior_hash)):
                if metadata[key] != expected:
                    raise ValueError(f"Checkpoint {key} differs from the requested experiment")
            if source_hashes is not None and metadata["source_sha256"] != source_hashes:
                raise ValueError("Numerical source changed since checkpoint")
            with np.load(path / "state.npz", allow_pickle=False) as record:
                arrays = {name: record[name].copy() for name in record.files}
            if any(not np.all(np.isfinite(a)) for a in arrays.values()):
                raise ValueError("Checkpoint contains nonfinite state")
            return arrays, metadata, {"loaded": path.name, "rejected_generations": errors}
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append({"checkpoint": path.name, "reason": str(error)})
    raise ValueError(f"No compatible intact checkpoint: {errors}")
