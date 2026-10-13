"""Independent frozen-model gradient PCA; calibration never trains A/B/C/D."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
from .artifacts import atomic_json, environment, sha256, source_hashes
from .config import load_config
from .data import TokenBlocks, verify_data
from .model import Decoder
from .runtime import global_batch, host_array, initialize, mesh_and_shardings, sync


def gram_rows(rows, chunk=262144):
    gram = np.zeros((rows.shape[0], rows.shape[0]), dtype=np.float64)
    for start in range(0, rows.shape[1], chunk):
        piece = np.asarray(rows[:, start:start + chunk], dtype=np.float64)
        gram += piece @ piece.T
    return gram


def fit_pca(rows, rank):
    values, vectors = np.linalg.eigh(gram_rows(rows))
    selected = np.argsort(values)[::-1][:rank]
    if len(selected) != rank or values[selected[-1]] <= max(values[-1], 1e-30) * 1e-10:
        raise ValueError("Calibration gradients do not support the requested fixed rank")
    weights = (vectors[:, selected] / np.sqrt(values[selected])).T
    frame = np.empty((rank, rows.shape[1]), dtype=np.float32)
    for start in range(0, rows.shape[1], 262144):
        frame[:, start:start + 262144] = weights @ np.asarray(rows[:, start:start + 262144], dtype=np.float64)
    # Fix otherwise arbitrary eigenvector signs for deterministic artifacts.
    for row in frame:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    if not np.allclose(gram_rows(frame), np.eye(rank), atol=2e-5):
        raise ValueError("PCA frame failed orthogonality check")
    return frame, values[selected]


def load_prior(path, decoder, cap, data_hash=None):
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text())
    if record["schema"] != 1 or record["model_fingerprint"] != decoder.fingerprint or record["rank"] < cap:
        raise ValueError("PCA prior does not match model tensor layout/rank")
    if data_hash is not None and record["data_manifest_sha256"] != data_hash:
        raise ValueError("PCA prior uses a different prepared dataset")
    if sha256(path) != record["basis_sha256"]:
        raise ValueError("PCA prior integrity failure")
    frame = np.load(path, mmap_mode="r", allow_pickle=False)
    if frame.shape != (record["rank"], decoder.parameter_count) or not np.all(np.isfinite(frame)):
        raise ValueError("Invalid fixed PCA frame")
    if not np.allclose(gram_rows(frame[:cap]), np.eye(cap), atol=2e-5):
        raise ValueError("Prior frame is not orthonormal")
    return frame[:cap], record


def calibrate(config, data, output, seed=30900, snapshots=16, batch_size=16, sequence_length=128):
    import jax
    decoder = Decoder(config.model)
    if seed in range(31001, 31005) or seed == config.seed:
        raise ValueError("Prior initialization must be independent of the four run seeds")
    manifest = verify_data(data, full=False)
    data_hash = sha256(Path(data) / "manifest.json")
    reader = TokenBlocks(Path(data) / "calibration.bin", sequence_length, seed=33000)
    if snapshots < config.optimizer.cap or snapshots * batch_size > len(reader.order):
        raise ValueError("Insufficient independent calibration minibatches/snapshots")
    mesh, replicated, batch_sharding = mesh_and_shardings()
    parameters = jax.device_put(decoder.initialize(seed), replicated)
    grad = jax.jit(jax.grad(lambda p, b: decoder.loss(p, b)),
                   in_shardings=(replicated, (batch_sharding,) * 3), out_shardings=replicated)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Refusing to replace an already frozen PCA prior")
    rows = None
    if jax.process_index() == 0:
        rows = np.lib.format.open_memmap(output.with_suffix(".snapshots.npy"), mode="w+", dtype=np.float32,
                                         shape=(snapshots, decoder.parameter_count))
    started, timings = time.perf_counter(), []
    for i in range(snapshots):
        batch = global_batch(reader, i * batch_size, batch_size, batch_sharding)
        t = time.perf_counter()
        gradient = grad(parameters, batch)
        gradient.block_until_ready()
        timings.append(time.perf_counter() - t)
        if rows is not None:
            rows[i] = host_array(gradient)
            rows.flush()
        if jax.process_index() == 0:
            print(f"calibration snapshot {i + 1}/{snapshots}: {timings[-1]:.3f}s", flush=True)
    if rows is not None:
        frame, eigenvalues = fit_pca(rows, config.optimizer.cap)
        with output.with_suffix(".npy.tmp").open("wb") as stream:
            np.save(stream, frame, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(output.with_suffix(".npy.tmp"), output)
        record = {"schema": 1, "seed": seed, "rank": config.optimizer.cap, "snapshots": snapshots,
                  "batch_size": batch_size, "sequence_length": sequence_length, "data_split": "calibration",
                  "model": asdict(config.model), "model_fingerprint": decoder.fingerprint,
                  "parameter_count": decoder.parameter_count, "data_manifest_sha256": data_hash,
                  "basis_sha256": sha256(output), "eigenvalues": eigenvalues.tolist(),
                  "orthogonality_max_error": float(np.max(np.abs(gram_rows(frame) - np.eye(len(frame))))),
                  "snapshot_seconds": timings, "seconds": time.perf_counter() - started,
                  "source_sha256": source_hashes(), "environment": environment(),
                  "method": "Uncentered PCA of independent minibatch gradients at one separately initialized frozen model; no training/validation examples and no model updates."}
        atomic_json(output.with_suffix(".json"), record)
        del rows
        output.with_suffix(".snapshots.npy").unlink()
    sync("calibration-complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=30900)
    parser.add_argument("--snapshots", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    args = parser.parse_args()
    initialize(args.distributed, args.coordinator, args.processes, args.process_id)
    calibrate(load_config(args.config), args.data, args.output, args.seed, args.snapshots, args.batch_size, args.sequence_length)


if __name__ == "__main__":
    main()
