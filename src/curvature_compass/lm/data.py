"""Pinned Parquet-to-token preprocessing and deterministic, resumable blocks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from .artifacts import atomic_json, sha256

DATASET = "codelion/fineweb-edu-1B"
DATA_REVISION = "66b78cd134ab7b0545a1a618a1cc029f3091eb20"
TOKENIZER = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SPLIT_SALT = b"curvature-compass-phase23-v1\0"
EOS = 50256


def document_key(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def split_for(key):
    bucket = int.from_bytes(hashlib.sha256(SPLIT_SALT + key).digest()[:8], "big") % 1000
    return "validation" if bucket < 5 else "calibration" if bucket < 10 else "train"


def prepare(output, train_tokens=1_000_000_000, validation_tokens=1_048_576,
            calibration_tokens=524_288, max_documents=None):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite prepared data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokenizer_path = hf_hub_download(TOKENIZER, "tokenizer.json", revision=TOKENIZER_REVISION)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.get_vocab_size() != 50257 or tokenizer.token_to_id("<|endoftext|>") != EOS:
        raise ValueError("Unexpected pinned tokenizer vocabulary")
    targets = {"train": train_tokens + 1, "validation": validation_tokens + 1,
               "calibration": calibration_tokens + 1}
    counts = dict.fromkeys(targets, 0)
    observed = dict.fromkeys(targets, 0)
    docs = dict.fromkeys(targets, 0)
    seen, duplicate_count, rows = set(), 0, 0
    sources = []
    streams = {s: (output / (s + ".bin.tmp")).open("wb") for s in targets}
    keys = {s: (output / (s + ".documents.sha256.tmp")).open("wb") for s in targets}
    try:
        for index in range(10):
            name = f"data/train-{index:05d}-of-00010.parquet"
            path = hf_hub_download(DATASET, name, repo_type="dataset", revision=DATA_REVISION)
            sources.append({"name": name, "bytes": Path(path).stat().st_size, "sha256": sha256(path)})
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
                pending = []
                for text in batch.column(0).to_pylist():
                    if max_documents is not None and rows >= max_documents:
                        break
                    rows += 1
                    if not isinstance(text, str) or not text:
                        continue
                    key = document_key(text)
                    if key in seen:
                        duplicate_count += 1
                        continue
                    seen.add(key)
                    pending.append((text, key, split_for(key)))
                for (_, key, split), encoded in zip(pending, tokenizer.encode_batch([p[0] for p in pending], add_special_tokens=False)):
                    ids = encoded.ids + [EOS]
                    observed[split] += len(ids)
                    docs[split] += 1
                    remaining = targets[split] - counts[split]
                    if remaining > 0:
                        np.asarray(ids[:remaining], dtype="<u2").tofile(streams[split])
                        counts[split] += min(remaining, len(ids))
                        keys[split].write(key)
                if max_documents is not None and rows >= max_documents:
                    break
            print(f"prepared shard {index}: rows={rows:,}, stored tokens={counts}", flush=True)
            if max_documents is not None and rows >= max_documents:
                break
        for stream in list(streams.values()) + list(keys.values()):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        files = {}
        for split in targets:
            for suffix in (".bin", ".documents.sha256"):
                tmp, final = output / (split + suffix + ".tmp"), output / (split + suffix)
                os.replace(tmp, final)
                files[final.name] = {"bytes": final.stat().st_size, "sha256": sha256(final)}
        manifest = {"schema": 1, "dataset": DATASET, "dataset_revision": DATA_REVISION,
                    "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
                    "tokenizer_sha256": sha256(tokenizer_path), "vocab_size": 50257, "eos_id": EOS,
                    "dtype": "<u2", "split_salt": SPLIT_SALT.decode(), "sources": sources,
                    "rows_read": rows, "duplicate_documents": duplicate_count,
                    "complete_source_scan": max_documents is None, "max_documents": max_documents,
                    "documents_after_deduplication": docs, "observed_tokens": observed,
                    "stored_tokens": counts, "predicted_tokens": {k: max(0, v - 1) for k, v in counts.items()},
                    "target_predicted_tokens": {k: v - 1 for k, v in targets.items()},
                    "files": files, "seconds": time.perf_counter() - started,
                    "packing": "Continuous exact-text BPE with one EOS per document, shifted next-token targets; no BOS; block contexts may cross document boundaries."}
        atomic_json(output / "manifest.json", manifest)
        return manifest
    finally:
        for stream in list(streams.values()) + list(keys.values()):
            if not stream.closed:
                stream.close()


def verify_data(directory, full=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["dataset_revision"] != DATA_REVISION or manifest["tokenizer_revision"] != TOKENIZER_REVISION:
        raise ValueError("Unexpected data provenance/schema")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or (full and sha256(path) != record["sha256"]):
            raise ValueError(f"Data integrity failure: {path}")
    inventories = []
    for split in ("train", "validation", "calibration"):
        content = (directory / (split + ".documents.sha256")).read_bytes()
        if len(content) % 32:
            raise ValueError("Truncated document inventory")
        inventory = {content[i:i + 32] for i in range(0, len(content), 32)}
        if any(split_for(key) != split for key in inventory):
            raise ValueError("Document assigned to incorrect split")
        inventories.append(inventory)
    if any(inventories[i] & inventories[j] for i in range(3) for j in range(i)):
        raise ValueError("Document leakage between data splits")
    return manifest


class TokenBlocks:
    """A permutation of blocks, never a repeated or replacement token stream."""
    def __init__(self, path, sequence_length, budget=None, seed=None, eos=EOS):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.budget = len(self.tokens) - 1 if budget is None else budget
        if self.budget < 1 or self.budget + 1 > len(self.tokens):
            raise ValueError("Prepared split cannot supply requested predicted tokens")
        self.sequence_length = sequence_length
        self.eos = eos
        self.order = np.arange(math.ceil(self.budget / sequence_length), dtype=np.int64)
        if seed is not None:
            np.random.default_rng(seed).shuffle(self.order)

    def batch(self, cursor, global_size, local_start=0, local_size=None, prefix_length=None):
        size = global_size if local_size is None else local_size
        length = self.sequence_length if prefix_length is None else prefix_length
        if not 0 < length <= self.sequence_length or cursor < 0 or local_start < 0 or local_start + size > global_size:
            raise ValueError("Invalid block batch request")
        x = np.full((size, length), self.eos, dtype=np.int32)
        y = x.copy()
        mask = np.zeros((size, length), dtype=np.float32)
        for i in range(size):
            position = cursor + local_start + i
            if position >= len(self.order):
                continue
            start = int(self.order[position]) * self.sequence_length
            valid = min(length, self.budget - start)
            values = np.asarray(self.tokens[start:start + valid + 1], dtype=np.int32)
            x[i, :valid], y[i, :valid], mask[i, :valid] = values[:-1], values[1:], 1
        return x, y, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--validation-tokens", type=int, default=1_048_576)
    parser.add_argument("--calibration-tokens", type=int, default=524_288)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify_data(args.output) if args.verify else prepare(args.output, args.train_tokens, args.validation_tokens, args.calibration_tokens, args.max_documents)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
