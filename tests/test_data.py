"""Comprehensive tests for canonical LLaMA tokenizer, modern salt, and TokenBlocks batching."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from curvature_compass.lm.data import (
    BOS, EOS, PAD, PADDED_VOCAB_SIZE, SPLIT_SALT, TOKENIZER, TOKENIZER_REVISION,
    UNK, VOCAB_SIZE, TokenBlocks, document_key, split_for, verify_data
)


def test_llama_tokenizer_constants_and_special_tokens():
    """Verify pinned un-gated modern LLaMA tokenizer constants."""
    assert TOKENIZER == "hf-internal-testing/llama-tokenizer"
    assert TOKENIZER_REVISION == "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
    assert VOCAB_SIZE == 32000
    assert PADDED_VOCAB_SIZE == 32768
    assert PADDED_VOCAB_SIZE % 128 == 0
    assert UNK == 0
    assert BOS == 1
    assert EOS == 2
    assert PAD == 2


def test_split_salt_has_zero_legacy_phase_terminology():
    """Verify SPLIT_SALT contains no legacy phase keywords and partitions correctly."""
    assert b"phase" not in SPLIT_SALT
    assert SPLIT_SALT == b"curvature-compass-v1\0"

    # Test split stability and distribution across 2000 synthetic documents
    counts = {"train": 0, "validation": 0, "calibration": 0}
    for i in range(2000):
        key = document_key(f"doc_{i}")
        split = split_for(key)
        counts[split] += 1

    # Validation & calibration expect ~0.5% (approx 5-25 out of 2000)
    assert 2 <= counts["validation"] <= 30
    assert 2 <= counts["calibration"] <= 30
    assert counts["train"] > 1900


def test_token_blocks_padding_with_eos_and_mask_accounting(tmp_path):
    """Verify next-token prediction alignment, padding with EOS/PAD, and mask accounting."""
    path = tmp_path / "tokens.bin"
    raw = np.array([1, 10, 20, 2, 1, 30, 40, 2, 1, 50, 60, 70, 80, 2], dtype="<u2")
    raw.tofile(path)

    # Budget of 13 predicted tokens with sequence length 4 -> 4 blocks of size 4
    reader = TokenBlocks(path, sequence_length=4, budget=13, eos=EOS, pad=PAD)

    # Fetch batch of 4 blocks
    x, y, mask = reader.batch(cursor=0, global_size=4)

    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    assert mask.shape == (4, 4)

    # Check valid predictions: where mask is 1, y must equal next token
    assert int(mask.sum()) == 13
    valid_x = x[mask.astype(bool)]
    valid_y = y[mask.astype(bool)]
    assert len(valid_x) == 13 and len(valid_y) == 13

    # Check padding positions: where mask is 0, x and y must be PAD/EOS (2)
    padding_x = x[~mask.astype(bool)]
    padding_y = y[~mask.astype(bool)]
    assert np.all(padding_x == PAD)
    assert np.all(padding_y == PAD)


def test_token_blocks_reproducible_shuffling_and_sharded_slicing(tmp_path):
    """Verify seed shuffling and multi-worker local slicing."""
    path = tmp_path / "stream.bin"
    np.arange(100, dtype="<u2").tofile(path)

    reader1 = TokenBlocks(path, sequence_length=8, budget=80, seed=42)
    reader2 = TokenBlocks(path, sequence_length=8, budget=80, seed=42)

    # Verify identical batch with identical seed
    x1, y1, m1 = reader1.batch(0, 4)
    x2, y2, m2 = reader2.batch(0, 4)
    np.testing.assert_array_equal(x1, x2)

    # Verify local partitioning (worker 0 gets 0:2, worker 1 gets 2:4)
    left_x, left_y, left_m = reader1.batch(0, 4, local_start=0, local_size=2)
    right_x, right_y, right_m = reader1.batch(0, 4, local_start=2, local_size=2)

    np.testing.assert_array_equal(np.concatenate([left_x, right_x]), x1)
    np.testing.assert_array_equal(np.concatenate([left_y, right_y]), y1)
    np.testing.assert_array_equal(np.concatenate([left_m, right_m]), m1)
