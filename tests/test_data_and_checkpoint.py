from dataclasses import replace
import json
import numpy as np
import pytest
from curvature_compass.lm import checkpoint
from curvature_compass.lm.artifacts import atomic_json, sha256
from curvature_compass.lm.calibrate import fit_pca, load_prior
from curvature_compass.lm.config import ModelConfig, OptimizerConfig, RunConfig, load_config
from curvature_compass.lm.data import TokenBlocks, document_key, split_for, verify_data
from curvature_compass.lm.model import Decoder
from curvature_compass.lm.optim import TomographyState


def test_token_blocks_exact_budget_no_replacement_and_resume_cursor(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(54, dtype="<u2").tofile(path)
    reader = TokenBlocks(path, 8, budget=43, seed=17)
    predictions = []
    for cursor in range(0, len(reader.order), 4):
        x, y, mask = reader.batch(cursor, 4)
        predictions.extend(y[mask.astype(bool)].tolist())
        np.testing.assert_array_equal(y[mask.astype(bool)], x[mask.astype(bool)] + 1)
        left = reader.batch(cursor, 4, 0, 2)
        right = reader.batch(cursor, 4, 2, 2)
        for actual, a, b in zip((x, y, mask), left, right):
            np.testing.assert_array_equal(actual, np.concatenate([a, b]))
    assert sorted(predictions) == list(range(1, 44))
    restarted = TokenBlocks(path, 8, budget=43, seed=17)
    np.testing.assert_array_equal(reader.batch(4, 4)[0], restarted.batch(4, 4)[0])
    with pytest.raises(ValueError):
        TokenBlocks(path, 8, budget=54)


def test_data_hash_and_document_split_integrity(fixture_data):
    verify_data(fixture_data)
    assert split_for(document_key("same exact document")) == split_for(document_key("same exact document"))
    path = fixture_data / "train.bin"
    content = bytearray(path.read_bytes()); content[0] ^= 1; path.write_bytes(content)
    with pytest.raises(ValueError, match="integrity"):
        verify_data(fixture_data)


def test_frozen_pca_frame_and_layout_integrity(tmp_path):
    model = Decoder(ModelConfig(vocab_size=23, padded_vocab_size=32, width=8, layers=1, heads=2, mlp_width=16, sequence_length=8))
    rows = np.random.default_rng(17).normal(size=(8, model.parameter_count)).astype(np.float32)
    basis, values = fit_pca(rows, 4)
    np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=2e-6)
    assert np.all(np.diff(values) <= 0)
    path = tmp_path / "prior.npy"; np.save(path, basis)
    atomic_json(path.with_suffix(".json"), {"schema": 1, "model_fingerprint": model.fingerprint, "rank": 4,
                "data_manifest_sha256": "data", "basis_sha256": sha256(path)})
    loaded, _ = load_prior(path, model, 4, "data")
    np.testing.assert_array_equal(loaded, basis)
    with pytest.raises(ValueError, match="layout"):
        load_prior(path, Decoder(replace(model.config, layers=2)), 4, "data")
    with pytest.raises(ValueError, match="different prepared"):
        load_prior(path, model, 4, "wrong")


def test_atomic_checkpoint_rollback_and_experiment_binding(tmp_path):
    state = TomographyState(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32), np.array(4, np.int32), np.array(0, np.int32))
    base = {"config_hash": "c", "data_hash": "d", "prior_hash": "p", "source_sha256": {"file": "h"}}
    p = np.arange(4, dtype=np.float32)
    for step in (1, 2):
        checkpoint.save(tmp_path, p + step, state, {**base, "step": step})
    arrays, metadata, recovery = checkpoint.load(tmp_path, "c", "d", "p", base["source_sha256"])
    np.testing.assert_array_equal(arrays["parameters"], p + 2)
    assert metadata["step"] == 2
    (tmp_path / "step-00000002/state.npz").write_bytes(b"interrupted checkpoint")
    arrays, metadata, recovery = checkpoint.load(tmp_path, "c", "d", "p", base["source_sha256"])
    assert metadata["step"] == 1 and recovery["rejected_generations"]
    (tmp_path / "step-00000003.tmp").mkdir()
    assert checkpoint.load(tmp_path, "c", "d", "p")[1]["step"] == 1
    with pytest.raises(ValueError, match="No compatible"):
        checkpoint.load(tmp_path, "wrong", "d", "p")


def test_four_configs_preserve_controls_and_independent_initializations():
    configs = [load_config(f"configs/lm60m_{cell}.json") for cell in "ABCD"]
    assert len({c.seed for c in configs}) == 4
    common = []
    for c in configs:
        values = c.to_dict(); values.pop("seed"); values["optimizer"].pop("configuration")
        common.append(values)
        assert .9e9 <= c.token_budget <= 1.1e9
    assert all(c == common[0] for c in common)
