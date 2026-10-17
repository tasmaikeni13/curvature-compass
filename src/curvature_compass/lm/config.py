"""Strict, serializable experimental controls."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    padded_vocab_size: int = 32768
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_width: int = 2560
    sequence_length: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    def validate(self):
        if min(asdict(self).values()) <= 0 or self.width % self.heads:
            raise ValueError("Positive dimensions and width divisible by heads required")
        if (self.width // self.heads) % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE")
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("Padded vocabulary cannot discard tokenizer classes")


@dataclass(frozen=True)
class OptimizerConfig:
    configuration: str = "D"
    cap: int = 8
    min_count: int = 1
    tolerance: float = .7
    damping: float = 1.
    eigenvalue_floor: float = .1
    refresh_interval: int = 16
    max_step_ratio: float = 10.

    def validate(self):
        if self.configuration not in "ABCD" or len(self.configuration) != 1:
            raise ValueError("Configuration must be A, B, C or D")
        if not 1 <= self.min_count <= self.cap or self.refresh_interval < 1:
            raise ValueError("Invalid acquisition count/refresh interval")
        if not 0 < self.tolerance < 1 or min(self.damping, self.eigenvalue_floor, self.max_step_ratio) <= 0:
            raise ValueError("Invalid stopping/safeguard controls")

    @property
    def adaptive_alignment(self):
        return self.configuration in "BD"

    @property
    def adaptive_count(self):
        return self.configuration in "CD"


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = 31004
    data_seed: int = 32000
    token_budget: int = 2_500_000_000
    global_batch_size: int = 64
    accumulation_steps: int = 1
    curvature_batch_size: int = 16
    curvature_sequence_length: int = 256
    learning_rate: float = .05
    warmup_steps: int = 200
    minimum_lr_ratio: float = .1
    weight_decay: float = .1
    gradient_clip: float = 1.
    validation_tokens: int = 1_048_576
    eval_interval: int = 100
    checkpoint_interval: int = 250
    keep_checkpoints: int = 2
    compute_dtype: str = "float32"
    attention: str = "reference"
    rematerialize: bool = False
    reuse_linearization: bool = True

    def validate(self):
        self.model.validate()
        self.optimizer.validate()
        for name in ("token_budget", "global_batch_size", "accumulation_steps", "curvature_batch_size",
                     "curvature_sequence_length", "validation_tokens", "eval_interval", "checkpoint_interval", "keep_checkpoints"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.global_batch_size % self.accumulation_steps:
            raise ValueError("Accumulation must divide the global batch")
        if self.curvature_batch_size > self.global_batch_size or self.curvature_sequence_length > self.model.sequence_length:
            raise ValueError("Curvature batch must be a prefix of the training batch")
        if self.compute_dtype not in ("float32", "bfloat16") or self.attention not in ("reference", "sdpa", "flash", "pallas"):
            raise ValueError("Unsupported execution option")
        if min(self.learning_rate, self.gradient_clip) <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("Invalid training controls")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("Invalid minimum learning rate ratio")

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.to_dict())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def from_dict(value):
    value = dict(value)
    value["model"] = ModelConfig(**value.get("model", {}))
    value["optimizer"] = OptimizerConfig(**value.get("optimizer", {}))
    config = RunConfig(**value)
    config.validate()
    return config


def load_config(path):
    return from_dict(json.loads(Path(path).read_text()))
