"""Experiment and sweep controls for nominal 125M and sweep trials."""
from dataclasses import dataclass
import json
from pathlib import Path
from .config import ModelConfig, OptimizerConfig, RunConfig, from_dict
from .competitors import BaselineConfig


@dataclass(frozen=True)
class SweepRunConfig(RunConfig):
    training_steps: int = None

    def __post_init__(self):
        if self.training_steps is None:
            step_tokens = self.global_batch_size * self.model.sequence_length
            object.__setattr__(self, "training_steps", self.token_budget // step_tokens)

    def validate(self):
        super().validate()
        if self.training_steps < 1:
            raise ValueError("Training steps must be positive")
        if self.token_budget != self.training_steps * self.global_batch_size * self.model.sequence_length:
            raise ValueError("Sweep steps and exact predicted-token budget disagree")
        if self.compute_dtype != "bfloat16" or self.attention != "reference" or self.rematerialize or not self.reuse_linearization:
            raise ValueError("Sweep trials use standard XLA, BF16 training, reference attention, and reused linearization")
        if isinstance(self.optimizer, BaselineConfig) and self.optimizer.name == "sophia_g" and self.curvature_sequence_length != self.model.sequence_length:
            raise ValueError("Sophia-G curvature samples must cover the full model context")


def optimizer_name(config):
    return getattr(config.optimizer, "name", "tomography")


def run_from_dict(value):
    value = dict(value)
    value["model"] = ModelConfig(**value.get("model", {}))
    optimizer = dict(value.get("optimizer", {}))
    name = optimizer.pop("name", "tomography")
    value["optimizer"] = OptimizerConfig(**optimizer) if name == "tomography" else BaselineConfig(name=name, **optimizer)
    value.pop("phase", None)
    config_class = SweepRunConfig if "training_steps" in value else RunConfig
    config = config_class(**value)
    config.validate()
    return config


def load_run_config(path):
    return run_from_dict(json.loads(Path(path).read_text()))
