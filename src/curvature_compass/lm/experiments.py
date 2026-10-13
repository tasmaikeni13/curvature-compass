"""Phase 4 controls, without changing the frozen Phase 3 numerical configs."""
from dataclasses import dataclass
import json
from pathlib import Path
from .config import ModelConfig, OptimizerConfig, RunConfig, from_dict
from .competitors import BaselineConfig


@dataclass(frozen=True)
class SweepRunConfig(RunConfig):
    phase: int = 4
    training_steps: int = 2500

    def validate(self):
        super().validate()
        if self.phase != 4 or self.training_steps != 2500:
            raise ValueError("Phase 4 trials require exactly 2,500 training steps")
        if self.token_budget != self.training_steps * self.global_batch_size * self.model.sequence_length:
            raise ValueError("Sweep steps and exact predicted-token budget disagree")
        if self.compute_dtype != "bfloat16" or self.attention != "reference" or self.rematerialize or not self.reuse_linearization:
            raise ValueError("Phase 4 uses standard XLA, BF16 training, reference attention, and reused linearization")
        if isinstance(self.optimizer, BaselineConfig) and self.optimizer.name == "sophia_g" and self.curvature_sequence_length != self.model.sequence_length:
            raise ValueError("Sophia-G curvature samples must cover the full model context")


def optimizer_name(config):
    return getattr(config.optimizer, "name", "tomography")


def run_from_dict(value):
    if "phase" not in value:
        return from_dict(value)
    value = dict(value)
    value["model"] = ModelConfig(**value.get("model", {}))
    optimizer = dict(value.get("optimizer", {}))
    name = optimizer.pop("name", "tomography")
    value["optimizer"] = OptimizerConfig(**optimizer) if name == "tomography" else BaselineConfig(name=name, **optimizer)
    config = SweepRunConfig(**value)
    config.validate()
    return config


def load_run_config(path):
    return run_from_dict(json.loads(Path(path).read_text()))
