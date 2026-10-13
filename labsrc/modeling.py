"""Modeling scaffolding for Qwen3-VL fine-tuning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExperimentConfig:
    name: str
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    train_projector: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    max_steps: int = 300
    batch_size: int = 2
    grad_accum: int = 4
    seed: int = 42


def build_experiment(name: str, **overrides: Any) -> ExperimentConfig:
    cfg = ExperimentConfig(name=name)
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def describe_trainable_modules(setup: str) -> List[str]:
    if setup == "E1":
        return ["language_model.layers.*.self_attn.{q,k,v,o}_proj", "language_model.layers.*.mlp.{gate,up,down}_proj"]
    if setup == "E2":
        return ["language_model.layers.*.self_attn.{q,k,v,o}_proj", "language_model.layers.*.mlp.{gate,up,down}_proj", "visual.merger", "visual.deepstack_merger_list"]
    if setup == "E3":
        return [
            "language_model.layers.*.self_attn.{q,k,v,o}_proj",
            "language_model.layers.*.mlp.{gate,up,down}_proj",
            "visual.merger",
            "visual.deepstack_merger_list",
            "visual.blocks.*.attn.{qkv,proj}",
            "visual.blocks.*.mlp.{linear_fc1,linear_fc2}",
        ]
    return []


def validate_model_setup(setup: str, trainable_params: int) -> Dict[str, Any]:
    return {
        "setup": setup,
        "trainable_params": int(trainable_params),
        "required_modules": describe_trainable_modules(setup),
    }
