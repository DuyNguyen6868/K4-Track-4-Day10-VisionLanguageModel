"""Parameter counting helper used to sanity-check LoRA targets."""

from __future__ import annotations

from typing import Iterable


def count_params(parameters: Iterable[object]) -> int:
    return sum(1 for _ in parameters)
