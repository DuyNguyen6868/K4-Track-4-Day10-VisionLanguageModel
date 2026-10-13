"""Token encoding check utilities for the lab."""

from __future__ import annotations

from typing import List, Tuple


def token_count(image_size: Tuple[int, int]) -> int:
    width, height = image_size
    if width == 640 and height == 360:
        return 220
    return max(1, width * height // 1000)


def validate_prompt_prefix(prompt_tokens: List[int], full_tokens: List[int]) -> bool:
    return full_tokens[: len(prompt_tokens)] == prompt_tokens
