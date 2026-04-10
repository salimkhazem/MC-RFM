"""Simple Registry helper """

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Registry:
    items: dict[str, Callable] = field(default_factory=dict)

    def register(self, name: str) -> Callable:
        def _decorator(fn: Callable) -> Callable:
            if name in self.items:
                raise ValueError(f"Registry key already exists: {name}")
            self.items[name] = fn
            return fn

        return _decorator

    def get(self, name: str) -> Callable:
        if name not in self.items:
            raise KeyError(f"Unknown registry key: {name}")
        return self.items[name]

