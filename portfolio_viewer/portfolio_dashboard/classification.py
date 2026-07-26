from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Classification:
    id: str
    label: str
    subgroup: str = ""
    badge: str = ""


def load_classification_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("groups"), list):
        raise ValueError(f"{path} 缺少 groups 数组")
    if not isinstance(data.get("fallback"), dict):
        raise ValueError(f"{path} 缺少 fallback 对象")
    return data


def classify_symbol(
    market: str,
    symbol: str,
    name: str,
    config: dict[str, Any],
) -> Classification:
    del market
    candidates = {symbol.upper(), name.upper()}
    for group in config["groups"]:
        members = group.get("members", {})
        for member, subgroup in members.items():
            if member.upper() in candidates:
                return Classification(
                    id=group["id"],
                    label=group["label"],
                    subgroup=subgroup,
                )

    fallback = config["fallback"]
    return Classification(
        id=fallback["id"],
        label=fallback["label"],
        badge=fallback.get("badge", ""),
    )

