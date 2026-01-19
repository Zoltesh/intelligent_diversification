from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

CORE_COLS = {"timestamp", "open", "high", "low", "close", "volume"}
TARGET_COLS = {"target_return_1w"}


@dataclass(frozen=True)
class FeatureSet:
    weekly_features: dict[str, dict[str, list[str]]]


class FeatureStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def feature_path(self, symbol: str) -> Path:
        return self.base_dir / f"{symbol.lower()}_features.json"

    def load_feature_set(self, symbol: str) -> dict[str, list[str]]:
        path = self.feature_path(symbol)
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}
        return {
            str(week): list(cols) for week, cols in data.items() if cols
        }

    def save_feature_set(self, symbol: str, weekly_features: dict[str, list[str]]) -> None:
        path = self.feature_path(symbol)
        path.write_text(json.dumps(weekly_features, indent=2))

    def base_features(self, df: pl.DataFrame) -> list[str]:
        return [
            col
            for col in df.columns
            if col not in CORE_COLS and col not in TARGET_COLS
        ]

    def resolve_weekly_features(
        self,
        symbol: str,
        df: pl.DataFrame,
        week_keys: Iterable[str],
        read_existing: bool = True,
    ) -> dict[str, list[str]]:
        existing = self.load_feature_set(symbol) if read_existing else {}
        base_features = self.base_features(df)
        resolved: dict[str, list[str]] = {}
        for week_key in week_keys:
            features = existing.get(week_key) if existing else None
            resolved[week_key] = features if features else base_features
        return resolved
