from __future__ import annotations

import polars as pl

WINDOW_SIZE = 2016  # 12 intervals * 24 hours * 7 days
START_TIMESTAMP = 1735689600000  # Jan 1, 2025 00:00:00 UTC
WEEKS_2025 = 52


def build_test_indices(reference_df: pl.DataFrame) -> list[int]:
    week_ms = WINDOW_SIZE * 5 * 60 * 1000
    target_timestamps = [
        START_TIMESTAMP + (week_ms * i) for i in range(WEEKS_2025)
    ]

    reference = reference_df.select("timestamp").with_row_index("idx")
    reference = reference.sort("timestamp")
    targets = pl.DataFrame({"timestamp": target_timestamps}).sort("timestamp")

    forward = targets.join_asof(
        reference,
        on="timestamp",
        strategy="forward",
    )
    if forward["idx"].null_count() > 0:
        backward = targets.join_asof(
            reference,
            on="timestamp",
            strategy="backward",
        )
        indices = forward["idx"].fill_null(backward["idx"])
    else:
        indices = forward["idx"]

    if indices.null_count() > 0:
        raise ValueError("Unable to map all weekly timestamps to indices")

    return indices.cast(pl.Int64).to_list()


def build_week_keys(total_weeks: int) -> list[str]:
    return [f"week{idx}" for idx in range(1, total_weeks + 1)]
