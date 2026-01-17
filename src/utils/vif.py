from typing import Iterable
import numpy as np
import polars as pl


CORE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def calculate_vif_fast(
    data: pl.DataFrame | pl.LazyFrame,
    cols: Iterable[str] | None = None,
    ridge: float = 1e-8,
    ddof: int = 1,
    validate: bool = True,
) -> pl.DataFrame:
    """
    Fast VIF via diag(inv(correlation)).
    Drops rows with NaN values.
    Drops constant cols; excludes intercept.
    """

    # Drop core columns (not part of VIF feature selection)
    data = data.drop([c for c in CORE_COLS if c in data.columns])

    lf = data.lazy() if isinstance(data, pl.DataFrame) else data

    # Use selectors to pick numeric columns (optionally restricted by `cols`)
    if cols is None:
        lf_num = lf.select(pl.selectors.numeric().cast(pl.Float64))
    else:
        schema = lf.collect_schema()
        cols = [c for c in cols if c in schema]  # keep only existing
        if not cols:
            return pl.DataFrame({"feature": [], "VIF": []})
        lf_subset = lf.select([pl.col(c) for c in cols])
        lf_num = lf_subset.select(pl.selectors.numeric().cast(pl.Float64))

    candidate_cols = list(lf_num.collect_schema().keys())
    if not candidate_cols:
        return pl.DataFrame({"feature": [], "VIF": []})

    # Drop any row that contains NaN in any column
    lf_num = lf_num.filter(~pl.any_horizontal(pl.all().is_nan()))

    if validate:
        # Scalar check (1x1) — row-wise any, then column-wise any
        bad_any = (
            lf_num
            .select(
                pl.any_horizontal(
                    [~pl.col(c).is_finite().fill_null(False) for c in candidate_cols]
                )
                .any()
                .alias("bad")
            )
            .collect()
            .item()
        )
        if bad_any:
            raise ValueError("Non-finite values detected. Clean upstream or call with validate=False.")

    # Means, stds (unbiased by default), and n
    stats = (
        lf_num
        .select(
            *[pl.col(c).mean().alias(f"mean_{c}") for c in candidate_cols],
            *[pl.col(c).std(ddof=ddof).alias(f"std_{c}") for c in candidate_cols],
            pl.len().alias("_n"),
        )
        .collect()
        .to_dicts()[0]
    )
    n = int(stats["_n"])
    if n < 3:
        return pl.DataFrame({"feature": [], "VIF": []})

    means = {c: stats[f"mean_{c}"] for c in candidate_cols}
    stds = {c: stats[f"std_{c}"] for c in candidate_cols}

    # Drop constants
    keep_cols = [c for c in candidate_cols if np.isfinite(stds[c]) and stds[c] > 0.0]
    if not keep_cols:
        return pl.DataFrame({"feature": [], "VIF": []})

    # Standardize lazily, then collect once
    lf_z = lf_num.select([(pl.col(c) - means[c]) / stds[c] for c in keep_cols])
    Xz = lf_z.collect().to_numpy()  # (n, p)

    denom = max(n - ddof, 1)
    R = (Xz.T @ Xz) / denom

    if ridge and ridge > 0.0:
        R = R + ridge * np.eye(R.shape[0], dtype=R.dtype)

    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(R, rcond=1e-8)

    vifs = np.diag(Rinv)
    return pl.DataFrame({"feature": keep_cols, "VIF": vifs})


def remove_high_vif(
    df: pl.DataFrame,
    max_vif: float = 10.0,
    label: str | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    work_df = df.clone()
    if label:
        print(f"Processing {label}")
    while True:
        vif_df = calculate_vif_fast(work_df)
        if vif_df.is_empty():
            break

        max_feature, max_value = (
            vif_df.sort("VIF", descending=True).row(0)
        )
        if max_value <= max_vif:
            break

        if verbose:
            print(f"Dropping {max_feature} (VIF={max_value:.4f})")
        work_df = work_df.drop(max_feature)

    return calculate_vif_fast(work_df)
    