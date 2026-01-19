from typing import Iterable
import numpy as np
import polars as pl


CORE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _prepare_vif_matrix(
    data: pl.DataFrame | pl.LazyFrame,
    cols: Iterable[str] | None = None,
    ddof: int = 1,
    validate: bool = True,
) -> tuple[np.ndarray, list[str]] | None:
    """Standardize features and return (Xz, feature_names) for VIF."""
    data = data.drop([c for c in CORE_COLS if c in data.columns])
    lf = data.lazy() if isinstance(data, pl.DataFrame) else data

    if cols is None:
        lf_num = lf.select(pl.selectors.numeric().cast(pl.Float64))
    else:
        schema = lf.collect_schema()
        cols = [c for c in cols if c in schema]
        if not cols:
            return None
        lf_subset = lf.select([pl.col(c) for c in cols])
        lf_num = lf_subset.select(pl.selectors.numeric().cast(pl.Float64))

    candidate_cols = list(lf_num.collect_schema().keys())
    if not candidate_cols:
        return None

    lf_num = lf_num.filter(~pl.any_horizontal(pl.all().is_nan()))
    df_num = lf_num.collect()
    if df_num.height < 3:
        return None

    X = df_num.to_numpy()
    if validate and not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values detected. Clean upstream or call with validate=False."
        )

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=ddof)
    keep_mask = np.isfinite(stds) & (stds > 0.0)
    if not keep_mask.any():
        return None

    X = X[:, keep_mask]
    means = means[keep_mask]
    stds = stds[keep_mask]
    Xz = (X - means) / stds
    keep_cols = [candidate_cols[i] for i in np.flatnonzero(keep_mask)]
    return Xz, keep_cols


def _vif_from_standardized(
    Xz: np.ndarray,
    ridge: float = 1e-8,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VIF via diag(inv(corr)) from standardized X."""
    n = Xz.shape[0]
    denom = max(n - ddof, 1)
    # Correlation from standardized design matrix.
    R = (Xz.T @ Xz) / denom
    if ridge and ridge > 0.0:
        R = R + ridge * np.eye(R.shape[0], dtype=R.dtype)
    try:
        Rinv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Rinv = np.linalg.pinv(R, rcond=1e-8)
    vifs = np.diag(Rinv)
    return vifs, Rinv


def stepwise_vif_prune(
    data: pl.DataFrame | pl.LazyFrame,
    max_vif: float = 10.0,
    cols: Iterable[str] | None = None,
    ridge: float = 1e-8,
    ddof: int = 1,
    validate: bool = True,
) -> list[str]:
    """
    Stepwise elimination using VIF = diag(inv(corr)).

    Updates the precision matrix via the Schur complement when dropping
    the max-VIF feature (O(p^2) per drop instead of re-inverting).
    """
    prepared = _prepare_vif_matrix(
        data,
        cols=cols,
        ddof=ddof,
        validate=validate,
    )
    if prepared is None:
        return []
    Xz, keep_cols = prepared

    vifs, precision = _vif_from_standardized(Xz, ridge=ridge, ddof=ddof)
    keep_indices = list(range(len(keep_cols)))

    while len(keep_indices) > 1:
        safe_vifs = np.where(np.isfinite(vifs), vifs, np.inf)
        max_idx = int(np.argmax(safe_vifs))
        if safe_vifs[max_idx] <= max_vif:
            break

        # Schur complement update after removing feature max_idx.
        pii = precision[max_idx, max_idx]
        if not np.isfinite(pii) or pii == 0.0:
            Xz_reduced = Xz[:, keep_indices]
            vifs, precision = _vif_from_standardized(
                Xz_reduced, ridge=ridge, ddof=ddof
            )
            safe_vifs = np.where(np.isfinite(vifs), vifs, np.inf)
            max_idx = int(np.argmax(safe_vifs))
            if safe_vifs[max_idx] <= max_vif:
                break
            pii = precision[max_idx, max_idx]

        mask = np.ones(len(keep_indices), dtype=bool)
        mask[max_idx] = False
        col = precision[:, max_idx]
        row = precision[max_idx, :]
        precision = precision[np.ix_(mask, mask)] - np.outer(
            col[mask], row[mask]
        ) / pii
        keep_indices = [idx for idx, keep in zip(keep_indices, mask) if keep]
        vifs = np.diag(precision)

    return [keep_cols[idx] for idx in keep_indices]


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
    prepared = _prepare_vif_matrix(
        data,
        cols=cols,
        ddof=ddof,
        validate=validate,
    )
    if prepared is None:
        return pl.DataFrame({"feature": [], "VIF": []})

    Xz, keep_cols = prepared
    vifs, _ = _vif_from_standardized(Xz, ridge=ridge, ddof=ddof)
    return pl.DataFrame({"feature": keep_cols, "VIF": vifs})


def remove_high_vif(
    df: pl.DataFrame,
    max_vif: float = 10.0,
    label: str | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    if label:
        print(f"Processing {label}")
    kept_features = stepwise_vif_prune(df, max_vif=max_vif)
    if not kept_features:
        return pl.DataFrame({"feature": [], "VIF": []})
    work_df = df.select(kept_features)
    vif_df = calculate_vif_fast(work_df)
    if verbose and not vif_df.is_empty():
        max_feature, max_value = vif_df.sort("VIF", descending=True).row(0)
        print(f"Max remaining VIF {max_feature}={max_value:.4f}")
    return vif_df
    