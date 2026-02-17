#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecosystem_analysis import (
    add_derived_metrics,
    flatten_model,
    load_payload,
    short_name,
)

EPS = 1e-12
RECURSIVE_PARADIGMS = {"hybrid", "ssm", "liquid", "rwkv", "xlstm"}
NAME_HINT_RE = re.compile(
    r"(mamba|zamba|falcon-h1|nemotron-h|granite-4\.0-h|rwkv|xlstm|lfm|liquid|hybrid)",
    re.IGNORECASE,
)


def safe_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def mean_pairwise_distance(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    if n < 2:
        return np.nan
    dists: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(matrix[i] - matrix[j])))
    return float(np.mean(dists)) if dists else np.nan


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def partial_corr(df: pd.DataFrame, x_col: str, y_col: str, controls: List[str]) -> Tuple[float, int]:
    cols = [x_col, y_col] + controls
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(work)
    if n < max(8, len(controls) + 3):
        return np.nan, n
    x_resid = residualize(work[x_col].to_numpy(dtype=float), work[controls].to_numpy(dtype=float))
    y_resid = residualize(work[y_col].to_numpy(dtype=float), work[controls].to_numpy(dtype=float))
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan, n
    return float(np.corrcoef(x_resid, y_resid)[0, 1]), n


def fit_standardized_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Tuple[pd.DataFrame, float, int]:
    work = df[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(work)
    if n < max(10, len(x_cols) + 3):
        return pd.DataFrame(), np.nan, n

    x = work[x_cols].astype(float)
    x = x.apply(safe_zscore, axis=0)
    y = safe_zscore(work[y_col])

    x_mat = np.column_stack([np.ones(n), x.to_numpy()])
    y_vec = y.to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    y_hat = x_mat @ beta
    resid = y_vec - y_hat

    dof = max(n - x_mat.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    t_vals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)

    ss_tot = float(((y_vec - y_vec.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - (ss_res / (ss_tot + EPS))

    coef = pd.DataFrame(
        {
            "term": ["intercept"] + x_cols,
            "beta_std": beta,
            "std_err": se,
            "t_value": t_vals,
            "n": n,
            "r2": r2,
        }
    )
    return coef, r2, n


def load_dataframe(input_path: Path) -> Tuple[dict, pd.DataFrame]:
    meta, models = load_payload(input_path)
    df = pd.DataFrame([flatten_model(m) for m in models])

    extras = []
    for m in models:
        ps = m.get("paradigm_specific") or {}
        extras.append(
            {
                "model_id": m.get("model_id"),
                "num_ssm_layers": ps.get("num_ssm_layers"),
                "num_attention_layers": ps.get("num_attention_layers"),
                "ssm_layer_ratio": ps.get("ssm_layer_ratio"),
                "attention_layer_ratio": ps.get("attention_layer_ratio"),
            }
        )
    extra_df = pd.DataFrame(extras)
    df = df.merge(extra_df, on="model_id", how="left")

    df = add_derived_metrics(df)
    return meta, df


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["num_ssm_layers", "num_attention_layers", "ssm_layer_ratio", "attention_layer_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latest_release = df["release_date"].max()
    if pd.isna(latest_release):
        df["recency_months"] = np.nan
    else:
        df["recency_months"] = (latest_release - df["release_date"]).dt.days / 30.44

    name_source = (
        df["model_id"].fillna("").astype(str)
        + " "
        + df["family"].fillna("").astype(str)
        + " "
        + df["org"].fillna("").astype(str)
    )
    df["name_ssm_hint"] = name_source.str.contains(NAME_HINT_RE).astype(float)
    df["has_attention"] = df["num_heads"].notna().astype(float)
    df["hard_hybrid_evidence"] = (df["has_ssm_signals"].fillna(False) & df["num_heads"].notna()).astype(float)

    layer_mix = (
        ((df["ssm_layer_ratio"] > 0) & (df["ssm_layer_ratio"] < 1))
        | ((df["num_ssm_layers"].fillna(0) > 0) & (df["num_attention_layers"].fillna(0) > 0))
    )
    df["layer_mix_evidence"] = layer_mix.astype(float)

    numeric_ssm = (
        df["has_mamba_key"].fillna(False)
        | df["has_ssm_key"].fillna(False)
        | (df["detected_numeric_count"].fillna(0) > 0)
    )
    df["numeric_ssm_evidence"] = numeric_ssm.astype(float)

    inferred_hybrid = df["inferred_paradigm"].eq("hybrid").astype(float)
    df["hybridization_score"] = (
        0.45 * df["hard_hybrid_evidence"]
        + 0.20 * df["name_ssm_hint"]
        + 0.15 * df["numeric_ssm_evidence"]
        + 0.10 * df["layer_mix_evidence"]
        + 0.10 * inferred_hybrid
    ).clip(0, 1)

    registry_non_recurrent = ~df["registry_paradigm"].isin(["hybrid", "ssm", "liquid", "rwkv", "xlstm"])
    df["hidden_hybrid_candidate"] = registry_non_recurrent & (df["hybridization_score"] >= 0.60)

    ffn_balance = 1.0 - (df["ffn_expansion"] - 4.0).abs() / 4.0
    df["ffn_balance"] = ffn_balance.clip(lower=0, upper=1)
    df["log_context_efficiency"] = np.log10(
        pd.to_numeric(df["context_per_param_b"], errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=1)
    )

    quality_inputs = pd.DataFrame(
        {
            "convergence_score": pd.to_numeric(df["convergence_score"], errors="coerce"),
            "kv_compression": pd.to_numeric(df["kv_compression"], errors="coerce"),
            "ffn_balance": pd.to_numeric(df["ffn_balance"], errors="coerce"),
            "log_context_efficiency": pd.to_numeric(df["log_context_efficiency"], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)
    quality_inputs = quality_inputs.fillna(quality_inputs.median(numeric_only=True))
    quality_z = quality_inputs.apply(safe_zscore, axis=0)
    df["architecture_quality_index"] = quality_z.mean(axis=1)

    return df


def add_adoption_residual(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    predictors = ["log10_params", "recency_months"]
    work = df[["log10_downloads"] + predictors].replace([np.inf, -np.inf], np.nan).dropna()

    df["adoption_residual"] = np.nan
    if len(work) < 8:
        return df, pd.DataFrame()

    x = np.column_stack([np.ones(len(work)), work[predictors].to_numpy(dtype=float)])
    y = work["log10_downloads"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    resid = y - y_hat
    df.loc[work.index, "adoption_residual"] = resid

    coef = pd.DataFrame({"term": ["intercept"] + predictors, "coef": beta, "n": len(work)})
    return df, coef


def compute_convergence_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "canonical_head_dim",
        "canonical_norm",
        "canonical_activation",
        "canonical_position",
        "kv_compression",
        "gqa_ratio",
        "ffn_expansion",
        "log10_context",
    ]
    work = df.dropna(subset=["release_month", "inferred_paradigm"]).copy()
    if work.empty:
        return pd.DataFrame()

    f = work[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    f = f.fillna(f.median(numeric_only=True))
    fz = f.apply(safe_zscore, axis=0).fillna(0.0)
    z_cols = [f"z_{c}" for c in feature_cols]
    work[z_cols] = fz.to_numpy()

    rows = []
    months = sorted(work["release_month"].dropna().unique())
    for month in months:
        sub = work[work["release_month"] <= month].copy()
        if len(sub) < 8:
            continue

        active = sub.groupby("inferred_paradigm")["model_id"].count()
        active = active[active >= 2].index
        sub = sub[sub["inferred_paradigm"].isin(active)]
        if sub["inferred_paradigm"].nunique() < 2:
            continue

        centroids = sub.groupby("inferred_paradigm")[z_cols].mean()
        between = mean_pairwise_distance(centroids.to_numpy())

        merged = sub.merge(centroids, left_on="inferred_paradigm", right_index=True, suffixes=("", "_c"))
        x = merged[z_cols].to_numpy()
        c = merged[[f"{col}_c" for col in z_cols]].to_numpy()
        within = float(np.mean(np.linalg.norm(x - c, axis=1)))

        overlap_ratio = within / (between + EPS)
        convergence_index = overlap_ratio / (1.0 + overlap_ratio)

        rows.append(
            {
                "release_month": month,
                "models_seen": len(sub),
                "active_paradigms": int(sub["inferred_paradigm"].nunique()),
                "between_distance": between,
                "within_distance": within,
                "overlap_ratio": overlap_ratio,
                "convergence_index": convergence_index,
            }
        )

    return pd.DataFrame(rows).sort_values("release_month")


def bucket_head_dim(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if np.isclose(v, 64):
        return "64"
    if np.isclose(v, 128):
        return "128"
    return "other"


def bucket_gqa(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v <= 1.01:
        return "mha"
    if v <= 4.0:
        return "gqa_mid"
    return "gqa_high"


def bucket_kv(v: float) -> str:
    if pd.isna(v):
        return "unknown"
    if v < 0.50:
        return "low"
    if v < 0.75:
        return "medium"
    return "high"


def feature_driver_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.dropna(subset=["release_month"]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    work["feature_norm"] = work["norm_type"].fillna("unknown").astype(str).str.lower()
    work["feature_activation"] = work["activation"].fillna("unknown").astype(str).str.lower()
    work["feature_position"] = work["positional_scheme"].fillna("unknown").astype(str).str.lower()
    work["feature_head_dim"] = work["head_dim"].apply(bucket_head_dim)
    work["feature_gqa"] = work["gqa_ratio"].apply(bucket_gqa)
    work["feature_kv_comp"] = work["kv_compression"].apply(bucket_kv)
    work["feature_context"] = work["context_bucket"].astype(str).replace("nan", "unknown")

    feature_cols = [
        "feature_norm",
        "feature_activation",
        "feature_position",
        "feature_head_dim",
        "feature_gqa",
        "feature_kv_comp",
        "feature_context",
    ]

    monthly_rows = []
    summary_rows = []

    for feature in feature_cols:
        tmp = work[["release_month", feature, "adoption_residual"]].copy()
        tmp[feature] = tmp[feature].fillna("unknown").astype(str)

        grouped = tmp.groupby(["release_month", feature]).size().reset_index(name="count")
        if grouped.empty:
            continue

        local_rows = []
        for month, mdf in grouped.groupby("release_month"):
            total = int(mdf["count"].sum())
            mdf = mdf.sort_values(["count", feature], ascending=[False, True]).reset_index(drop=True)
            top = mdf.iloc[0]
            probs = mdf["count"] / total
            entropy = float(-(probs * np.log(probs + EPS)).sum())
            max_entropy = float(np.log(len(mdf))) if len(mdf) > 1 else 0.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            row = {
                "feature": feature,
                "release_month": month,
                "dominant_value": str(top[feature]),
                "dominance_share": float(top["count"] / total),
                "normalized_entropy": norm_entropy,
                "model_count": total,
            }
            local_rows.append(row)
            monthly_rows.append(row)

        feature_ts = pd.DataFrame(local_rows).sort_values("release_month").reset_index(drop=True)
        idx = np.arange(len(feature_ts), dtype=float)

        if len(feature_ts) >= 2:
            dom_slope = float(np.polyfit(idx, feature_ts["dominance_share"], 1)[0])
            entropy_slope = float(np.polyfit(idx, feature_ts["normalized_entropy"], 1)[0])
            dom_change = float(feature_ts["dominance_share"].iloc[-1] - feature_ts["dominance_share"].iloc[0])
        else:
            dom_slope = np.nan
            entropy_slope = np.nan
            dom_change = np.nan

        latest_dom = feature_ts["dominant_value"].iloc[-1]
        latest_dom_share = float(feature_ts["dominance_share"].iloc[-1])

        premium = np.nan
        premium_n = 0
        premium_df = tmp.dropna(subset=["adoption_residual"]).copy()
        if not premium_df.empty:
            dom_vals = premium_df[premium_df[feature] == latest_dom]["adoption_residual"]
            other_vals = premium_df[premium_df[feature] != latest_dom]["adoption_residual"]
            premium_n = int(len(dom_vals) + len(other_vals))
            if len(dom_vals) >= 2 and len(other_vals) >= 2:
                premium = float(dom_vals.mean() - other_vals.mean())

        summary_rows.append(
            {
                "feature": feature,
                "latest_dominant_value": latest_dom,
                "latest_dominance_share": latest_dom_share,
                "dominance_slope_per_month": dom_slope,
                "dominance_change": dom_change,
                "entropy_slope_per_month": entropy_slope,
                "dominant_value_premium": premium,
                "premium_sample_n": premium_n,
            }
        )

    monthly_df = pd.DataFrame(monthly_rows).sort_values(["feature", "release_month"])
    summary_df = pd.DataFrame(summary_rows).sort_values("dominance_change", ascending=False)
    return monthly_df, summary_df


def build_hidden_hybrid_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def reason(row: pd.Series) -> str:
        reasons: List[str] = []
        if row.get("hard_hybrid_evidence", 0) > 0:
            reasons.append("ssm_plus_attention")
        if row.get("name_ssm_hint", 0) > 0:
            reasons.append("name_hint")
        if row.get("numeric_ssm_evidence", 0) > 0:
            reasons.append("ssm_numeric_keys")
        if row.get("layer_mix_evidence", 0) > 0:
            reasons.append("mixed_layers")
        return "|".join(reasons)

    candidates = df[df["hidden_hybrid_candidate"]].copy()
    if not candidates.empty:
        candidates["candidate_reasons"] = candidates.apply(reason, axis=1)
    else:
        candidates["candidate_reasons"] = pd.Series(dtype=str)

    candidate_cols = [
        "model_id",
        "family",
        "org",
        "release_date",
        "registry_paradigm",
        "inferred_paradigm",
        "hybridization_score",
        "hard_hybrid_evidence",
        "name_ssm_hint",
        "numeric_ssm_evidence",
        "layer_mix_evidence",
        "downloads",
        "param_b",
        "max_context",
        "candidate_reasons",
        "detected_numeric_keys",
    ]
    candidates = candidates[candidate_cols].sort_values(
        ["hybridization_score", "downloads"], ascending=[False, False]
    )

    family = (
        df.groupby("family", dropna=False)
        .agg(
            model_count=("model_id", "count"),
            hidden_hybrid_count=("hidden_hybrid_candidate", "sum"),
            hidden_hybrid_share=("hidden_hybrid_candidate", "mean"),
            mean_hybridization_score=("hybridization_score", "mean"),
            median_downloads=("downloads", "median"),
        )
        .reset_index()
        .sort_values(["hidden_hybrid_count", "mean_hybridization_score"], ascending=[False, False])
    )

    return candidates, family


def build_adoption_tables(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    controls = ["log10_params", "recency_months"]

    metrics = {
        "architecture_quality_index": "Architecture Quality Index",
        "convergence_score": "Convergence Score",
        "hybridization_score": "Hybridization Score",
        "kv_compression": "KV Compression",
        "log_context_efficiency": "Log Context Efficiency",
        "ffn_balance": "FFN Balance",
        "is_long_context_128k_f": "Long Context (>=128k)",
        "has_quant_f": "Has Quantized Variant",
        "has_ssm_signals_f": "Has SSM Signals",
    }

    corr_rows = []
    for col, label in metrics.items():
        raw_df = df[[col, "log10_downloads"]].replace([np.inf, -np.inf], np.nan).dropna()
        raw_corr = np.nan
        if len(raw_df) >= 3 and raw_df[col].std(ddof=0) > 0 and raw_df["log10_downloads"].std(ddof=0) > 0:
            raw_corr = float(raw_df[col].corr(raw_df["log10_downloads"]))

        p_corr, n_partial = partial_corr(df, col, "log10_downloads", controls)
        corr_rows.append(
            {
                "metric": col,
                "metric_label": label,
                "n_raw": len(raw_df),
                "raw_corr_log_downloads": raw_corr,
                "partial_corr_log_downloads": p_corr,
                "n_partial": n_partial,
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values(
        "partial_corr_log_downloads", key=lambda s: s.abs(), ascending=False
    )

    baseline_predictors = ["log10_params", "recency_months"]
    full_predictors = baseline_predictors + [
        "architecture_quality_index",
        "hybridization_score",
        "kv_compression",
        "is_long_context_128k_f",
        "has_quant_f",
    ]

    baseline_coef, baseline_r2, baseline_n = fit_standardized_ols(df, "log10_downloads", baseline_predictors)
    full_coef, full_r2, full_n = fit_standardized_ols(df, "log10_downloads", full_predictors)

    fit_summary = pd.DataFrame(
        [
            {"model": "baseline_size_recency", "r2": baseline_r2, "n": baseline_n},
            {"model": "full_plus_architecture", "r2": full_r2, "n": full_n},
            {
                "model": "incremental_architecture_r2",
                "r2": (full_r2 - baseline_r2) if pd.notna(full_r2) and pd.notna(baseline_r2) else np.nan,
                "n": full_n,
            },
        ]
    )

    premium = (
        df.dropna(subset=["adoption_residual"])
        .groupby("inferred_paradigm", dropna=False)
        .agg(
            model_count=("model_id", "count"),
            mean_adoption_premium=("adoption_residual", "mean"),
            median_adoption_premium=("adoption_residual", "median"),
            median_downloads=("downloads", "median"),
        )
        .reset_index()
        .sort_values("mean_adoption_premium", ascending=False)
    )

    return corr_df, baseline_coef, full_coef, fit_summary, premium


def monthly_evolution(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        df.dropna(subset=["release_month"])
        .groupby("release_month")
        .agg(
            model_count=("model_id", "count"),
            transformer_share=("inferred_paradigm", lambda s: (s == "transformer").mean()),
            recurrent_share=("inferred_paradigm", lambda s: s.isin(RECURSIVE_PARADIGMS).mean()),
            hidden_hybrid_share=("hidden_hybrid_candidate", "mean"),
            median_convergence_score=("convergence_score", "median"),
            median_hybridization_score=("hybridization_score", "median"),
            median_context_k=("context_k", "median"),
            median_kv_compression=("kv_compression", "median"),
            median_arch_quality=("architecture_quality_index", "median"),
            median_log_downloads=("log10_downloads", "median"),
            median_adoption_residual=("adoption_residual", "median"),
        )
        .reset_index()
        .sort_values("release_month")
    )

    delta_metrics = [
        "model_count",
        "transformer_share",
        "recurrent_share",
        "hidden_hybrid_share",
        "median_convergence_score",
        "median_hybridization_score",
        "median_context_k",
        "median_arch_quality",
        "median_adoption_residual",
    ]
    for col in delta_metrics:
        monthly[f"delta_{col}"] = monthly[col].diff()

    events = []
    for col in delta_metrics:
        d = monthly[f"delta_{col}"].dropna()
        if len(d) < 3 or d.std(ddof=0) == 0:
            continue
        z = (d - d.mean()) / d.std(ddof=0)
        flagged = z[abs(z) >= 1.5]
        for idx, z_val in flagged.items():
            events.append(
                {
                    "release_month": monthly.loc[idx, "release_month"],
                    "metric": col,
                    "delta": monthly.loc[idx, f"delta_{col}"],
                    "zscore": float(z_val),
                    "direction": "up" if z_val > 0 else "down",
                }
            )

    changepoints = pd.DataFrame(events).sort_values(["release_month", "metric"]) if events else pd.DataFrame()
    return monthly, changepoints


def save_figures(
    df: pd.DataFrame,
    convergence: pd.DataFrame,
    hidden: pd.DataFrame,
    feature_drivers: pd.DataFrame,
    corr_df: pd.DataFrame,
    monthly: pd.DataFrame,
    out_figs: Path,
) -> None:
    out_figs.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    if not convergence.empty:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(convergence["release_month"], convergence["convergence_index"], marker="o", color="#1f77b4")
        ax1.set_ylabel("Convergence Index", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_xlabel("Release Month")
        ax1.set_title("Cross-Paradigm Convergence Dynamics (Expanding Window)")

        ax2 = ax1.twinx()
        ax2.plot(convergence["release_month"], convergence["between_distance"], marker="s", color="#d62728", label="Between")
        ax2.plot(convergence["release_month"], convergence["within_distance"], marker="^", color="#2ca02c", label="Within")
        ax2.set_ylabel("Feature-Space Distance")
        ax2.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_figs / "01_convergence_dynamics.png", dpi=240)
        plt.close()

    scatter_df = df.dropna(subset=["hybridization_score", "log10_downloads"]).copy()
    if not scatter_df.empty:
        plt.figure(figsize=(11, 7))
        sns.scatterplot(
            data=scatter_df,
            x="hybridization_score",
            y="log10_downloads",
            hue="inferred_paradigm",
            size="param_b",
            sizes=(40, 320),
            alpha=0.85,
        )
        plt.xlabel("Hybridization Score")
        plt.ylabel("log10(Downloads + 1)")
        plt.title("Hidden Hybridization vs Adoption")
        if not hidden.empty:
            ann = hidden.head(8)
            for _, row in ann.iterrows():
                mid = row["model_id"]
                point = scatter_df[scatter_df["model_id"] == mid]
                if not point.empty:
                    x = point["hybridization_score"].iloc[0]
                    y = point["log10_downloads"].iloc[0]
                    plt.annotate(short_name(mid), (x, y), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_figs / "02_hidden_hybridization_map.png", dpi=240)
        plt.close()

    if not feature_drivers.empty:
        plot_df = feature_drivers.dropna(subset=["dominance_change"]).copy()
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=plot_df,
                x="dominance_change",
                y="dominant_value_premium",
                size="latest_dominance_share",
                sizes=(80, 360),
                color="#1f77b4",
                alpha=0.8,
            )
            for _, row in plot_df.iterrows():
                plt.annotate(row["feature"].replace("feature_", ""), (row["dominance_change"], row["dominant_value_premium"]), fontsize=8)
            plt.axhline(0, color="gray", linewidth=1)
            plt.axvline(0, color="gray", linewidth=1)
            plt.xlabel("Dominance Change (Early -> Late)")
            plt.ylabel("Adoption Premium of Latest Dominant Value")
            plt.title("Convergence Drivers: Dominance Shift vs Adoption Premium")
            plt.tight_layout()
            plt.savefig(out_figs / "03_feature_convergence_drivers.png", dpi=240)
            plt.close()

    if not corr_df.empty:
        cdf = corr_df.dropna(subset=["partial_corr_log_downloads"]).copy()
        if not cdf.empty:
            cdf = cdf.sort_values("partial_corr_log_downloads")
            plt.figure(figsize=(10, 6))
            sns.barplot(data=cdf, x="partial_corr_log_downloads", y="metric_label", color="#4c78a8")
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("Partial Correlation with log10(Downloads + 1)\n(controls: size, recency)")
            plt.ylabel("")
            plt.title("Architecture Signals vs Adoption (Partial Correlations)")
            plt.tight_layout()
            plt.savefig(out_figs / "04_adoption_partial_correlations.png", dpi=240)
            plt.close()

    if not monthly.empty:
        heat_cols = [
            "transformer_share",
            "recurrent_share",
            "hidden_hybrid_share",
            "median_convergence_score",
            "median_hybridization_score",
            "median_context_k",
            "median_arch_quality",
            "median_adoption_residual",
        ]
        heat = monthly.set_index("release_month")[heat_cols].copy()
        heat = heat.apply(safe_zscore, axis=0).fillna(0.0)
        plt.figure(figsize=(12, 5))
        sns.heatmap(heat.T, cmap="coolwarm", center=0, cbar_kws={"label": "z-score"})
        plt.title("Monthly Ecosystem Evolution (Normalized)")
        plt.xlabel("Release Month")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_figs / "05_monthly_evolution_heatmap.png", dpi=240)
        plt.close()


def write_key_findings(
    df: pd.DataFrame,
    convergence: pd.DataFrame,
    feature_drivers: pd.DataFrame,
    corr_df: pd.DataFrame,
    fit_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    changepoints: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []
    total = len(df)
    hidden_n = int(df["hidden_hybrid_candidate"].sum())
    lines.append(f"models_total: {total}")
    lines.append(f"hidden_hybrid_candidates: {hidden_n} ({hidden_n / total:.1%})")
    strict_hidden = int(
        (
            df["hidden_hybrid_candidate"]
            & df["registry_paradigm"].isin(["unassigned", "transformer", "reasoning", "diffusion"])
        ).sum()
    )
    lines.append(f"strict_hidden_hybrids: {strict_hidden}")

    if not convergence.empty:
        start = float(convergence["convergence_index"].iloc[0])
        end = float(convergence["convergence_index"].iloc[-1])
        lines.append(f"convergence_index_start: {start:.4f}")
        lines.append(f"convergence_index_end: {end:.4f}")
        lines.append(f"convergence_index_change: {end - start:+.4f}")

    if not feature_drivers.empty:
        top = feature_drivers.dropna(subset=["dominance_change"]).head(3)
        if not top.empty:
            summary = "; ".join(
                f"{r.feature.replace('feature_', '')}:{r.dominance_change:+.3f},premium={r.dominant_value_premium:+.3f}"
                for _, r in top.iterrows()
            )
            lines.append(f"top_convergence_drivers: {summary}")

    if not corr_df.empty:
        best = corr_df.dropna(subset=["partial_corr_log_downloads"]).head(1)
        if not best.empty:
            r = best.iloc[0]
            lines.append(
                f"strongest_partial_corr_metric: {r['metric']} ({r['partial_corr_log_downloads']:+.3f}, n={int(r['n_partial'])})"
            )

    if not fit_summary.empty and fit_summary["model"].eq("incremental_architecture_r2").any():
        inc = fit_summary[fit_summary["model"] == "incremental_architecture_r2"]["r2"].iloc[0]
        lines.append(f"incremental_architecture_r2: {inc:.4f}")

    if not monthly.empty:
        peak = monthly.loc[monthly["hidden_hybrid_share"].idxmax()]
        lines.append(f"peak_hidden_hybrid_month: {peak['release_month'].date()} ({peak['hidden_hybrid_share']:.1%})")

    if not changepoints.empty:
        lines.append(f"changepoints_detected: {len(changepoints)}")
        top_cp = changepoints.reindex(changepoints["zscore"].abs().sort_values(ascending=False).index).head(3)
        cp_text = "; ".join(
            f"{row['release_month'].date()}:{row['metric']}({row['direction']},{row['zscore']:+.2f})"
            for _, row in top_cp.iterrows()
        )
        lines.append(f"top_changepoints: {cp_text}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep ecosystem dynamics analysis for cross-paradigm convergence.")
    parser.add_argument("--input", default="metadata_enriched.json", help="Path to metadata_enriched.json")
    parser.add_argument("--outdir", default="analysis/ecosystem_deep", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    out_tables = outdir / "tables"
    out_figs = outdir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)

    meta, df = load_dataframe(input_path)
    df = add_research_features(df)

    df["is_long_context_128k_f"] = df["is_long_context_128k"].fillna(False).astype(float)
    df["has_quant_f"] = df["has_quant"].fillna(False).astype(float)
    df["has_ssm_signals_f"] = df["has_ssm_signals"].fillna(False).astype(float)

    df, baseline_coef_unstd = add_adoption_residual(df)

    convergence = compute_convergence_dynamics(df)
    feature_monthly, feature_drivers = feature_driver_analysis(df)
    hidden, hidden_family = build_hidden_hybrid_tables(df)
    corr_df, baseline_coef, full_coef, fit_summary, premium = build_adoption_tables(df)
    monthly, changepoints = monthly_evolution(df)

    df.to_csv(out_tables / "model_research_metrics.csv", index=False)
    baseline_coef_unstd.to_csv(out_tables / "adoption_baseline_unstandardized.csv", index=False)
    convergence.to_csv(out_tables / "convergence_dynamics_expanding.csv", index=False)
    feature_monthly.to_csv(out_tables / "feature_dominance_monthly.csv", index=False)
    feature_drivers.to_csv(out_tables / "feature_convergence_drivers.csv", index=False)
    hidden.to_csv(out_tables / "hidden_hybrid_candidates.csv", index=False)
    hidden_family.to_csv(out_tables / "hidden_hybrid_family_stats.csv", index=False)
    corr_df.to_csv(out_tables / "adoption_quality_correlations.csv", index=False)
    baseline_coef.to_csv(out_tables / "adoption_baseline_standardized.csv", index=False)
    full_coef.to_csv(out_tables / "adoption_full_standardized.csv", index=False)
    fit_summary.to_csv(out_tables / "adoption_model_fit.csv", index=False)
    premium.to_csv(out_tables / "paradigm_adoption_premium.csv", index=False)
    monthly.to_csv(out_tables / "monthly_evolution_metrics.csv", index=False)
    changepoints.to_csv(out_tables / "changepoints.csv", index=False)

    save_figures(df, convergence, hidden, feature_drivers, corr_df, monthly, out_figs)
    write_key_findings(
        df=df,
        convergence=convergence,
        feature_drivers=feature_drivers,
        corr_df=corr_df,
        fit_summary=fit_summary,
        monthly=monthly,
        changepoints=changepoints,
        out_path=outdir / "key_findings.txt",
    )

    print(f"Analyzed {len(df)} models")
    print(f"Output tables: {out_tables}")
    print(f"Output figures: {out_figs}")
    print(f"Key findings: {outdir / 'key_findings.txt'}")
    if isinstance(meta, dict):
        print(f"source_file={meta.get('source_file')}, generated_at_utc={meta.get('generated_at_utc')}")


if __name__ == "__main__":
    main()
