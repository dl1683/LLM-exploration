#!/usr/bin/env python3
"""
measured_compatibility_tax.py

Replaces heuristic-based compatibility tax with OBSERVED framework support data.
Uses actual vLLM, llama.cpp, TGI, GPTQ, AWQ, BitsAndBytes support per model.

Key innovation: the compatibility tax is no longer derived from paradigm labels
(which would be circular), but from actual toolchain support matrices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecosystem_analysis import add_derived_metrics, flatten_model, load_payload, short_name
from ecosystem_deep_dynamics import add_research_features, add_adoption_residual, safe_zscore

EPS = 1e-12

# ── Observed framework support matrix ──────────────────────────────────────
# Source: vLLM docs, llama.cpp llama-arch.h, TGI docs, GPTQModel, AutoAWQ
# Y=1.0, P=0.5, N=0.0
# Columns: vllm, llamacpp, tgi, gptq, awq, bnb

ARCH_SUPPORT: Dict[str, Dict[str, float]] = {
    # Pure transformers - full ecosystem
    "llama":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 1.0, "bnb": 1.0},
    "qwen2":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 1.0, "bnb": 1.0},
    "qwen3":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 1.0, "awq": 1.0, "bnb": 1.0},
    "gemma":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 1.0, "bnb": 1.0},
    "gemma3":       {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 0.0, "bnb": 1.0},
    "phi3":         {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 1.0, "bnb": 1.0},
    "granite":      {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 0.0, "bnb": 1.0},
    "olmo2":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 1.0, "awq": 0.0, "bnb": 1.0},
    "smollm3":      {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 1.0, "awq": 0.0, "bnb": 1.0},
    "deepseek_r1":  {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 1.0, "awq": 1.0, "bnb": 1.0},
    # Pure SSM
    "mamba":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 1.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    "falcon_mamba": {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    # Hybrid SSM+Attention
    "falcon_h1":    {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 1.0, "awq": 0.0, "bnb": 0.5},
    "granite_hybrid": {"vllm": 0.5, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    "nemotron_h":   {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 1.0, "awq": 0.0, "bnb": 0.5},
    "zamba2":       {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    "jamba":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    "bamba":        {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    # Liquid
    "lfm2":         {"vllm": 1.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    # RWKV
    "rwkv7":        {"vllm": 0.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    "arwkv7":       {"vllm": 0.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    # xLSTM
    "xlstm":        {"vllm": 0.0, "llamacpp": 0.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    # Diffusion LLM
    "llada":        {"vllm": 0.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
    "llada2":       {"vllm": 0.0, "llamacpp": 1.0, "tgi": 0.0, "gptq": 0.0, "awq": 0.0, "bnb": 0.5},
}

# Map each of our 50 model_ids to their architecture key
MODEL_TO_ARCH: Dict[str, str] = {
    # Tier 1
    "HuggingFaceTB/SmolLM3-3B": "smollm3",
    "LiquidAI/LFM2-1.2B-Exp": "lfm2",
    "LiquidAI/LFM2-2.6B-Exp": "lfm2",
    "LiquidAI/LFM2-350M-Exp": "lfm2",
    "Qwen/Qwen3-0.6B": "qwen3",
    "Qwen/Qwen3-1.7B": "qwen3",
    "RWKV/RWKV7-Goose-World3-1.5B-HF": "rwkv7",
    "Zyphra/Zamba2-1.2B": "zamba2",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek_r1",
    "google/gemma-3-1b-it": "gemma3",
    "ibm-granite/granite-4.0-1b": "granite_hybrid",
    "ibm-granite/granite-4.0-350m": "granite_hybrid",
    "ibm-granite/granite-4.0-h-1b": "granite_hybrid",
    "ibm-granite/granite-4.0-h-350m": "granite_hybrid",
    "state-spaces/mamba-1.4b-hf": "mamba",
    "state-spaces/mamba-370m-hf": "mamba",
    "state-spaces/mamba-790m-hf": "mamba",
    "tiiuae/Falcon-H1-0.5B-Instruct": "falcon_h1",
    "tiiuae/Falcon-H1-1.5B-Instruct": "falcon_h1",
    # Tier 2
    "ML-GSAI/LLaDA-8B-Base": "llada",
    "NX-AI/xLSTM-7b": "xlstm",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "RWKV-Red-Team/ARWKV-R1-7B": "arwkv7",
    "Zyphra/Zamba2-2.7B": "zamba2",
    "Zyphra/Zamba2-7B": "zamba2",
    "allenai/Olmo-3-7B": "olmo2",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek_r1",
    "google/gemma-3-4b-it": "gemma3",
    "ibm-granite/granite-4.0-h-tiny": "granite_hybrid",
    "ibm-granite/granite-4.0-micro": "granite_hybrid",
    "ibm-granite/granite-4.0-tiny-preview": "granite_hybrid",
    "microsoft/Phi-4-mini-reasoning": "phi3",
    "nvidia/Nemotron-H-4B-Instruct-128K": "nemotron_h",
    "tiiuae/Falcon-H1-3B-Instruct": "falcon_h1",
    "tiiuae/Falcon-H1-7B-Instruct": "falcon_h1",
    "tiiuae/falcon-mamba-7b": "falcon_mamba",
    # Tier 3
    "Qwen/Qwen3-14B": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
    "allenai/Olmo-3-1125-7B": "olmo2",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek_r1",
    "google/gemma-3-12b-it": "gemma3",
    "google/gemma-3-27b-it": "gemma3",
    "ibm-granite/granite-4.0-h-small": "granite_hybrid",
    "meta-llama/Llama-3.2-3B-Instruct": "llama",
    "microsoft/Phi-4-mini-instruct": "phi3",
    "microsoft/Phi-4-multimodal-instruct": "phi3",
    "tiiuae/Falcon-H1-34B-Instruct": "falcon_h1",
    # Unassigned
    "ML-GSAI/LLaDA2.0-8B-Base": "llada2",
}

# Framework weights for computing composite score
# Reflects real-world usage importance
FRAMEWORK_WEIGHTS: Dict[str, float] = {
    "vllm": 0.25,       # Most important GPU inference server
    "llamacpp": 0.20,   # Most important local/CPU inference
    "tgi": 0.10,        # HuggingFace native, declining
    "gptq": 0.20,       # Key GPU quantization
    "awq": 0.10,        # Second GPU quantization (deprecated)
    "bnb": 0.15,        # Easy 4/8-bit via HuggingFace
}


def compute_measured_support(model_id: str) -> Dict[str, Any]:
    """Look up actual framework support for a model."""
    arch_key = MODEL_TO_ARCH.get(model_id)
    if arch_key is None:
        return {
            "arch_key": "unknown",
            "vllm": np.nan, "llamacpp": np.nan, "tgi": np.nan,
            "gptq": np.nan, "awq": np.nan, "bnb": np.nan,
            "support_score": np.nan, "measured_tax": np.nan,
            "frameworks_full": 0, "frameworks_partial": 0, "frameworks_none": 0,
        }

    support = ARCH_SUPPORT.get(arch_key, {})
    if not support:
        return {
            "arch_key": arch_key,
            "vllm": np.nan, "llamacpp": np.nan, "tgi": np.nan,
            "gptq": np.nan, "awq": np.nan, "bnb": np.nan,
            "support_score": np.nan, "measured_tax": np.nan,
            "frameworks_full": 0, "frameworks_partial": 0, "frameworks_none": 0,
        }

    weighted_sum = sum(
        support.get(fw, 0.0) * FRAMEWORK_WEIGHTS[fw]
        for fw in FRAMEWORK_WEIGHTS
    )
    total_weight = sum(FRAMEWORK_WEIGHTS.values())
    support_score = weighted_sum / total_weight

    full = sum(1 for v in support.values() if v >= 1.0)
    partial = sum(1 for v in support.values() if 0 < v < 1.0)
    none = sum(1 for v in support.values() if v <= 0.0)

    return {
        "arch_key": arch_key,
        **{fw: support.get(fw, 0.0) for fw in FRAMEWORK_WEIGHTS},
        "support_score": support_score,
        "measured_tax": 1.0 - support_score,
        "frameworks_full": full,
        "frameworks_partial": partial,
        "frameworks_none": none,
    }


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
    x_resid = residualize(work[x_col].to_numpy(float), work[controls].to_numpy(float))
    y_resid = residualize(work[y_col].to_numpy(float), work[controls].to_numpy(float))
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan, n
    return float(np.corrcoef(x_resid, y_resid)[0, 1]), n


def ols_coef_vector(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    x_mat = np.column_stack([np.ones(len(y)), xv])
    beta, *_ = np.linalg.lstsq(x_mat, np.asarray(y, float), rcond=None)
    return beta


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: Sequence[str], standardize: bool = False) -> Tuple[pd.DataFrame, float, int]:
    cols = [y_col] + list(x_cols)
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(work)
    if n < max(10, len(x_cols) + 3):
        return pd.DataFrame(), np.nan, n
    if standardize:
        for col in cols:
            work[col] = safe_zscore(work[col])
    x = work[list(x_cols)].to_numpy(float)
    y = work[y_col].to_numpy(float)
    x_mat = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    y_hat = x_mat @ beta
    resid = y - y_hat
    dof = max(n - x_mat.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    t_vals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - (ss_res / (ss_tot + EPS))
    coef = pd.DataFrame({
        "term": ["intercept"] + list(x_cols),
        "coef": beta, "std_err": se, "t_value": t_vals,
        "n": n, "r2": r2, "standardized": standardize,
    })
    return coef, r2, n


def staged_ols(df: pd.DataFrame, y_col: str = "adoption_residual") -> Tuple[pd.DataFrame, pd.DataFrame]:
    stages = [
        ("s1_hybrid_only", ["hybridization_score"]),
        ("s2_plus_measured_tax", ["hybridization_score", "measured_tax"]),
        ("s3_plus_efficiency", [
            "hybridization_score", "measured_tax",
            "is_long_context_128k_f", "has_quant_f",
        ]),
        ("s4_plus_quality", [
            "hybridization_score", "measured_tax",
            "is_long_context_128k_f", "has_quant_f",
            "architecture_quality_index", "convergence_score",
        ]),
    ]

    summary_rows = []
    coef_frames = []

    for stage_name, x_cols in stages:
        coef_u, r2_u, n_u = fit_ols(df, y_col, x_cols, standardize=False)
        coef_s, _, _ = fit_ols(df, y_col, x_cols, standardize=True)

        if not coef_u.empty:
            tmp = coef_u.copy()
            tmp["stage"] = stage_name
            tmp["scale"] = "unstandardized"
            coef_frames.append(tmp)
        if not coef_s.empty:
            tmp = coef_s.copy()
            tmp["stage"] = stage_name
            tmp["scale"] = "standardized"
            coef_frames.append(tmp)

        def _get(df_c, term, col):
            if df_c.empty or term not in set(df_c["term"]):
                return np.nan
            return float(df_c[df_c["term"] == term].iloc[0][col])

        summary_rows.append({
            "stage": stage_name,
            "predictors": "|".join(x_cols),
            "n": n_u,
            "r2": r2_u,
            "hybrid_coef": _get(coef_u, "hybridization_score", "coef"),
            "hybrid_se": _get(coef_u, "hybridization_score", "std_err"),
            "hybrid_t": _get(coef_u, "hybridization_score", "t_value"),
            "hybrid_beta_std": _get(coef_s, "hybridization_score", "coef"),
            "measured_tax_coef": _get(coef_u, "measured_tax", "coef"),
            "measured_tax_t": _get(coef_u, "measured_tax", "t_value"),
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        base = summary_df["hybrid_coef"].iloc[0]
        summary_df["hybrid_coef_shift"] = summary_df["hybrid_coef"] - base

    coef_df = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    return summary_df, coef_df


def bootstrap_mediation(
    df: pd.DataFrame,
    x_col: str = "hybridization_score",
    m_col: str = "measured_tax",
    y_col: str = "adoption_residual",
    n_boot: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    cols = [x_col, m_col, y_col]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(work)
    if n < 12:
        return pd.DataFrame([{"n": n, "indirect": np.nan, "ci_low": np.nan,
                              "ci_high": np.nan, "p": np.nan}])

    x = work[x_col].to_numpy(float)
    m = work[m_col].to_numpy(float)
    y = work[y_col].to_numpy(float)

    # Point estimates
    a = float(ols_coef_vector(m, x)[1])
    beta_ym = ols_coef_vector(y, np.column_stack([x, m]))
    c_prime = float(beta_ym[1])
    b = float(beta_ym[2])
    c_total = float(ols_coef_vector(y, x)[1])
    indirect = a * b

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_indirect = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, mb, yb = x[idx], m[idx], y[idx]
        ai = float(ols_coef_vector(mb, xb)[1])
        bi_full = ols_coef_vector(yb, np.column_stack([xb, mb]))
        bi = float(bi_full[2])
        boot_indirect[i] = ai * bi

    ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    p = 2.0 * min(float((boot_indirect <= 0).mean()), float((boot_indirect >= 0).mean()))

    return pd.DataFrame([{
        "n": n,
        "a_path": a,
        "b_path": b,
        "c_total": c_total,
        "c_prime_direct": c_prime,
        "indirect_effect": indirect,
        "ci_low_95": float(ci_low),
        "ci_high_95": float(ci_high),
        "proportion_mediated": indirect / c_total if abs(c_total) > EPS else np.nan,
        "bootstrap_p": float(np.clip(p, 0, 1)),
        "bootstrap_n": n_boot,
    }])


def permutation_test(
    df: pd.DataFrame,
    x_col: str = "hybridization_score",
    y_col: str = "adoption_residual",
    controls: Optional[List[str]] = None,
    n_perm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    controls = controls or []
    cols = [y_col, x_col] + controls
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(work)
    if n < 10:
        return pd.DataFrame([{"n": n, "observed_beta": np.nan, "p": np.nan}])

    y = work[y_col].to_numpy(float)
    x = work[x_col].to_numpy(float)
    c = work[controls].to_numpy(float) if controls else np.empty((n, 0))

    def _beta(y_, x_, c_):
        design = np.column_stack([x_, c_]) if c_.size else x_.reshape(-1, 1)
        return float(ols_coef_vector(y_, design)[1])

    obs = _beta(y, x, c)
    rng = np.random.default_rng(seed)
    perm_betas = np.array([_beta(y, rng.permutation(x), c) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm_betas) >= abs(obs)) + 1) / (n_perm + 1)

    return pd.DataFrame([{
        "n": n,
        "observed_beta": obs,
        "perm_mean": float(perm_betas.mean()),
        "perm_std": float(perm_betas.std()),
        "z_vs_null": float((obs - perm_betas.mean()) / (perm_betas.std() + EPS)),
        "p_two_sided": float(p),
        "n_permutations": n_perm,
    }])


def save_figures(df: pd.DataFrame, staged: pd.DataFrame, mediation: pd.DataFrame, out_figs: Path) -> None:
    out_figs.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    # 1. Measured tax by paradigm
    tax_by_paradigm = (
        df.dropna(subset=["measured_tax", "inferred_paradigm"])
        .groupby("inferred_paradigm")["measured_tax"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    if not tax_by_paradigm.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=tax_by_paradigm, x="measured_tax", y="inferred_paradigm", color="#d62728")
        plt.xlabel("Measured Compatibility Tax (0=full support, 1=no support)")
        plt.ylabel("")
        plt.title("Observed Toolchain Compatibility Tax by Paradigm")
        plt.tight_layout()
        plt.savefig(out_figs / "01_measured_tax_by_paradigm.png", dpi=240)
        plt.close()

    # 2. Staged OLS coefficient attenuation
    if not staged.empty:
        plt.figure(figsize=(10, 6))
        x_pos = range(len(staged))
        plt.bar(x_pos, staged["hybrid_coef"], color="#1f77b4", alpha=0.8)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(list(x_pos), staged["stage"], rotation=30, ha="right")
        plt.ylabel("Hybridization Coefficient (unstandardized)")
        plt.title("Hybrid Penalty Attenuation as Controls Enter\n(Sign flip = tax explains penalty)")
        for i, row in staged.iterrows():
            plt.annotate(f"R²={row['r2']:.2f}", (i, row["hybrid_coef"]),
                         textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_figs / "02_staged_ols_attenuation.png", dpi=240)
        plt.close()

    # 3. Measured tax vs adoption scatter
    scatter = df.dropna(subset=["measured_tax", "log10_downloads"]).copy()
    if not scatter.empty:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=scatter, x="measured_tax", y="log10_downloads",
            hue="inferred_paradigm", size="param_b", sizes=(40, 320), alpha=0.85,
        )
        plt.xlabel("Measured Compatibility Tax")
        plt.ylabel("log10(Downloads + 1)")
        plt.title("Toolchain Tax vs Adoption (Observed)")
        for _, row in scatter.nlargest(5, "downloads").iterrows():
            plt.annotate(short_name(row["model_id"]),
                         (row["measured_tax"], row["log10_downloads"]), fontsize=8)
        plt.tight_layout()
        plt.savefig(out_figs / "03_measured_tax_vs_adoption.png", dpi=240)
        plt.close()

    # 4. Framework support heatmap
    fw_cols = ["vllm", "llamacpp", "tgi", "gptq", "awq", "bnb"]
    heat_data = df.dropna(subset=["inferred_paradigm"]).groupby("inferred_paradigm")[fw_cols].mean()
    if not heat_data.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                    cbar_kws={"label": "Support Level (0=None, 1=Full)"})
        plt.title("Framework Support Matrix by Paradigm (Observed)")
        plt.tight_layout()
        plt.savefig(out_figs / "04_framework_support_heatmap.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measured compatibility tax analysis.")
    parser.add_argument("--input", default="metadata_enriched.json")
    parser.add_argument("--outdir", default="analysis/measured_tax")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
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
    df, _ = add_adoption_residual(df)

    # Compute measured support
    support_records = [compute_measured_support(mid) for mid in df["model_id"]]
    support_df = pd.DataFrame(support_records, index=df.index)
    for col in support_df.columns:
        df[col] = support_df[col]

    # Summary by paradigm
    fw_cols = ["vllm", "llamacpp", "tgi", "gptq", "awq", "bnb"]
    paradigm_tax = (
        df.dropna(subset=["measured_tax"])
        .groupby("inferred_paradigm")
        .agg(
            model_count=("model_id", "count"),
            mean_measured_tax=("measured_tax", "mean"),
            mean_support_score=("support_score", "mean"),
            mean_adoption_residual=("adoption_residual", "mean"),
            mean_hybridization=("hybridization_score", "mean"),
            **{f"mean_{fw}": (fw, "mean") for fw in fw_cols},
        )
        .reset_index()
        .sort_values("mean_measured_tax", ascending=False)
    )

    # Partial correlations
    controls = ["log10_params", "recency_months"]
    corr_rows = []
    for metric in ["measured_tax", "hybridization_score", "support_score",
                    "convergence_score", "architecture_quality_index",
                    "is_long_context_128k_f", "has_quant_f"]:
        raw_df = df[[metric, "log10_downloads"]].replace([np.inf, -np.inf], np.nan).dropna()
        raw_r = float(raw_df[metric].corr(raw_df["log10_downloads"])) if len(raw_df) >= 3 else np.nan
        p_r, n_p = partial_corr(df, metric, "log10_downloads", controls)
        corr_rows.append({
            "metric": metric, "n_raw": len(raw_df), "raw_corr": raw_r,
            "partial_corr": p_r, "n_partial": n_p,
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("partial_corr", key=lambda s: s.abs(), ascending=False)

    # Staged OLS
    staged_summary, staged_coef = staged_ols(df)

    # Mediation: hybridization -> measured_tax -> adoption
    mediation = bootstrap_mediation(df, n_boot=args.bootstrap, seed=args.seed)

    # Permutation test
    perm = permutation_test(df, controls=["measured_tax"], n_perm=args.permutations, seed=args.seed)

    # Save tables
    model_cols = [
        "model_id", "inferred_paradigm", "arch_key",
        "vllm", "llamacpp", "tgi", "gptq", "awq", "bnb",
        "support_score", "measured_tax",
        "frameworks_full", "frameworks_partial", "frameworks_none",
        "hybridization_score", "adoption_residual", "log10_downloads",
        "param_b", "downloads",
    ]
    model_cols = [c for c in model_cols if c in df.columns]
    df[model_cols].to_csv(out_tables / "model_measured_tax.csv", index=False)
    paradigm_tax.to_csv(out_tables / "paradigm_measured_tax.csv", index=False)
    corr_df.to_csv(out_tables / "partial_correlations.csv", index=False)
    staged_summary.to_csv(out_tables / "staged_ols_summary.csv", index=False)
    staged_coef.to_csv(out_tables / "staged_ols_coefficients.csv", index=False)
    mediation.to_csv(out_tables / "mediation_summary.csv", index=False)
    perm.to_csv(out_tables / "permutation_test.csv", index=False)

    # Figures
    save_figures(df, staged_summary, mediation, out_figs)

    # Key findings
    lines = []
    lines.append(f"models_total: {len(df)}")
    lines.append(f"models_with_measured_tax: {df['measured_tax'].notna().sum()}")

    if not paradigm_tax.empty:
        for _, row in paradigm_tax.iterrows():
            lines.append(f"tax_{row['inferred_paradigm']}: {row['mean_measured_tax']:.4f} (n={int(row['model_count'])})")

    if not corr_df.empty:
        top = corr_df.iloc[0]
        lines.append(f"strongest_partial_corr: {top['metric']} ({top['partial_corr']:+.4f}, n={int(top['n_partial'])})")

    if not staged_summary.empty:
        s1 = staged_summary.iloc[0]
        s2 = staged_summary.iloc[1] if len(staged_summary) > 1 else s1
        lines.append(f"stage1_hybrid_coef: {s1['hybrid_coef']:+.4f} (R²={s1['r2']:.4f})")
        lines.append(f"stage2_hybrid_coef: {s2['hybrid_coef']:+.4f} (R²={s2['r2']:.4f})")
        lines.append(f"SIGN_FLIP: {'YES' if s1['hybrid_coef'] * s2['hybrid_coef'] < 0 else 'NO'}")
        last = staged_summary.iloc[-1]
        lines.append(f"final_stage_r2: {last['r2']:.4f}")

    if not mediation.empty:
        m = mediation.iloc[0]
        lines.append(f"mediation_indirect: {m['indirect_effect']:+.4f}")
        lines.append(f"mediation_ci95: [{m['ci_low_95']:+.4f}, {m['ci_high_95']:+.4f}]")
        lines.append(f"mediation_p: {m['bootstrap_p']:.4f}")
        lines.append(f"proportion_mediated: {m['proportion_mediated']:.4f}")

    if not perm.empty:
        p = perm.iloc[0]
        lines.append(f"permutation_p: {p['p_two_sided']:.4f} (n={int(p['n'])})")

    findings_path = outdir / "key_findings.txt"
    findings_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Analyzed {len(df)} models")
    print(f"Tables: {out_tables}")
    print(f"Figures: {out_figs}")
    print(f"Key findings: {findings_path}")


if __name__ == "__main__":
    main()
