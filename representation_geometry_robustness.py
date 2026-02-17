#!/usr/bin/env python3
"""
representation_geometry_robustness.py — Robustness Controls & Deeper Analysis

Stage 1.5: Validates the intrinsic dimensionality gap between architectures
with proper controls as recommended by Codex review:
  1. PR/d_model normalization
  2. Bootstrap confidence intervals (resample prompts)
  3. Core-layer-only analysis (excluding embedding + final layers)
  4. Variance spectra / eigenvalue distributions per paradigm
  5. Effective rank (Shannon entropy of singular value distribution)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Model registry (must match representation_geometry.py) ─────────────────
EXPERIMENT_MODELS: List[Tuple[str, str, str]] = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "transformer"),
    ("google/gemma-3-1b-it", "Gemma3-1B", "transformer"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "transformer"),
    ("google/gemma-2-2b-it", "Gemma2-2B", "transformer"),
    ("state-spaces/mamba-790m-hf", "Mamba-790M", "ssm"),
    ("state-spaces/mamba-1.4b-hf", "Mamba-1.4B", "ssm"),
    ("state-spaces/mamba-2.8b-hf", "Mamba-2.8B", "ssm"),
    ("tiiuae/Falcon-H1-0.5B-Instruct", "FalconH1-0.5B", "hybrid"),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "FalconH1-1.5B", "hybrid"),
    ("Zyphra/Zamba2-1.2B", "Zamba2-1.2B", "hybrid"),
    ("nvidia/Hymba-1.5B-Base", "Hymba-1.5B", "hybrid"),
    ("RWKV/RWKV7-Goose-World3-1.5B-HF", "RWKV7-1.5B", "rwkv"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning"),
]


# ── Metric functions ───────────────────────────────────────────────────────
def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum λ)^2 / sum(λ^2) where λ are squared singular values."""
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return 0.0
    return float((total ** 2) / (s2 ** 2).sum())


def effective_rank(X: np.ndarray) -> float:
    """Shannon entropy-based effective rank (Roy & Vetterli 2007)."""
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return 0.0
    p = s2 / total
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def anisotropy(X: np.ndarray) -> float:
    """Mean cosine similarity between all pairs."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_norm = X / norms
    cos_sim = X_norm @ X_norm.T
    n = cos_sim.shape[0]
    if n < 2:
        return 0.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(cos_sim[mask].mean())


def variance_spectrum(X: np.ndarray) -> np.ndarray:
    """Normalized singular value spectrum (variance explained per component)."""
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return s2
    return s2 / total


def bootstrap_metric(X: np.ndarray, metric_fn, n_boot: int = 200,
                     seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI for a metric by resampling samples (prompts).
    Returns (mean, ci_low, ci_high) at 95% level."""
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        values.append(metric_fn(X[idx]))
    values = np.array(values)
    return float(values.mean()), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ── Main analysis ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Robustness controls for representation geometry")
    parser.add_argument("--n-boot", type=int, default=200, help="Bootstrap resamples")
    parser.add_argument("--core-skip", type=int, default=2,
                        help="Skip first/last N layers for core analysis")
    args = parser.parse_args()

    cache_dir = Path("analysis/representation_geometry/activation_cache")
    out_dir = Path("analysis/representation_geometry_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figs = out_dir / "figures"
    out_figs.mkdir(exist_ok=True)
    out_tables = out_dir / "tables"
    out_tables.mkdir(exist_ok=True)

    # Load all cached activations
    models = {}
    for model_id, short_name, paradigm in EXPERIMENT_MODELS:
        cache_path = cache_dir / f"{short_name}.npz"
        if cache_path.exists():
            data = np.load(cache_path)["hidden_states"]  # (n_layers, n_prompts, hidden_dim)
            models[short_name] = {"data": data, "paradigm": paradigm, "model_id": model_id}
            print(f"  Loaded {short_name}: {data.shape}")

    print(f"\nLoaded {len(models)} models from cache\n")

    # ── 1. Core-layer metrics with PR/d_model normalization ────────────────
    print("=" * 60)
    print("Analysis 1: Core-layer metrics with PR/d_model normalization")
    print("=" * 60)

    skip = args.core_skip
    rows = []

    for name, info in models.items():
        data = info["data"]
        n_layers, n_prompts, d_model = data.shape
        core = data[skip:-1] if n_layers > skip + 1 else data  # skip first N and last 1

        # Flatten core layers: each layer's activations across prompts
        for li, layer_data in enumerate(core):
            pr = participation_ratio(layer_data)
            er = effective_rank(layer_data)
            aniso = anisotropy(layer_data)
            rows.append({
                "model": name,
                "paradigm": info["paradigm"],
                "layer": li + skip,
                "layer_frac": (li + skip) / (n_layers - 1),  # normalized position
                "d_model": d_model,
                "PR": pr,
                "PR_over_d": pr / d_model,
                "effective_rank": er,
                "ER_over_d": er / d_model,
                "anisotropy": aniso,
            })

    core_df = pd.DataFrame(rows)
    core_df.to_csv(out_tables / "core_layer_metrics.csv", index=False)

    # Summary by paradigm
    paradigm_summary = core_df.groupby("paradigm").agg(
        PR_mean=("PR", "mean"),
        PR_std=("PR", "std"),
        PR_over_d_mean=("PR_over_d", "mean"),
        PR_over_d_std=("PR_over_d", "std"),
        ER_mean=("effective_rank", "mean"),
        ER_std=("effective_rank", "std"),
        ER_over_d_mean=("ER_over_d", "mean"),
        ER_over_d_std=("ER_over_d", "std"),
        aniso_mean=("anisotropy", "mean"),
        aniso_std=("anisotropy", "std"),
        n_layers=("layer", "count"),
    ).reset_index()
    paradigm_summary.to_csv(out_tables / "paradigm_core_summary.csv", index=False)
    print(paradigm_summary.to_string(index=False))
    print()

    # ── 2. Bootstrap CIs for model-level metrics ──────────────────────────
    print("=" * 60)
    print("Analysis 2: Bootstrap confidence intervals")
    print("=" * 60)

    boot_rows = []
    for name, info in models.items():
        data = info["data"]
        n_layers, n_prompts, d_model = data.shape
        core = data[skip:-1] if n_layers > skip + 1 else data

        # Average across core layers to get model-level representation
        # (n_prompts, d_model) averaged across layers
        avg_repr = core.mean(axis=0)  # (n_prompts, d_model)

        pr_mean, pr_lo, pr_hi = bootstrap_metric(avg_repr, participation_ratio, args.n_boot)
        er_mean, er_lo, er_hi = bootstrap_metric(avg_repr, effective_rank, args.n_boot)
        aniso_mean, aniso_lo, aniso_hi = bootstrap_metric(avg_repr, anisotropy, args.n_boot)

        boot_rows.append({
            "model": name,
            "paradigm": info["paradigm"],
            "d_model": d_model,
            "PR_mean": pr_mean,
            "PR_ci_low": pr_lo,
            "PR_ci_high": pr_hi,
            "PR_over_d": pr_mean / d_model,
            "ER_mean": er_mean,
            "ER_ci_low": er_lo,
            "ER_ci_high": er_hi,
            "aniso_mean": aniso_mean,
            "aniso_ci_low": aniso_lo,
            "aniso_ci_high": aniso_hi,
        })
        ci_width = pr_hi - pr_lo
        print(f"  {name}: PR={pr_mean:.2f} [{pr_lo:.2f}, {pr_hi:.2f}] "
              f"(CI width={ci_width:.2f}), ER={er_mean:.2f}, Aniso={aniso_mean:.3f}")

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_tables / "bootstrap_model_metrics.csv", index=False)
    print()

    # ── 3. Variance spectra per paradigm ──────────────────────────────────
    print("=" * 60)
    print("Analysis 3: Variance spectra (eigenvalue distributions)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: log-scale spectra per model
    paradigm_colors = {
        "transformer": "#2196F3",
        "ssm": "#FF5722",
        "hybrid": "#4CAF50",
        "reasoning": "#9C27B0",
        "rwkv": "#FF9800",
    }
    paradigm_styles = {
        "transformer": "-",
        "ssm": "--",
        "hybrid": "-.",
        "reasoning": ":",
        "rwkv": "-",
    }

    spectra_by_paradigm = {}

    for name, info in models.items():
        data = info["data"]
        n_layers = data.shape[0]
        core = data[skip:-1] if n_layers > skip + 1 else data
        paradigm = info["paradigm"]

        # Average spectrum across core layers
        specs = []
        for layer_data in core:
            spec = variance_spectrum(layer_data)
            specs.append(spec[:min(16, len(spec))])  # Top 16 components

        # Pad to same length and average
        max_len = max(len(s) for s in specs)
        padded = np.zeros((len(specs), max_len))
        for i, s in enumerate(specs):
            padded[i, :len(s)] = s
        avg_spec = padded.mean(axis=0)

        if paradigm not in spectra_by_paradigm:
            spectra_by_paradigm[paradigm] = []
        spectra_by_paradigm[paradigm].append(avg_spec)

        axes[0].semilogy(
            range(1, len(avg_spec) + 1), avg_spec,
            color=paradigm_colors.get(paradigm, "gray"),
            linestyle=paradigm_styles.get(paradigm, "-"),
            alpha=0.6, linewidth=1.5, label=name,
        )

    axes[0].set_xlabel("Component rank")
    axes[0].set_ylabel("Variance explained (log)")
    axes[0].set_title("Variance Spectra (Core Layers, per Model)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Right: averaged per paradigm
    for paradigm, specs in spectra_by_paradigm.items():
        max_len = max(len(s) for s in specs)
        padded = np.zeros((len(specs), max_len))
        for i, s in enumerate(specs):
            padded[i, :len(s)] = s
        mean_spec = padded.mean(axis=0)
        std_spec = padded.std(axis=0)

        x = np.arange(1, len(mean_spec) + 1)
        color = paradigm_colors.get(paradigm, "gray")
        axes[1].semilogy(x, mean_spec, color=color, linewidth=2, label=paradigm)
        axes[1].fill_between(x, np.maximum(mean_spec - std_spec, 1e-6),
                             mean_spec + std_spec, color=color, alpha=0.15)

    axes[1].set_xlabel("Component rank")
    axes[1].set_ylabel("Variance explained (log)")
    axes[1].set_title("Paradigm-Averaged Variance Spectra")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_figs / "variance_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved variance_spectra.png")

    # ── 4. Layerwise PR trajectory comparison ─────────────────────────────
    print("=" * 60)
    print("Analysis 4: Layerwise PR trajectories")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PR vs normalized layer position
    for name, info in models.items():
        data = info["data"]
        n_layers = data.shape[0]
        paradigm = info["paradigm"]

        prs = []
        for li in range(n_layers):
            prs.append(participation_ratio(data[li]))
        prs = np.array(prs)
        x = np.linspace(0, 1, n_layers)

        axes[0].plot(x, prs, color=paradigm_colors.get(paradigm, "gray"),
                     linestyle=paradigm_styles.get(paradigm, "-"),
                     alpha=0.7, linewidth=1.5, label=name)

    axes[0].set_xlabel("Normalized layer position")
    axes[0].set_ylabel("Participation Ratio")
    axes[0].set_title("PR Across Layers")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Anisotropy vs layer position
    for name, info in models.items():
        data = info["data"]
        n_layers = data.shape[0]
        paradigm = info["paradigm"]

        anisos = []
        for li in range(n_layers):
            anisos.append(anisotropy(data[li]))
        anisos = np.array(anisos)
        x = np.linspace(0, 1, n_layers)

        axes[1].plot(x, anisos, color=paradigm_colors.get(paradigm, "gray"),
                     linestyle=paradigm_styles.get(paradigm, "-"),
                     alpha=0.7, linewidth=1.5, label=name)

    axes[1].set_xlabel("Normalized layer position")
    axes[1].set_ylabel("Anisotropy")
    axes[1].set_title("Anisotropy Across Layers")
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(True, alpha=0.3)

    # Effective rank vs layer position
    for name, info in models.items():
        data = info["data"]
        n_layers = data.shape[0]
        paradigm = info["paradigm"]

        ers = []
        for li in range(n_layers):
            ers.append(effective_rank(data[li]))
        ers = np.array(ers)
        x = np.linspace(0, 1, n_layers)

        axes[2].plot(x, ers, color=paradigm_colors.get(paradigm, "gray"),
                     linestyle=paradigm_styles.get(paradigm, "-"),
                     alpha=0.7, linewidth=1.5, label=name)

    axes[2].set_xlabel("Normalized layer position")
    axes[2].set_ylabel("Effective Rank")
    axes[2].set_title("Effective Rank Across Layers")
    axes[2].legend(fontsize=6, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_figs / "layerwise_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved layerwise_trajectories.png")

    # ── 5. Bootstrap CI forest plot ───────────────────────────────────────
    print("=" * 60)
    print("Analysis 5: Bootstrap CI forest plots")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Sort by paradigm then PR
    boot_df_sorted = boot_df.sort_values(["paradigm", "PR_mean"], ascending=[True, False])

    y_pos = range(len(boot_df_sorted))

    # PR forest plot
    colors = [paradigm_colors.get(p, "gray") for p in boot_df_sorted["paradigm"]]
    axes[0].barh(y_pos, boot_df_sorted["PR_mean"], color=colors, alpha=0.7, height=0.6)
    axes[0].errorbar(
        boot_df_sorted["PR_mean"], y_pos,
        xerr=[boot_df_sorted["PR_mean"] - boot_df_sorted["PR_ci_low"],
              boot_df_sorted["PR_ci_high"] - boot_df_sorted["PR_mean"]],
        fmt="none", color="black", capsize=3,
    )
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(boot_df_sorted["model"])
    axes[0].set_xlabel("Participation Ratio (core-layer avg)")
    axes[0].set_title("PR with 95% Bootstrap CI")
    axes[0].grid(True, alpha=0.3, axis="x")

    # ER forest plot
    axes[1].barh(y_pos, boot_df_sorted["ER_mean"], color=colors, alpha=0.7, height=0.6)
    axes[1].errorbar(
        boot_df_sorted["ER_mean"], y_pos,
        xerr=[boot_df_sorted["ER_mean"] - boot_df_sorted["ER_ci_low"],
              boot_df_sorted["ER_ci_high"] - boot_df_sorted["ER_mean"]],
        fmt="none", color="black", capsize=3,
    )
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(boot_df_sorted["model"])
    axes[1].set_xlabel("Effective Rank (core-layer avg)")
    axes[1].set_title("Effective Rank with 95% Bootstrap CI")
    axes[1].grid(True, alpha=0.3, axis="x")

    # Anisotropy forest plot
    axes[2].barh(y_pos, boot_df_sorted["aniso_mean"], color=colors, alpha=0.7, height=0.6)
    axes[2].errorbar(
        boot_df_sorted["aniso_mean"], y_pos,
        xerr=[boot_df_sorted["aniso_mean"] - boot_df_sorted["aniso_ci_low"],
              boot_df_sorted["aniso_ci_high"] - boot_df_sorted["aniso_mean"]],
        fmt="none", color="black", capsize=3,
    )
    axes[2].set_yticks(list(y_pos))
    axes[2].set_yticklabels(boot_df_sorted["model"])
    axes[2].set_xlabel("Anisotropy (core-layer avg)")
    axes[2].set_title("Anisotropy with 95% Bootstrap CI")
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_figs / "bootstrap_forest_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved bootstrap_forest_plots.png")

    # ── 6. Cumulative variance explained (% of variance in top-k PCs) ────
    print("=" * 60)
    print("Analysis 6: Cumulative variance explained")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, info in models.items():
        data = info["data"]
        n_layers = data.shape[0]
        paradigm = info["paradigm"]
        core = data[skip:-1] if n_layers > skip + 1 else data

        # Average spectrum across core layers
        specs = []
        for layer_data in core:
            spec = variance_spectrum(layer_data)
            specs.append(spec[:min(16, len(spec))])

        max_len = max(len(s) for s in specs)
        padded = np.zeros((len(specs), max_len))
        for i, s in enumerate(specs):
            padded[i, :len(s)] = s
        avg_spec = padded.mean(axis=0)
        cumsum = np.cumsum(avg_spec)

        ax.plot(range(1, len(cumsum) + 1), cumsum,
                color=paradigm_colors.get(paradigm, "gray"),
                linestyle=paradigm_styles.get(paradigm, "-"),
                alpha=0.7, linewidth=1.5, label=name)

    ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="90% threshold")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance: How Many PCs to Explain 90%/95%?")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 16)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_figs / "cumulative_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved cumulative_variance.png")

    # ── 7. Statistical tests ──────────────────────────────────────────────
    print("=" * 60)
    print("Analysis 7: Statistical significance tests")
    print("=" * 60)

    from scipy import stats

    # Mann-Whitney U test: SSM core PR vs Transformer core PR
    ssm_prs = core_df[core_df["paradigm"] == "ssm"]["PR"].values
    trans_prs = core_df[core_df["paradigm"] == "transformer"]["PR"].values
    hybrid_prs = core_df[core_df["paradigm"] == "hybrid"]["PR"].values

    stat_results = []

    if len(ssm_prs) > 0 and len(trans_prs) > 0:
        u_stat, u_p = stats.mannwhitneyu(ssm_prs, trans_prs, alternative="greater")
        effect_size = u_stat / (len(ssm_prs) * len(trans_prs))  # rank-biserial
        stat_results.append({
            "test": "Mann-Whitney U (SSM > Transformer PR)",
            "statistic": u_stat,
            "p_value": u_p,
            "effect_size": effect_size,
            "n1": len(ssm_prs),
            "n2": len(trans_prs),
        })
        print(f"  SSM vs Transformer PR: U={u_stat:.0f}, p={u_p:.2e}, effect_size={effect_size:.4f}")

    if len(hybrid_prs) > 0 and len(trans_prs) > 0:
        u_stat, u_p = stats.mannwhitneyu(hybrid_prs, trans_prs, alternative="greater")
        effect_size = u_stat / (len(hybrid_prs) * len(trans_prs))
        stat_results.append({
            "test": "Mann-Whitney U (Hybrid > Transformer PR)",
            "statistic": u_stat,
            "p_value": u_p,
            "effect_size": effect_size,
            "n1": len(hybrid_prs),
            "n2": len(trans_prs),
        })
        print(f"  Hybrid vs Transformer PR: U={u_stat:.0f}, p={u_p:.2e}, effect_size={effect_size:.4f}")

    # Normalized PR/d_model
    ssm_norm = core_df[core_df["paradigm"] == "ssm"]["PR_over_d"].values
    trans_norm = core_df[core_df["paradigm"] == "transformer"]["PR_over_d"].values

    if len(ssm_norm) > 0 and len(trans_norm) > 0:
        u_stat, u_p = stats.mannwhitneyu(ssm_norm, trans_norm, alternative="greater")
        effect_size = u_stat / (len(ssm_norm) * len(trans_norm))
        stat_results.append({
            "test": "Mann-Whitney U (SSM > Transformer PR/d_model)",
            "statistic": u_stat,
            "p_value": u_p,
            "effect_size": effect_size,
            "n1": len(ssm_norm),
            "n2": len(trans_norm),
        })
        print(f"  SSM vs Transformer PR/d_model: U={u_stat:.0f}, p={u_p:.2e}, effect_size={effect_size:.4f}")

    stat_df = pd.DataFrame(stat_results)
    stat_df.to_csv(out_tables / "statistical_tests.csv", index=False)
    print()

    # ── 8. Key findings ──────────────────────────────────────────────────
    print("=" * 60)
    print("Key Findings Summary")
    print("=" * 60)

    findings = []
    findings.append(f"models_analyzed: {len(models)}")
    findings.append(f"core_layer_skip: {skip} (first {skip} + last 1)")
    findings.append(f"bootstrap_resamples: {args.n_boot}")

    for para in ["transformer", "ssm", "hybrid", "reasoning"]:
        sub = paradigm_summary[paradigm_summary["paradigm"] == para]
        if not sub.empty:
            row = sub.iloc[0]
            findings.append(f"{para}_core_PR_mean: {row['PR_mean']:.2f} +/- {row['PR_std']:.2f}")
            findings.append(f"{para}_core_PR_over_d: {row['PR_over_d_mean']:.6f}")
            findings.append(f"{para}_core_ER_mean: {row['ER_mean']:.2f}")
            findings.append(f"{para}_core_anisotropy: {row['aniso_mean']:.4f}")

    if len(ssm_prs) > 0 and len(trans_prs) > 0:
        ratio = np.mean(ssm_prs) / max(np.mean(trans_prs), 1e-6)
        findings.append(f"ssm_vs_transformer_PR_ratio: {ratio:.1f}x")

    for sr in stat_results:
        findings.append(f"{sr['test']}: p={sr['p_value']:.2e}")

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")

    for f in findings:
        print(f"  {f}")

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
