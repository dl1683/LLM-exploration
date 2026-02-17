#!/usr/bin/env python3
"""
reasoning_activation_divergence.py — Stage 2: How Reasoning Training Changes Representations

Compares DeepSeek-R1-Distill-Qwen-1.5B (reasoning) vs Qwen3-1.7B (base transformer)
to understand how reasoning distillation transforms internal representations.

Key analyses:
1. Layer-by-layer CKA between base and reasoning models on same prompts
2. Token-position-conditioned geometry (early vs late tokens)
3. Reasoning divergence index (RDI) — where do representations diverge most?
4. Subspace overlap via principal angles between models
5. Prompt-type analysis: math vs code vs language vs factual
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
import torch

# ── Prompt categories for analysis ─────────────────────────────────────────
CATEGORIZED_PROMPTS = {
    "factual": [
        "The capital of France is",
        "The chemical formula for water is H2O, which means",
        "The GDP of the United States in 2024 was approximately",
        "In the year 2025, artificial intelligence has become",
    ],
    "reasoning": [
        "To solve this math problem: 2 + 3 * 4 =",
        "Let me think step by step about this problem.",
        "f(x) = x^2 + 3x - 7, so f'(x) =",
        "In quantum mechanics, the uncertainty principle states that",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "import torch\nmodel = torch.nn.Linear(768,",
        "SELECT * FROM users WHERE age > 18 AND",
        "The transformer architecture was introduced in the paper",
    ],
    "narrative": [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time, in a land far away, there lived a",
        "According to recent studies, climate change has caused",
        "The mitochondria is the powerhouse of the cell because",
    ],
}

# Flat prompt list (must match the order used in representation_geometry.py)
ALL_PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states that",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The quick brown fox jumps over the lazy dog.",
    "According to recent studies, climate change has caused",
    "To solve this math problem: 2 + 3 * 4 =",
    "Once upon a time, in a land far away, there lived a",
    "The chemical formula for water is H2O, which means",
    "import torch\nmodel = torch.nn.Linear(768,",
    "Let me think step by step about this problem.",
    "The transformer architecture was introduced in the paper",
    "SELECT * FROM users WHERE age > 18 AND",
    "In the year 2025, artificial intelligence has become",
    "The mitochondria is the powerhouse of the cell because",
    "f(x) = x^2 + 3x - 7, so f'(x) =",
    "The GDP of the United States in 2024 was approximately",
]

# Map prompt index to category
PROMPT_CATEGORIES = {}
for cat, prompts in CATEGORIZED_PROMPTS.items():
    for p in prompts:
        idx = ALL_PROMPTS.index(p)
        PROMPT_CATEGORIES[idx] = cat


# ── Metric functions ───────────────────────────────────────────────────────
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA between two (n_samples x n_features) matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def participation_ratio(X: np.ndarray) -> float:
    """PR = (sum λ)^2 / sum(λ^2)."""
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return 0.0
    return float((total ** 2) / (s2 ** 2).sum())


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


def principal_angles(X: np.ndarray, Y: np.ndarray, k: int = 5) -> np.ndarray:
    """Compute principal angles between top-k subspaces of X and Y.
    Works with different feature dimensions by using sample-space SVD."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Use sample-space representation (n_samples x n_samples kernel)
    # This avoids dimension mismatch
    Ux, _, _ = np.linalg.svd(X @ X.T, full_matrices=False)
    Uy, _, _ = np.linalg.svd(Y @ Y.T, full_matrices=False)

    k = min(k, Ux.shape[1], Uy.shape[1])
    Ux = Ux[:, :k]
    Uy = Uy[:, :k]

    _, s, _ = np.linalg.svd(Ux.T @ Uy)
    s = np.clip(s, -1, 1)
    return np.arccos(s)


def rsa_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Representational Similarity Analysis: 1 - Spearman correlation between
    pairwise distance matrices. Works across different hidden dimensions.
    Returns 0 if representations are perfectly aligned, 1 if orthogonal."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    dx = pdist(X, metric="cosine")
    dy = pdist(Y, metric="cosine")

    if len(dx) < 2:
        return 0.0
    corr, _ = spearmanr(dx, dy)
    if np.isnan(corr):
        return 1.0
    return float(1.0 - corr)


# ── Token-position analysis ──────────────────────────────────────────────
def extract_token_position_activations(model, tokenizer, prompts: List[str],
                                        max_length: int = 64):
    """Extract hidden states at specific token positions.
    Returns dict with 'early' (first 25%), 'middle' (25-75%), 'late' (last 25%) activations.
    """
    device = next(model.parameters()).device

    all_early = []
    all_mid = []
    all_late = []
    all_last = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length, padding=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
        seq_len = hidden_states[0].shape[1]

        q1 = max(1, seq_len // 4)
        q3 = max(q1 + 1, 3 * seq_len // 4)

        for li, hs in enumerate(hidden_states):
            h = hs[0].float().cpu().numpy()  # (seq_len, hidden_dim)

            if len(all_early) <= li:
                all_early.append([])
                all_mid.append([])
                all_late.append([])
                all_last.append([])

            all_early[li].append(h[:q1].mean(axis=0))
            all_mid[li].append(h[q1:q3].mean(axis=0))
            all_late[li].append(h[q3:].mean(axis=0))
            all_last[li].append(h[-1])

    result = {}
    for pos_name, data in [("early", all_early), ("middle", all_mid),
                            ("late", all_late), ("last_token", all_last)]:
        result[pos_name] = np.array([np.array(layer) for layer in data])

    return result


# ── Main analysis ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-extract", action="store_true",
                        help="Re-extract token-position activations from models")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    cache_dir = Path("analysis/representation_geometry/activation_cache")
    out_dir = Path("analysis/reasoning_activation_divergence")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figs = out_dir / "figures"
    out_figs.mkdir(exist_ok=True)
    out_tables = out_dir / "tables"
    out_tables.mkdir(exist_ok=True)

    # ── Load cached activations ────────────────────────────────────────────
    # Primary comparison pair: DSR1-1.5B (reasoning) vs Qwen3-1.7B (base)
    # Secondary pairs: All transformers vs DSR1, Hybrids vs DSR1

    model_pairs = [
        ("DSR1-1.5B", "reasoning", "Qwen3-1.7B", "transformer"),  # Primary
        ("DSR1-1.5B", "reasoning", "Qwen3-0.6B", "transformer"),
        ("DSR1-1.5B", "reasoning", "Gemma3-1B", "transformer"),
        ("DSR1-1.5B", "reasoning", "Gemma2-2B", "transformer"),
        ("DSR1-1.5B", "reasoning", "FalconH1-1.5B", "hybrid"),
        ("DSR1-1.5B", "reasoning", "Mamba-1.4B", "ssm"),
    ]

    models_data = {}
    needed = set()
    for a, _, b, _ in model_pairs:
        needed.add(a)
        needed.add(b)

    for name in needed:
        cache_path = cache_dir / f"{name}.npz"
        if cache_path.exists():
            models_data[name] = np.load(cache_path)["hidden_states"]
            print(f"  Loaded {name}: {models_data[name].shape}")
        else:
            print(f"  [MISSING] {name}")

    print(f"\nLoaded {len(models_data)} models\n")

    # ── Analysis 1: Layer-by-layer CKA divergence ─────────────────────────
    print("=" * 60)
    print("Analysis 1: Layer-by-layer CKA (Reasoning Divergence Index)")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 6))
    rdi_rows = []

    for model_a, para_a, model_b, para_b in model_pairs:
        if model_a not in models_data or model_b not in models_data:
            continue

        data_a = models_data[model_a]
        data_b = models_data[model_b]

        n_layers_a = data_a.shape[0]
        n_layers_b = data_b.shape[0]
        min_layers = min(n_layers_a, n_layers_b)

        ckas = []
        for li in range(min_layers):
            # Use prompt-level mean representations
            a_repr = data_a[li]  # (n_prompts, hidden_dim)
            b_repr = data_b[li]  # (n_prompts, hidden_dim)
            cka_val = linear_cka(a_repr, b_repr)
            ckas.append(cka_val)

            rdi_rows.append({
                "model_a": model_a,
                "paradigm_a": para_a,
                "model_b": model_b,
                "paradigm_b": para_b,
                "layer": li,
                "layer_frac": li / (min_layers - 1),
                "cka": cka_val,
                "rdi": 1.0 - cka_val,  # Reasoning Divergence Index
            })

        x = np.linspace(0, 1, len(ckas))
        label = f"{model_a} vs {model_b}"
        is_primary = model_b == "Qwen3-1.7B"
        ax.plot(x, ckas, linewidth=3 if is_primary else 1.5,
                alpha=1.0 if is_primary else 0.5,
                label=label)

    ax.set_xlabel("Normalized layer position")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("Layer-by-Layer CKA: Where Does Reasoning Diverge?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_figs / "layerwise_cka_divergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    rdi_df = pd.DataFrame(rdi_rows)
    rdi_df.to_csv(out_tables / "layerwise_rdi.csv", index=False)
    print("  Saved layerwise_cka_divergence.png")

    # Find max divergence layer for primary pair
    primary_rdi = rdi_df[(rdi_df["model_b"] == "Qwen3-1.7B")]
    if not primary_rdi.empty:
        max_rdi_row = primary_rdi.loc[primary_rdi["rdi"].idxmax()]
        print(f"  Max divergence: layer {int(max_rdi_row['layer'])} "
              f"(frac={max_rdi_row['layer_frac']:.2f}), RDI={max_rdi_row['rdi']:.4f}")

    # ── Analysis 2: Prompt-category-conditioned geometry ──────────────────
    print()
    print("=" * 60)
    print("Analysis 2: Prompt-category-conditioned representation geometry")
    print("=" * 60)

    cat_rows = []
    skip = 2  # Skip first 2 and last 1 layers (embedding/head)

    for cat_name, cat_prompts in CATEGORIZED_PROMPTS.items():
        cat_indices = [ALL_PROMPTS.index(p) for p in cat_prompts]

        for name, data in models_data.items():
            paradigm = "reasoning" if name == "DSR1-1.5B" else \
                       "ssm" if "Mamba" in name else \
                       "hybrid" if "Falcon" in name or "Zamba" in name else \
                       "transformer"

            n_layers = data.shape[0]
            core = data[skip:-1] if n_layers > skip + 1 else data

            # Extract only this category's prompts
            cat_data = core[:, cat_indices, :]

            # Mean across core layers
            avg = cat_data.mean(axis=0)  # (n_cat_prompts, hidden_dim)
            pr = participation_ratio(avg)
            aniso = anisotropy(avg)

            cat_rows.append({
                "model": name,
                "paradigm": paradigm,
                "category": cat_name,
                "n_prompts": len(cat_indices),
                "PR": pr,
                "anisotropy": aniso,
            })

    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(out_tables / "category_conditioned_geometry.csv", index=False)

    # Plot: PR by category for each model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pivot_pr = cat_df.pivot_table(index="model", columns="category", values="PR")
    pivot_aniso = cat_df.pivot_table(index="model", columns="category", values="anisotropy")

    # Sort by paradigm
    paradigm_order_map = {"reasoning": 0, "transformer": 1, "hybrid": 2, "ssm": 3}
    model_paradigms = cat_df[["model", "paradigm"]].drop_duplicates().set_index("model")["paradigm"]
    sort_key = pivot_pr.index.map(lambda x: paradigm_order_map.get(model_paradigms.get(x, ""), 5))
    pivot_pr = pivot_pr.iloc[sort_key.argsort()]
    pivot_aniso = pivot_aniso.iloc[sort_key.argsort()]

    pivot_pr.plot(kind="barh", ax=axes[0], width=0.7)
    axes[0].set_xlabel("Participation Ratio")
    axes[0].set_title("PR by Prompt Category")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="x")

    pivot_aniso.plot(kind="barh", ax=axes[1], width=0.7)
    axes[1].set_xlabel("Anisotropy")
    axes[1].set_title("Anisotropy by Prompt Category")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_figs / "category_conditioned_geometry.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved category_conditioned_geometry.png")

    # ── Analysis 3: RSA-based representational distance by layer ─────────
    print()
    print("=" * 60)
    print("Analysis 3: RSA-based representational distance (reasoning vs base)")
    print("=" * 60)

    if "DSR1-1.5B" in models_data and "Qwen3-1.7B" in models_data:
        dsr = models_data["DSR1-1.5B"]
        qwen = models_data["Qwen3-1.7B"]
        min_layers = min(dsr.shape[0], qwen.shape[0])

        # RSA per layer: compare pairwise-distance structure between models
        rsa_rows = []
        for li in range(min_layers):
            rsa_dist = rsa_distance(dsr[li], qwen[li])
            rsa_rows.append({
                "layer": li,
                "layer_frac": li / (min_layers - 1),
                "rsa_distance": rsa_dist,
            })

        rsa_df = pd.DataFrame(rsa_rows)
        rsa_df.to_csv(out_tables / "layerwise_rsa_distance.csv", index=False)

        # RSA by category: compute per-category pairwise structure
        cat_rsa_rows = []
        for cat, cat_prompts in CATEGORIZED_PROMPTS.items():
            cat_indices = [ALL_PROMPTS.index(p) for p in cat_prompts]
            for li in range(min_layers):
                rsa_dist = rsa_distance(dsr[li, cat_indices], qwen[li, cat_indices])
                cat_rsa_rows.append({
                    "layer": li,
                    "layer_frac": li / (min_layers - 1),
                    "category": cat,
                    "rsa_distance": rsa_dist,
                })

        cat_rsa_df = pd.DataFrame(cat_rsa_rows)
        cat_rsa_df.to_csv(out_tables / "category_rsa_distance.csv", index=False)

        # Plot: RSA distance by layer (overall + by category)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(rsa_df["layer_frac"], rsa_df["rsa_distance"],
                     linewidth=2, color="black")
        axes[0].set_xlabel("Normalized layer position")
        axes[0].set_ylabel("RSA Distance (1 - Spearman r)")
        axes[0].set_title("DSR1 vs Qwen3: Representational Similarity by Layer")
        axes[0].grid(True, alpha=0.3)

        cat_colors = {"factual": "#2196F3", "reasoning": "#FF5722",
                      "code": "#4CAF50", "narrative": "#9C27B0"}

        for cat in ["factual", "reasoning", "code", "narrative"]:
            cat_data = cat_rsa_df[cat_rsa_df["category"] == cat]
            axes[1].plot(cat_data["layer_frac"], cat_data["rsa_distance"],
                         color=cat_colors[cat], linewidth=2, label=cat)

        axes[1].set_xlabel("Normalized layer position")
        axes[1].set_ylabel("RSA Distance")
        axes[1].set_title("RSA Distance by Prompt Category")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_figs / "category_distance_by_layer.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved category_distance_by_layer.png")

    # ── Analysis 4: Subspace overlap (principal angles) ───────────────────
    print()
    print("=" * 60)
    print("Analysis 4: Subspace overlap via principal angles")
    print("=" * 60)

    pa_rows = []
    if "DSR1-1.5B" in models_data and "Qwen3-1.7B" in models_data:
        dsr = models_data["DSR1-1.5B"]
        qwen = models_data["Qwen3-1.7B"]
        min_layers = min(dsr.shape[0], qwen.shape[0])

        for li in range(min_layers):
            k = min(5, dsr[li].shape[0] - 1)
            if k < 1:
                continue
            angles = principal_angles(dsr[li], qwen[li], k=k)
            pa_rows.append({
                "layer": li,
                "layer_frac": li / (min_layers - 1),
                "angle_1": float(angles[0]) if len(angles) > 0 else np.nan,
                "angle_2": float(angles[1]) if len(angles) > 1 else np.nan,
                "angle_3": float(angles[2]) if len(angles) > 2 else np.nan,
                "mean_angle": float(angles.mean()),
                "max_angle": float(angles.max()),
            })

    if pa_rows:
        pa_df = pd.DataFrame(pa_rows)
        pa_df.to_csv(out_tables / "principal_angles.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pa_df["layer_frac"], np.degrees(pa_df["angle_1"]),
                label="1st angle", linewidth=2)
        ax.plot(pa_df["layer_frac"], np.degrees(pa_df["mean_angle"]),
                label="Mean angle", linewidth=2, linestyle="--")
        ax.plot(pa_df["layer_frac"], np.degrees(pa_df["max_angle"]),
                label="Max angle", linewidth=1.5, linestyle=":", alpha=0.7)
        ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5, label="Orthogonal")
        ax.set_xlabel("Normalized layer position")
        ax.set_ylabel("Principal Angle (degrees)")
        ax.set_title("DSR1 vs Qwen3: Subspace Alignment by Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 95)

        plt.tight_layout()
        plt.savefig(out_figs / "principal_angles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved principal_angles.png")

    # ── Analysis 5: Cross-paradigm reasoning divergence comparison ────────
    print()
    print("=" * 60)
    print("Analysis 5: Cross-paradigm reasoning divergence comparison")
    print("=" * 60)

    # For each paradigm comparison with DSR1, compute the mean CKA across layers
    comparison_summary = []
    for model_a, para_a, model_b, para_b in model_pairs:
        if model_a not in models_data or model_b not in models_data:
            continue

        pair_rdi = rdi_df[(rdi_df["model_a"] == model_a) & (rdi_df["model_b"] == model_b)]
        if pair_rdi.empty:
            continue

        comparison_summary.append({
            "comparison": f"{model_a} vs {model_b}",
            "paradigm_b": para_b,
            "mean_cka": pair_rdi["cka"].mean(),
            "min_cka": pair_rdi["cka"].min(),
            "max_rdi": pair_rdi["rdi"].max(),
            "max_divergence_layer_frac": pair_rdi.loc[pair_rdi["rdi"].idxmax(), "layer_frac"],
        })

    comp_df = pd.DataFrame(comparison_summary)
    comp_df.to_csv(out_tables / "reasoning_divergence_summary.csv", index=False)
    print(comp_df.to_string(index=False))

    # ── Key findings ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Key Findings")
    print("=" * 60)

    findings = []
    findings.append(f"models_compared: {len(models_data)}")

    if not primary_rdi.empty:
        findings.append(f"primary_pair: DSR1-1.5B vs Qwen3-1.7B")
        findings.append(f"primary_mean_cka: {primary_rdi['cka'].mean():.4f}")
        findings.append(f"primary_max_rdi: {max_rdi_row['rdi']:.4f}")
        findings.append(f"primary_max_divergence_layer: {int(max_rdi_row['layer'])}")
        findings.append(f"primary_max_divergence_frac: {max_rdi_row['layer_frac']:.3f}")

    if not comp_df.empty:
        for _, row in comp_df.iterrows():
            findings.append(f"{row['comparison']}_mean_cka: {row['mean_cka']:.4f}")

    # Category analysis
    if not cat_df.empty:
        dsr_cats = cat_df[cat_df["model"] == "DSR1-1.5B"]
        for _, row in dsr_cats.iterrows():
            findings.append(f"DSR1_{row['category']}_PR: {row['PR']:.2f}")

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")

    for f in findings:
        print(f"  {f}")

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
