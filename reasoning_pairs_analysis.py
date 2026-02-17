#!/usr/bin/env python3
"""
reasoning_pairs_analysis.py — Stage 2b: Multi-Pair Reasoning Geometry Analysis

Expands reasoning activation divergence analysis from n=1 to n=8 matched pairs.
Each pair has a reasoning-trained model and its matched base model.
Tests whether reasoning training universally compresses representations (PR→1D)
or if this is distillation-specific.

Training method diversity:
  - Strong-to-weak distillation (Qwen3 family)
  - SFT distillation from R1 traces (DeepSeek-R1, NVIDIA Nemotron)
  - Multi-stage RL (LiquidAI, AMD Instella)
  - SFT + alignment (SmolLM3)
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# ── Matched Reasoning Pairs ───────────────────────────────────────────────
# (reasoning_id, reasoning_name, base_id, base_name, training_method, arch_family)
REASONING_PAIRS: List[Tuple[str, str, str, str, str, str]] = [
    # Already cached (reasoning side)
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B-R",
     "Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base",
     "strong-to-weak-distill", "qwen3"),

    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B-R",
     "Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base",
     "strong-to-weak-distill", "qwen3"),

    # True matched base for DSR1
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B",
     "Qwen/Qwen2.5-Math-1.5B", "Qwen2.5-Math-1.5B",
     "sft-distill-r1", "qwen2.5"),

    # NVIDIA reasoning distill
    ("nvidia/OpenReasoning-Nemotron-1.5B", "Nemotron-1.5B-R",
     "Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B",
     "sft-distill-r1-5M", "qwen2.5"),

    # Hybrid SSM/Attention - RL trained
    ("LiquidAI/LFM2.5-1.2B-Thinking", "LFM2.5-Think",
     "LiquidAI/LFM2.5-1.2B-Instruct", "LFM2.5-Instruct",
     "multi-stage-rl", "lfm-hybrid"),

    # RL-heavy (SFT + 3-stage GRPO)
    ("amd/Instella-3B-Math", "Instella-3B-Math",
     "amd/Instella-3B-Instruct", "Instella-3B-Instruct",
     "sft-grpo-rl", "olmo"),

    # Llama-family R1 distill
    ("NousResearch/DeepHermes-3-Llama-3-3B-Preview", "DeepHermes-3B",
     "meta-llama/Llama-3.2-3B-Instruct", "Llama3.2-3B",
     "r1-distill", "llama3"),

    # SmolLM3 with reasoning midtraining
    ("HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B-R",
     "HuggingFaceTB/SmolLM3-3B-Base", "SmolLM3-3B-Base",
     "sft-apo-midtrain", "smollm"),
]

# Same prompts as representation_geometry.py
PROMPT_PACK = [
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


# ── Metric functions ───────────────────────────────────────────────────────
def participation_ratio(X: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return 0.0
    return float((total ** 2) / (s2 ** 2).sum())


def effective_rank(X: np.ndarray) -> float:
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
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_norm = X / norms
    cos_sim = X_norm @ X_norm.T
    n = cos_sim.shape[0]
    if n < 2:
        return 0.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(cos_sim[mask].mean())


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ── Model loading & extraction ────────────────────────────────────────────
def load_model_and_tokenizer(model_id: str, use_4bit: bool = True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


def extract_hidden_states(model, tokenizer, prompts, max_length=64):
    """Extract mean-pooled hidden states per layer per prompt."""
    device = next(model.parameters()).device
    all_hidden = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length, padding=False).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        prompt_repr = []
        for hs in hidden_states:
            h = hs[0].float().cpu().numpy()
            prompt_repr.append(h.mean(axis=0))  # Mean-pool across tokens

        all_hidden.append(prompt_repr)

    n_layers = len(all_hidden[0])
    n_prompts = len(all_hidden)
    hidden_dim = all_hidden[0][0].shape[0]

    result = np.zeros((n_layers, n_prompts, hidden_dim))
    for pi in range(n_prompts):
        for li in range(n_layers):
            result[li, pi] = all_hidden[pi][li]

    return result


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--core-skip", type=int, default=2)
    args = parser.parse_args()

    cache_dir = Path("analysis/reasoning_pairs/activation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("analysis/reasoning_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figs = out_dir / "figures"
    out_figs.mkdir(exist_ok=True)
    out_tables = out_dir / "tables"
    out_tables.mkdir(exist_ok=True)

    # Also check main activation cache for already-cached models
    main_cache = Path("analysis/representation_geometry/activation_cache")

    use_4bit = not args.no_4bit
    pairs_to_run = REASONING_PAIRS
    if args.max_pairs:
        pairs_to_run = pairs_to_run[:args.max_pairs]

    print(f"Running reasoning pairs analysis: {len(pairs_to_run)} pairs")
    print(f"Prompts: {len(PROMPT_PACK)}, max_length: {args.max_length}")
    print(f"NF4 quantization: {use_4bit}")
    print()

    skip = args.core_skip
    pair_results = []

    for pair_idx, (r_id, r_name, b_id, b_name, method, family) in enumerate(pairs_to_run):
        print(f"\n{'=' * 60}")
        print(f"Pair {pair_idx + 1}/{len(pairs_to_run)}: {r_name} vs {b_name}")
        print(f"  Method: {method}, Family: {family}")
        print(f"{'=' * 60}")

        # Load/cache both models
        pair_data = {}
        for model_id, model_name, role in [(r_id, r_name, "reasoning"), (b_id, b_name, "base")]:
            safe_name = model_name.replace("/", "_").replace(" ", "_")
            cache_path = cache_dir / f"{safe_name}.npz"

            # Check main cache first (for models already processed in Stage 1)
            main_cache_name = model_name.replace("-R", "").replace("-Base", "")
            main_cache_path = main_cache / f"{main_cache_name}.npz"

            # Also try exact name match in main cache
            main_exact = main_cache / f"{safe_name}.npz"

            if cache_path.exists():
                print(f"  [CACHED] {model_name} ({role})")
                data = np.load(cache_path)["hidden_states"]
            elif main_exact.exists():
                print(f"  [MAIN-CACHED] {model_name} ({role})")
                data = np.load(main_exact)["hidden_states"]
            elif main_cache_path.exists() and role == "reasoning":
                # For reasoning models that might be cached under different name
                print(f"  [MAIN-CACHED] {model_name} ({role}) as {main_cache_name}")
                data = np.load(main_cache_path)["hidden_states"]
            else:
                print(f"  [LOADING] {model_name} ({role}) — {model_id}")
                t0 = time.time()
                try:
                    model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
                    print(f"    Loaded in {time.time() - t0:.1f}s")

                    t1 = time.time()
                    data = extract_hidden_states(model, tokenizer, PROMPT_PACK,
                                                 max_length=args.max_length)
                    print(f"    Extracted {data.shape[0]} layers × {data.shape[1]} prompts "
                          f"in {time.time() - t1:.1f}s")

                    np.savez_compressed(cache_path, hidden_states=data)
                    print(f"    Cached to {cache_path}")

                    del model, tokenizer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    data = None

            if data is not None:
                pair_data[role] = data
                n_layers, n_prompts, d_model = data.shape

                # Compute core metrics
                core = data[skip:-1] if n_layers > skip + 1 else data
                core_pr_vals = [participation_ratio(core[li]) for li in range(len(core))]
                core_er_vals = [effective_rank(core[li]) for li in range(len(core))]
                core_aniso_vals = [anisotropy(core[li]) for li in range(len(core))]

                print(f"    Layers={n_layers}, d_model={d_model}, "
                      f"CorePR={np.mean(core_pr_vals):.2f}, "
                      f"CoreAniso={np.mean(core_aniso_vals):.3f}")

                pair_data[f"{role}_metrics"] = {
                    "core_pr": np.mean(core_pr_vals),
                    "core_pr_std": np.std(core_pr_vals),
                    "core_er": np.mean(core_er_vals),
                    "core_aniso": np.mean(core_aniso_vals),
                    "n_layers": n_layers,
                    "d_model": d_model,
                }

        # Skip pair if either model failed
        if "reasoning" not in pair_data or "base" not in pair_data:
            print(f"  [SKIP] Missing model data for this pair")
            continue

        # Compute paired CKA divergence
        r_data = pair_data["reasoning"]
        b_data = pair_data["base"]
        min_layers = min(r_data.shape[0], b_data.shape[0])

        ckas = []
        for li in range(min_layers):
            cka_val = linear_cka(r_data[li], b_data[li])
            ckas.append(cka_val)
        ckas = np.array(ckas)

        # Core CKA (excluding embedding + final)
        core_ckas = ckas[skip:-1] if len(ckas) > skip + 1 else ckas

        # Find max divergence
        rdis = 1.0 - ckas
        max_rdi_layer = int(np.argmax(rdis))
        max_rdi_frac = max_rdi_layer / max(min_layers - 1, 1)

        r_met = pair_data["reasoning_metrics"]
        b_met = pair_data["base_metrics"]

        pair_result = {
            "pair_idx": pair_idx,
            "reasoning_model": r_name,
            "base_model": b_name,
            "training_method": method,
            "arch_family": family,
            "reasoning_core_pr": r_met["core_pr"],
            "base_core_pr": b_met["core_pr"],
            "pr_delta": r_met["core_pr"] - b_met["core_pr"],
            "pr_ratio": r_met["core_pr"] / max(b_met["core_pr"], 1e-6),
            "reasoning_core_aniso": r_met["core_aniso"],
            "base_core_aniso": b_met["core_aniso"],
            "aniso_delta": r_met["core_aniso"] - b_met["core_aniso"],
            "mean_cka": float(ckas.mean()),
            "core_mean_cka": float(core_ckas.mean()),
            "min_cka": float(ckas.min()),
            "max_rdi": float(rdis.max()),
            "max_rdi_layer": max_rdi_layer,
            "max_rdi_frac": max_rdi_frac,
            "reasoning_n_layers": r_met["n_layers"],
            "base_n_layers": b_met["n_layers"],
            "reasoning_d_model": r_met["d_model"],
            "base_d_model": b_met["d_model"],
        }
        pair_results.append(pair_result)

        print(f"\n  Results:")
        print(f"    Reasoning CorePR={r_met['core_pr']:.2f}, Base CorePR={b_met['core_pr']:.2f}")
        print(f"    PR delta={pair_result['pr_delta']:.2f}, ratio={pair_result['pr_ratio']:.2f}")
        print(f"    Mean CKA={pair_result['mean_cka']:.4f}, "
              f"Core CKA={pair_result['core_mean_cka']:.4f}")
        print(f"    Max divergence at layer {max_rdi_layer} "
              f"(frac={max_rdi_frac:.2f}), RDI={pair_result['max_rdi']:.4f}")

    # ── Aggregate results ──────────────────────────────────────────────────
    if not pair_results:
        print("\nNo pairs completed successfully!")
        return

    results_df = pd.DataFrame(pair_results)
    results_df.to_csv(out_tables / "reasoning_pairs_summary.csv", index=False)

    print(f"\n\n{'=' * 60}")
    print(f"AGGREGATE RESULTS ({len(pair_results)} pairs)")
    print(f"{'=' * 60}")

    # Test: does reasoning training consistently compress PR?
    pr_deltas = results_df["pr_delta"].values
    from scipy import stats

    # Wilcoxon signed-rank test (paired, non-parametric)
    if len(pr_deltas) >= 5:
        stat, p_val = stats.wilcoxon(pr_deltas, alternative="less")  # H1: reasoning < base
        print(f"\n  Wilcoxon test (reasoning PR < base PR):")
        print(f"    W={stat:.1f}, p={p_val:.4f}, n={len(pr_deltas)}")
        print(f"    Mean PR delta: {pr_deltas.mean():.3f} ± {pr_deltas.std():.3f}")
    else:
        stat, p_val = np.nan, np.nan
        print(f"\n  Too few pairs ({len(pr_deltas)}) for Wilcoxon test")

    # By training method
    print(f"\n  By training method:")
    for method in results_df["training_method"].unique():
        sub = results_df[results_df["training_method"] == method]
        print(f"    {method}: PR delta={sub['pr_delta'].mean():.3f}, "
              f"core CKA={sub['core_mean_cka'].mean():.4f}, n={len(sub)}")

    # ── Visualization ──────────────────────────────────────────────────────
    # 1. PR comparison: reasoning vs base (paired)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Paired dot plot
    for i, row in results_df.iterrows():
        color = {"strong-to-weak-distill": "#2196F3", "sft-distill-r1": "#FF5722",
                 "sft-distill-r1-5M": "#FF9800", "multi-stage-rl": "#4CAF50",
                 "sft-grpo-rl": "#9C27B0", "r1-distill": "#795548",
                 "sft-apo-midtrain": "#009688"}.get(row["training_method"], "gray")

        axes[0].plot([0, 1], [row["base_core_pr"], row["reasoning_core_pr"]],
                     "o-", color=color, markersize=8, linewidth=2, alpha=0.7)
        axes[0].annotate(row["reasoning_model"], (1.05, row["reasoning_core_pr"]),
                         fontsize=6, va="center")

    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Base", "Reasoning"])
    axes[0].set_ylabel("Core Layer PR")
    axes[0].set_title("PR: Base → Reasoning (each line = one pair)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # PR delta bar chart
    method_colors = {m: c for m, c in zip(
        results_df["training_method"].unique(),
        plt.cm.Set2(np.linspace(0, 1, len(results_df["training_method"].unique())))
    )}
    colors = [method_colors[m] for m in results_df["training_method"]]

    axes[1].barh(range(len(results_df)), results_df["pr_delta"],
                 color=colors, alpha=0.8)
    axes[1].set_yticks(range(len(results_df)))
    axes[1].set_yticklabels([f"{r['reasoning_model']}\nvs {r['base_model']}"
                             for _, r in results_df.iterrows()], fontsize=7)
    axes[1].axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    axes[1].set_xlabel("PR Delta (reasoning - base)")
    axes[1].set_title("PR Change from Reasoning Training")
    axes[1].grid(True, alpha=0.3, axis="x")

    # CKA distribution
    axes[2].barh(range(len(results_df)), results_df["core_mean_cka"],
                 color=colors, alpha=0.8)
    axes[2].set_yticks(range(len(results_df)))
    axes[2].set_yticklabels([r["reasoning_model"] for _, r in results_df.iterrows()],
                             fontsize=7)
    axes[2].set_xlabel("Core Mean CKA")
    axes[2].set_title("Base→Reasoning Representation Similarity")
    axes[2].set_xlim(0, 1.05)
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_figs / "reasoning_pairs_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved reasoning_pairs_overview.png")

    # 2. Max divergence layer distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(results_df["max_rdi_frac"], results_df["max_rdi"],
               s=100, c=colors, edgecolors="black", linewidths=0.5, zorder=5)
    for _, row in results_df.iterrows():
        ax.annotate(f"{row['reasoning_model']}\n({row['training_method']})",
                     (row["max_rdi_frac"], row["max_rdi"]),
                     fontsize=6, ha="center", va="bottom")
    ax.set_xlabel("Layer position of max divergence (normalized)")
    ax.set_ylabel("Max RDI (1 - CKA)")
    ax.set_title("Where Does Reasoning Diverge Most?")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_figs / "divergence_layer_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved divergence_layer_scatter.png")

    # ── Key findings ──────────────────────────────────────────────────────
    findings = [
        f"pairs_analyzed: {len(pair_results)}",
        f"training_methods: {', '.join(results_df['training_method'].unique())}",
        f"arch_families: {', '.join(results_df['arch_family'].unique())}",
        f"mean_pr_delta: {pr_deltas.mean():.3f} (neg = compression)",
        f"median_pr_delta: {np.median(pr_deltas):.3f}",
        f"pairs_with_compression: {(pr_deltas < 0).sum()}/{len(pr_deltas)}",
        f"wilcoxon_p: {p_val:.4f}" if not np.isnan(p_val) else "wilcoxon_p: N/A",
        f"mean_core_cka: {results_df['core_mean_cka'].mean():.4f}",
    ]

    for _, row in results_df.iterrows():
        findings.append(
            f"{row['reasoning_model']}_vs_{row['base_model']}: "
            f"PR_delta={row['pr_delta']:.2f}, CKA={row['core_mean_cka']:.4f}, "
            f"method={row['training_method']}"
        )

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("Key Findings")
    print(f"{'=' * 60}")
    for f in findings:
        print(f"  {f}")


if __name__ == "__main__":
    main()
