#!/usr/bin/env python3
"""
representation_geometry.py — Cross-Architecture Representation Geometry Analysis

Stage 1 of model internals experiments.
Computes CKA similarity, intrinsic dimensionality, and anisotropy
across layers for different model architectures (Transformer, SSM, Hybrid, RWKV, etc.).

Loads models one at a time in NF4 quantization, caches activations to disk,
then computes pairwise CKA matrices.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ── Model registry for experiments ──────────────────────────────────────────
# Organized by paradigm. Each entry: (model_id, short_name, paradigm)
EXPERIMENT_MODELS: List[Tuple[str, str, str]] = [
    # Transformer anchors
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "transformer"),
    ("google/gemma-3-1b-it", "Gemma3-1B", "transformer"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "transformer"),
    ("google/gemma-2-2b-it", "Gemma2-2B", "transformer"),
    # SSM (Mamba)
    ("state-spaces/mamba-790m-hf", "Mamba-790M", "ssm"),
    ("state-spaces/mamba-1.4b-hf", "Mamba-1.4B", "ssm"),
    ("state-spaces/mamba-2.8b-hf", "Mamba-2.8B", "ssm"),
    # Hybrid
    ("tiiuae/Falcon-H1-0.5B-Instruct", "FalconH1-0.5B", "hybrid"),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "FalconH1-1.5B", "hybrid"),
    ("Zyphra/Zamba2-1.2B", "Zamba2-1.2B", "hybrid"),
    ("nvidia/Hymba-1.5B-Base", "Hymba-1.5B", "hybrid"),
    # RWKV
    ("RWKV/RWKV7-Goose-World3-1.5B-HF", "RWKV7-1.5B", "rwkv"),
    # Reasoning
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning"),
]

# Prompt pack: diverse short prompts to get representative activations
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


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices (n_samples x n_features)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def intrinsic_dimensionality(X: np.ndarray) -> float:
    """Participation ratio as estimate of intrinsic dimensionality."""
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    s2_sum = s2.sum()
    if s2_sum < 1e-12:
        return 0.0
    return float((s2_sum ** 2) / (s2 ** 2).sum())


def anisotropy(X: np.ndarray) -> float:
    """Mean cosine similarity between all pairs of representations."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_norm = X / norms
    cos_sim = X_norm @ X_norm.T
    n = cos_sim.shape[0]
    if n < 2:
        return 0.0
    # Mean of upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(cos_sim[mask].mean())


def load_model_and_tokenizer(model_id: str, use_4bit: bool = True):
    """Load model in NF4 quantization for memory efficiency."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if use_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
        except Exception:
            print(f"  [WARN] NF4 not available for {model_id}, falling back to bf16")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    max_length: int = 64,
) -> np.ndarray:
    """Extract hidden states from all layers for all prompts.

    Returns: array of shape (n_layers+1, n_tokens, hidden_size)
    where layer 0 is the embedding output.
    """
    all_hidden: List[List[np.ndarray]] = []

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden)
            except Exception as e:
                print(f"  [WARN] output_hidden_states failed: {e}")
                print("  Trying manual hook extraction...")
                hidden_states = _extract_with_hooks(model, inputs)
                if hidden_states is None:
                    continue

        # Take mean over sequence positions to get one vector per layer
        layer_means = []
        for h in hidden_states:
            if isinstance(h, torch.Tensor):
                layer_means.append(h.squeeze(0).float().cpu().numpy().mean(axis=0))
            else:
                layer_means.append(np.zeros(1))
        all_hidden.append(layer_means)

    if not all_hidden:
        return np.array([])

    # Stack: (n_layers, n_prompts, hidden_size)
    n_layers = len(all_hidden[0])
    result = np.zeros((n_layers, len(all_hidden), all_hidden[0][0].shape[0]))
    for i, layers in enumerate(all_hidden):
        for j, h in enumerate(layers):
            result[j, i, :h.shape[0]] = h

    return result


def _extract_with_hooks(model, inputs):
    """Fallback: extract hidden states via forward hooks for models that
    don't support output_hidden_states."""
    hidden_states = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            elif isinstance(output, torch.Tensor):
                h = output
            else:
                return
            hidden_states.append(h.detach())
        return hook_fn

    hooks = []
    # Try to find the main layer stack
    layer_list = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 2:
            layer_list = module
            break

    if layer_list is None:
        return None

    for i, layer in enumerate(layer_list):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    if not hidden_states:
        return None

    return hidden_states


def extract_token_level_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    max_length: int = 64,
) -> np.ndarray:
    """Extract ALL token-level hidden states (not averaged).

    Returns: array of shape (n_layers+1, total_tokens, hidden_size)
    """
    all_layer_tokens: Dict[int, List[np.ndarray]] = {}

    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            except Exception:
                hidden_states = _extract_with_hooks(model, inputs)
                if hidden_states is None:
                    continue

        for layer_idx, h in enumerate(hidden_states):
            if isinstance(h, torch.Tensor):
                tokens = h.squeeze(0).float().cpu().numpy()  # (seq_len, hidden)
                if layer_idx not in all_layer_tokens:
                    all_layer_tokens[layer_idx] = []
                all_layer_tokens[layer_idx].append(tokens)

    if not all_layer_tokens:
        return np.array([])

    n_layers = max(all_layer_tokens.keys()) + 1
    result = []
    for layer_idx in range(n_layers):
        if layer_idx in all_layer_tokens:
            result.append(np.concatenate(all_layer_tokens[layer_idx], axis=0))
        else:
            result.append(np.zeros((1, 1)))
    return result


def compute_self_cka_matrix(hidden_states: np.ndarray) -> np.ndarray:
    """Compute CKA between all layer pairs within a single model.

    hidden_states: (n_layers, n_prompts, hidden_size)
    Returns: (n_layers, n_layers) CKA matrix
    """
    n_layers = hidden_states.shape[0]
    cka_matrix = np.eye(n_layers)

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            cka_val = linear_cka(hidden_states[i], hidden_states[j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val

    return cka_matrix


def compute_cross_cka(
    hidden_a: np.ndarray,
    hidden_b: np.ndarray,
) -> np.ndarray:
    """Compute CKA between layers of model A and model B.

    hidden_a: (n_layers_a, n_prompts, hidden_a)
    hidden_b: (n_layers_b, n_prompts, hidden_b)
    Returns: (n_layers_a, n_layers_b) CKA matrix
    """
    n_a, n_b = hidden_a.shape[0], hidden_b.shape[0]
    cka_matrix = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            cka_matrix[i, j] = linear_cka(hidden_a[i], hidden_b[j])

    return cka_matrix


def compute_layer_metrics(hidden_states: np.ndarray) -> pd.DataFrame:
    """Compute per-layer intrinsic dim and anisotropy.

    hidden_states: (n_layers, n_prompts, hidden_size)
    """
    rows = []
    for layer_idx in range(hidden_states.shape[0]):
        X = hidden_states[layer_idx]
        rows.append({
            "layer": layer_idx,
            "intrinsic_dim": intrinsic_dimensionality(X),
            "anisotropy": anisotropy(X),
            "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
            "std_norm": float(np.linalg.norm(X, axis=1).std()),
        })
    return pd.DataFrame(rows)


def save_self_cka_heatmap(cka_matrix: np.ndarray, model_name: str, out_path: Path):
    """Save self-CKA heatmap for a single model."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cka_matrix, vmin=0, vmax=1, cmap="viridis",
        square=True, ax=ax,
        xticklabels=5, yticklabels=5,
    )
    ax.set_title(f"Self-CKA: {model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cross_cka_heatmap(
    cka_matrix: np.ndarray,
    name_a: str,
    name_b: str,
    out_path: Path,
):
    """Save cross-model CKA heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cka_matrix, vmin=0, vmax=1, cmap="viridis",
        ax=ax, xticklabels=5, yticklabels=5,
    )
    ax.set_title(f"Cross-CKA: {name_a} vs {name_b}")
    ax.set_xlabel(f"{name_b} layers")
    ax.set_ylabel(f"{name_a} layers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_layer_metrics_plot(
    all_metrics: Dict[str, pd.DataFrame],
    out_dir: Path,
):
    """Save combined layer metrics plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Intrinsic dimensionality
    ax = axes[0]
    for name, df in all_metrics.items():
        depth_frac = np.linspace(0, 1, len(df))
        ax.plot(depth_frac, df["intrinsic_dim"], label=name, alpha=0.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Intrinsic dimensionality")
    ax.set_title("Intrinsic Dimensionality Across Depth")
    ax.legend(fontsize=7, ncol=2)

    # Anisotropy
    ax = axes[1]
    for name, df in all_metrics.items():
        depth_frac = np.linspace(0, 1, len(df))
        ax.plot(depth_frac, df["anisotropy"], label=name, alpha=0.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Anisotropy (mean cosine sim)")
    ax.set_title("Anisotropy Across Depth")
    ax.legend(fontsize=7, ncol=2)

    # Mean norm
    ax = axes[2]
    for name, df in all_metrics.items():
        depth_frac = np.linspace(0, 1, len(df))
        ax.plot(depth_frac, df["mean_norm"], label=name, alpha=0.8)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("Mean activation norm")
    ax.set_title("Activation Norm Across Depth")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(out_dir / "layer_metrics_combined.png", dpi=200)
    plt.close()


def save_paradigm_summary_plot(summary_df: pd.DataFrame, out_dir: Path):
    """Plot paradigm-level summary statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    paradigm_order = ["transformer", "ssm", "hybrid", "rwkv", "reasoning"]
    palette = {
        "transformer": "#1f77b4",
        "ssm": "#ff7f0e",
        "hybrid": "#2ca02c",
        "rwkv": "#d62728",
        "reasoning": "#9467bd",
    }

    for ax_idx, metric in enumerate(["mean_intrinsic_dim", "mean_anisotropy", "self_cka_block_diag"]):
        ax = axes[ax_idx]
        data = summary_df[summary_df["paradigm"].isin(paradigm_order)].copy()
        if data.empty or metric not in data.columns:
            continue
        plot_order = [p for p in paradigm_order if p in data["paradigm"].values]
        sns.barplot(
            data=data, x="paradigm", y=metric, hue="paradigm",
            order=plot_order, hue_order=plot_order,
            palette=palette, ax=ax, legend=False,
        )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_dir / "paradigm_summary.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cross-architecture representation geometry analysis")
    parser.add_argument("--outdir", default="analysis/representation_geometry", help="Output directory")
    parser.add_argument("--max-models", type=int, default=None, help="Limit number of models (for testing)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable NF4 quantization")
    parser.add_argument("--max-length", type=int, default=64, help="Max token length per prompt")
    parser.add_argument("--cache-dir", default=None, help="Directory to cache activations")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_tables = out_dir / "tables"
    out_figs = out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = EXPERIMENT_MODELS
    if args.max_models:
        models_to_run = models_to_run[:args.max_models]

    use_4bit = not args.no_4bit
    all_metrics: Dict[str, pd.DataFrame] = {}
    all_hidden_cache: Dict[str, Path] = {}
    model_info: List[Dict[str, Any]] = []

    print(f"Running representation geometry on {len(models_to_run)} models")
    print(f"Prompts: {len(PROMPT_PACK)}, max_length: {args.max_length}")
    print(f"NF4 quantization: {use_4bit}")
    print()

    # Phase 1: Extract activations from each model
    for model_id, short_name, paradigm in models_to_run:
        cache_path = cache_dir / f"{short_name.replace('/', '_')}.npz"

        if cache_path.exists():
            print(f"[CACHED] {short_name} ({paradigm})")
            cached = np.load(cache_path)
            hidden = cached["hidden_states"]
        else:
            print(f"[LOADING] {short_name} ({paradigm}) — {model_id}")
            t0 = time.time()
            try:
                model, tokenizer = load_model_and_tokenizer(model_id, use_4bit=use_4bit)
                print(f"  Loaded in {time.time() - t0:.1f}s")

                t1 = time.time()
                hidden = extract_hidden_states(
                    model, tokenizer, PROMPT_PACK,
                    max_length=args.max_length,
                )
                print(f"  Extracted {hidden.shape[0]} layers × {hidden.shape[1]} prompts in {time.time() - t1:.1f}s")

                np.savez_compressed(cache_path, hidden_states=hidden)
                print(f"  Cached to {cache_path}")

                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

        if hidden.size == 0:
            print(f"  [SKIP] Empty hidden states")
            continue

        all_hidden_cache[short_name] = cache_path

        # Compute per-model metrics
        metrics = compute_layer_metrics(hidden)
        metrics["model"] = short_name
        metrics["paradigm"] = paradigm
        all_metrics[short_name] = metrics

        # Self-CKA
        self_cka = compute_self_cka_matrix(hidden)
        save_self_cka_heatmap(self_cka, short_name, out_figs / f"self_cka_{short_name}.png")

        # Block diagonal score (mean CKA within first/last third)
        n = self_cka.shape[0]
        third = max(n // 3, 1)
        early_block = self_cka[:third, :third]
        late_block = self_cka[-third:, -third:]
        block_diag_mean = (np.mean(early_block) + np.mean(late_block)) / 2
        off_diag = self_cka[:third, -third:]
        block_off_mean = np.mean(off_diag)

        info = {
            "model": short_name,
            "model_id": model_id,
            "paradigm": paradigm,
            "n_layers": hidden.shape[0],
            "hidden_size": hidden.shape[2],
            "mean_intrinsic_dim": float(metrics["intrinsic_dim"].mean()),
            "max_intrinsic_dim": float(metrics["intrinsic_dim"].max()),
            "mean_anisotropy": float(metrics["anisotropy"].mean()),
            "max_anisotropy": float(metrics["anisotropy"].max()),
            "self_cka_block_diag": float(block_diag_mean),
            "self_cka_block_off": float(block_off_mean),
            "self_cka_block_ratio": float(block_diag_mean / max(block_off_mean, 1e-6)),
        }
        model_info.append(info)
        print(f"  Layers={info['n_layers']}, IntDim={info['mean_intrinsic_dim']:.1f}, "
              f"Aniso={info['mean_anisotropy']:.3f}, BlockCKA={info['self_cka_block_diag']:.3f}")
        print()

    # Phase 2: Cross-model CKA for key pairs
    print("=" * 60)
    print("Phase 2: Cross-model CKA comparisons")
    print("=" * 60)

    names = list(all_hidden_cache.keys())
    cross_cka_results: List[Dict[str, Any]] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            para_a = next(p for _, n, p in models_to_run if n == name_a)
            para_b = next(p for _, n, p in models_to_run if n == name_b)

            cached_a = np.load(all_hidden_cache[name_a])["hidden_states"]
            cached_b = np.load(all_hidden_cache[name_b])["hidden_states"]

            # Match number of prompts (should be same)
            n_prompts = min(cached_a.shape[1], cached_b.shape[1])
            cached_a = cached_a[:, :n_prompts, :]
            cached_b = cached_b[:, :n_prompts, :]

            cross_cka = compute_cross_cka(cached_a, cached_b)
            max_cka = float(cross_cka.max())
            mean_cka = float(cross_cka.mean())

            # Diagonal alignment score (how well layers correspond 1:1)
            min_layers = min(cross_cka.shape[0], cross_cka.shape[1])
            diag_indices = np.array([
                cross_cka[int(i * cross_cka.shape[0] / min_layers),
                          int(i * cross_cka.shape[1] / min_layers)]
                for i in range(min_layers)
            ])
            diag_mean = float(diag_indices.mean())

            cross_cka_results.append({
                "model_a": name_a,
                "model_b": name_b,
                "paradigm_a": para_a,
                "paradigm_b": para_b,
                "pair_type": f"{para_a}_vs_{para_b}" if para_a <= para_b else f"{para_b}_vs_{para_a}",
                "max_cka": max_cka,
                "mean_cka": mean_cka,
                "diag_alignment": diag_mean,
                "n_layers_a": cross_cka.shape[0],
                "n_layers_b": cross_cka.shape[1],
            })

            # Save heatmap for key cross-paradigm pairs
            if para_a != para_b:
                save_cross_cka_heatmap(
                    cross_cka, name_a, name_b,
                    out_figs / f"cross_cka_{name_a}_vs_{name_b}.png",
                )

            print(f"  {name_a} vs {name_b}: max_CKA={max_cka:.3f}, mean={mean_cka:.3f}, diag={diag_mean:.3f}")

    # Phase 3: Save results
    print()
    print("=" * 60)
    print("Phase 3: Saving results")
    print("=" * 60)

    # Model summary table
    summary_df = pd.DataFrame(model_info)
    summary_df.to_csv(out_tables / "model_summary.csv", index=False)

    # Layer metrics
    if all_metrics:
        all_layer_df = pd.concat(all_metrics.values(), ignore_index=True)
        all_layer_df.to_csv(out_tables / "layer_metrics.csv", index=False)

    # Cross-CKA results
    cross_df = pd.DataFrame(cross_cka_results)
    cross_df.to_csv(out_tables / "cross_cka_summary.csv", index=False)

    # Combined plots
    save_layer_metrics_plot(all_metrics, out_figs)
    if not summary_df.empty:
        save_paradigm_summary_plot(summary_df, out_figs)

    # Key findings
    findings = []
    findings.append(f"models_analyzed: {len(model_info)}")
    findings.append(f"cross_cka_pairs: {len(cross_cka_results)}")

    if not summary_df.empty:
        for para in ["transformer", "ssm", "hybrid", "rwkv", "reasoning"]:
            sub = summary_df[summary_df["paradigm"] == para]
            if not sub.empty:
                findings.append(
                    f"{para}_mean_intrinsic_dim: {sub['mean_intrinsic_dim'].mean():.2f}"
                )
                findings.append(
                    f"{para}_mean_anisotropy: {sub['mean_anisotropy'].mean():.4f}"
                )
                findings.append(
                    f"{para}_self_cka_block_ratio: {sub['self_cka_block_ratio'].mean():.3f}"
                )

    if not cross_df.empty:
        # Compare within-paradigm vs cross-paradigm CKA
        within = cross_df[cross_df["paradigm_a"] == cross_df["paradigm_b"]]
        across = cross_df[cross_df["paradigm_a"] != cross_df["paradigm_b"]]
        if not within.empty and not across.empty:
            findings.append(f"within_paradigm_mean_cka: {within['mean_cka'].mean():.4f}")
            findings.append(f"cross_paradigm_mean_cka: {across['mean_cka'].mean():.4f}")
            findings.append(f"cka_gap: {within['mean_cka'].mean() - across['mean_cka'].mean():.4f}")

        # Best cross-paradigm alignment
        if not across.empty:
            best = across.loc[across["max_cka"].idxmax()]
            findings.append(
                f"best_cross_paradigm_pair: {best['model_a']}_vs_{best['model_b']} "
                f"(max_cka={best['max_cka']:.4f})"
            )

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")

    print(f"\nResults saved to {out_dir}")
    print(f"Key findings: {findings_path}")
    for f in findings:
        print(f"  {f}")


if __name__ == "__main__":
    main()
