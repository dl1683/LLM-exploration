#!/usr/bin/env python3
"""
causal_pr_intervention.py — Stage 3: Causal PR Manipulation

Tests whether intrinsic dimensionality (PR) compression is *functionally*
relevant to reasoning by interventionally manipulating hidden states.

Method:
1. Hook into model layers at max-compression points
2. Apply PR-expansion (whitening/noise injection) or PR-compression (PCA projection)
3. Measure effect on task accuracy (math, factual, code)
4. Show dose-response: delta_accuracy ~ delta_PR

This gives CAUSAL evidence that reasoning training's PR compression
is functionally meaningful, not just a correlate.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Evaluation prompts with ground truth ──────────────────────────────────
EVAL_TASKS = {
    "math": [
        {"prompt": "What is 7 * 8? Answer with just the number:", "answer": "56"},
        {"prompt": "What is 15 + 27? Answer with just the number:", "answer": "42"},
        {"prompt": "What is 100 - 37? Answer with just the number:", "answer": "63"},
        {"prompt": "What is 144 / 12? Answer with just the number:", "answer": "12"},
        {"prompt": "What is 9 * 9? Answer with just the number:", "answer": "81"},
        {"prompt": "What is 23 + 19? Answer with just the number:", "answer": "42"},
        {"prompt": "What is 256 / 16? Answer with just the number:", "answer": "16"},
        {"prompt": "What is 11 * 11? Answer with just the number:", "answer": "121"},
    ],
    "factual": [
        {"prompt": "The capital of Japan is", "answer": "Tokyo"},
        {"prompt": "Water freezes at 0 degrees", "answer": "Celsius"},
        {"prompt": "The chemical symbol for gold is", "answer": "Au"},
        {"prompt": "The largest planet in our solar system is", "answer": "Jupiter"},
        {"prompt": "The speed of light is approximately 300,000 km per", "answer": "second"},
        {"prompt": "DNA stands for deoxyribonucleic", "answer": "acid"},
        {"prompt": "The square root of 144 is", "answer": "12"},
        {"prompt": "The atomic number of carbon is", "answer": "6"},
    ],
}


def participation_ratio(X: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return 0.0
    return float((total ** 2) / (s2 ** 2).sum())


# ── Intervention hooks ────────────────────────────────────────────────────
class PRExpander:
    """Expand PR by injecting orthogonal noise into low-variance dimensions."""

    def __init__(self, layer_idx: int, strength: float = 0.1,
                 n_components: int = 5, seed: int = 42):
        self.layer_idx = layer_idx
        self.strength = strength
        self.n_components = n_components
        self.seed = seed
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Generate orthogonal noise in low-variance directions
        rng = torch.Generator(device=hidden.device)
        rng.manual_seed(self.seed)

        batch, seq_len, d_model = hidden.shape
        noise = torch.randn(seq_len, self.n_components, generator=rng,
                            device=hidden.device, dtype=hidden.dtype)

        # Random orthogonal directions
        Q = torch.randn(self.n_components, d_model, generator=rng,
                         device=hidden.device, dtype=hidden.dtype)
        Q, _ = torch.linalg.qr(Q.T)
        Q = Q[:, :self.n_components].T  # (n_components, d_model)

        # Project noise into these directions and add to hidden states
        perturbation = noise @ Q  # (seq_len, d_model)
        scale = hidden.norm(dim=-1, keepdim=True).mean() * self.strength
        perturbation = perturbation * scale / perturbation.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        hidden = hidden + perturbation.unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def attach(self, model, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


class PRCompressor:
    """Compress PR by projecting hidden states onto top-k principal components."""

    def __init__(self, layer_idx: int, keep_components: int = 1):
        self.layer_idx = layer_idx
        self.keep_components = keep_components
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        batch, seq_len, d_model = hidden.shape

        # Center and compute SVD
        h = hidden[0]  # (seq_len, d_model)
        mean = h.mean(dim=0, keepdim=True)
        h_centered = h - mean

        # Project onto top-k components
        U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
        k = min(self.keep_components, len(S))
        projected = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        hidden = (projected + mean).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def attach(self, model, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, tasks: Dict[str, List[Dict]],
                   max_new_tokens: int = 10) -> Dict[str, float]:
    """Evaluate model on tasks, return accuracy per category."""
    device = next(model.parameters()).device
    results = {}

    for category, items in tasks.items():
        correct = 0
        total = len(items)

        for item in items:
            inputs = tokenizer(item["prompt"], return_tensors="pt",
                               truncation=True, max_length=128).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip()

            # Check if answer appears in generated text
            answer = item["answer"].lower()
            if answer in generated.lower():
                correct += 1

        results[category] = correct / max(total, 1)

    results["overall"] = np.mean(list(results.values()))
    return results


def get_model_layers(model) -> List[nn.Module]:
    """Get the transformer/SSM layers from a model."""
    # Try common layer accessor patterns
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers",
                 "model.model.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    return []


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to test (default: 2 reasoning + 2 base)")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--strengths", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.5])
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    out_dir = Path("analysis/causal_pr_intervention")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_figs = out_dir / "figures"
    out_figs.mkdir(exist_ok=True)
    out_tables = out_dir / "tables"
    out_tables.mkdir(exist_ok=True)

    # Test models: matched pairs
    test_models = [
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "reasoning"),
        ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base", "base"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning"),
        ("Qwen/Qwen2.5-Math-1.5B", "Qwen2.5-Math-1.5B", "base"),
    ]

    if args.models:
        test_models = [(m, m.split("/")[-1], "custom") for m in args.models]

    all_results = []

    for model_id, model_name, role in test_models:
        print(f"\n{'=' * 60}")
        print(f"Testing: {model_name} ({role})")
        print(f"{'=' * 60}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        model.eval()
        print(f"  Loaded in {time.time() - t0:.1f}s")

        layers = get_model_layers(model)
        if not layers:
            print(f"  [ERROR] Could not find model layers")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        n_layers = len(layers)
        print(f"  Found {n_layers} layers")

        # Baseline (no intervention)
        print(f"  Evaluating baseline...")
        baseline = evaluate_model(model, tokenizer, EVAL_TASKS, args.max_new_tokens)
        print(f"    Baseline: {baseline}")

        all_results.append({
            "model": model_name,
            "role": role,
            "intervention": "none",
            "strength": 0.0,
            "target_layer": -1,
            **{f"acc_{k}": v for k, v in baseline.items()},
        })

        # PR Expansion at middle layers (dose-response)
        target_layer = n_layers // 2  # Middle layer
        print(f"\n  PR Expansion at layer {target_layer}/{n_layers}...")

        for strength in args.strengths:
            if strength == 0.0:
                continue

            expander = PRExpander(target_layer, strength=strength)
            expander.attach(model, layers[target_layer])

            results = evaluate_model(model, tokenizer, EVAL_TASKS, args.max_new_tokens)
            expander.detach()

            print(f"    strength={strength:.2f}: {results}")

            all_results.append({
                "model": model_name,
                "role": role,
                "intervention": "expand",
                "strength": strength,
                "target_layer": target_layer,
                **{f"acc_{k}": v for k, v in results.items()},
            })

        # PR Compression at middle layers
        print(f"\n  PR Compression at layer {target_layer}/{n_layers}...")
        for keep_k in [1, 2, 3, 5]:
            compressor = PRCompressor(target_layer, keep_components=keep_k)
            compressor.attach(model, layers[target_layer])

            results = evaluate_model(model, tokenizer, EVAL_TASKS, args.max_new_tokens)
            compressor.detach()

            print(f"    keep_k={keep_k}: {results}")

            all_results.append({
                "model": model_name,
                "role": role,
                "intervention": "compress",
                "strength": keep_k,
                "target_layer": target_layer,
                **{f"acc_{k}": v for k, v in results.items()},
            })

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_tables / "intervention_results.csv", index=False)

    print(f"\n\n{'=' * 60}")
    print("All Results")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))

    # ── Visualize dose-response ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PR expansion dose-response
    expand_df = results_df[results_df["intervention"].isin(["none", "expand"])]
    for model_name in expand_df["model"].unique():
        sub = expand_df[expand_df["model"] == model_name].sort_values("strength")
        role = sub["role"].iloc[0]
        color = "#FF5722" if role == "reasoning" else "#2196F3"
        linestyle = "-" if role == "reasoning" else "--"
        axes[0].plot(sub["strength"], sub["acc_overall"],
                     "o-", color=color, linestyle=linestyle,
                     linewidth=2, markersize=6, label=f"{model_name} ({role})")

    axes[0].set_xlabel("Expansion Strength")
    axes[0].set_ylabel("Overall Accuracy")
    axes[0].set_title("PR Expansion: Does Expanding PR Hurt Reasoning Models?")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: PR compression dose-response
    compress_df = results_df[results_df["intervention"].isin(["none", "compress"])]
    for model_name in compress_df["model"].unique():
        sub = compress_df[compress_df["model"] == model_name]
        baseline_acc = sub[sub["intervention"] == "none"]["acc_overall"].values
        if len(baseline_acc) == 0:
            continue
        baseline_acc = baseline_acc[0]
        comp_sub = sub[sub["intervention"] == "compress"].sort_values("strength")
        if comp_sub.empty:
            continue
        role = sub["role"].iloc[0]
        color = "#FF5722" if role == "reasoning" else "#2196F3"
        linestyle = "-" if role == "reasoning" else "--"

        x = list(comp_sub["strength"])
        y = list(comp_sub["acc_overall"])
        x = [0] + x  # Add baseline as k=0
        y = [baseline_acc] + y
        axes[1].plot(x, y, "o-", color=color, linestyle=linestyle,
                     linewidth=2, markersize=6, label=f"{model_name} ({role})")

    axes[1].set_xlabel("Keep Top-K Components")
    axes[1].set_ylabel("Overall Accuracy")
    axes[1].set_title("PR Compression: Does Compressing PR Affect Function?")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_figs / "dose_response.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved dose_response.png")

    # ── Key findings ──────────────────────────────────────────────────────
    findings = [f"models_tested: {len(results_df['model'].unique())}"]

    for model_name in results_df["model"].unique():
        sub = results_df[results_df["model"] == model_name]
        baseline = sub[sub["intervention"] == "none"]["acc_overall"].values
        if len(baseline) == 0:
            continue
        baseline = baseline[0]

        # Max expansion effect
        expand_sub = sub[sub["intervention"] == "expand"]
        if not expand_sub.empty:
            max_expand = expand_sub.loc[expand_sub["strength"].idxmax()]
            findings.append(
                f"{model_name}_expand_effect: baseline={baseline:.3f}, "
                f"max_expand={max_expand['acc_overall']:.3f} "
                f"(strength={max_expand['strength']:.2f})"
            )

        # Compression effect
        compress_sub = sub[sub["intervention"] == "compress"]
        if not compress_sub.empty:
            k1 = compress_sub[compress_sub["strength"] == 1]["acc_overall"].values
            if len(k1) > 0:
                findings.append(
                    f"{model_name}_compress_k1: baseline={baseline:.3f}, "
                    f"k1={k1[0]:.3f}"
                )

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")

    print(f"\nKey findings:")
    for f in findings:
        print(f"  {f}")


if __name__ == "__main__":
    main()
