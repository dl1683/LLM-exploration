#!/usr/bin/env python3
"""
jitter_pressure_analysis.py — Stage 4: Jitter-Pressure Inference Stability (JPIS)

Codex-designed analysis measuring dynamic stability under hidden-state
perturbation. Unlike geometry-focused analyses (PR, anisotropy, CKA), this
measures how close each model is to response collapse under noise injection.

Method:
1. Run clean baseline inference, capture outputs + logits
2. Inject scaled Gaussian noise into hidden states at various layers via hooks
3. Measure output-level changes: accuracy, flip rate, edit distance,
   entropy, logit margin, KL divergence
4. Build pressure curves per model/paradigm/domain
5. Extract critical-pressure thresholds and sensitivity slopes

Tests 10 models across 5 paradigms (Transformer, SSM, Hybrid, RWKV, Reasoning)
on math and factual domains.
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
import torch.nn as nn
from scipy import stats as sp_stats

# ── Model registry ───────────────────────────────────────────────────────
# (model_id, short_name, paradigm) — all < 3B, proven to work in this repo
JITTER_MODELS: List[Tuple[str, str, str]] = [
    # Transformers
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "transformer"),
    ("google/gemma-3-1b-it", "Gemma3-1B", "transformer"),
    ("google/gemma-2-2b-it", "Gemma2-2B", "transformer"),
    # SSM (Mamba)
    ("state-spaces/mamba-790m-hf", "Mamba-790M", "ssm"),
    ("state-spaces/mamba-1.4b-hf", "Mamba-1.4B", "ssm"),
    # Hybrid
    ("tiiuae/Falcon-H1-0.5B-Instruct", "FalconH1-0.5B", "hybrid"),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "FalconH1-1.5B", "hybrid"),
    ("Zyphra/Zamba2-1.2B", "Zamba2-1.2B", "hybrid"),
    # RWKV
    ("RWKV/RWKV7-Goose-World3-1.5B-HF", "RWKV7-1.5B", "rwkv"),
    # Reasoning
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning"),
]

# ── Evaluation prompts with ground truth ─────────────────────────────────
EVAL_PROMPTS = {
    "math": [
        {"prompt": "What is 7 * 8? Answer with just the number:", "answer": "56"},
        {"prompt": "What is 15 + 27? Answer with just the number:", "answer": "42"},
        {"prompt": "What is 100 - 37? Answer with just the number:", "answer": "63"},
        {"prompt": "What is 144 / 12? Answer with just the number:", "answer": "12"},
        {"prompt": "What is 9 * 9? Answer with just the number:", "answer": "81"},
        {"prompt": "What is 23 + 19? Answer with just the number:", "answer": "42"},
        {"prompt": "What is 256 / 16? Answer with just the number:", "answer": "16"},
        {"prompt": "What is 11 * 11? Answer with just the number:", "answer": "121"},
        {"prompt": "What is 50 - 23? Answer with just the number:", "answer": "27"},
        {"prompt": "What is 6 * 7? Answer with just the number:", "answer": "42"},
        {"prompt": "What is 200 / 8? Answer with just the number:", "answer": "25"},
        {"prompt": "What is 33 + 44? Answer with just the number:", "answer": "77"},
        {"prompt": "What is 13 * 5? Answer with just the number:", "answer": "65"},
        {"prompt": "What is 96 / 8? Answer with just the number:", "answer": "12"},
        {"prompt": "What is 88 - 29? Answer with just the number:", "answer": "59"},
        {"prompt": "What is 14 * 6? Answer with just the number:", "answer": "84"},
        {"prompt": "What is 75 + 38? Answer with just the number:", "answer": "113"},
        {"prompt": "What is 1000 / 25? Answer with just the number:", "answer": "40"},
        {"prompt": "What is 17 * 3? Answer with just the number:", "answer": "51"},
        {"prompt": "What is 500 - 167? Answer with just the number:", "answer": "333"},
        {"prompt": "What is 8 * 12? Answer with just the number:", "answer": "96"},
        {"prompt": "What is 45 + 67? Answer with just the number:", "answer": "112"},
        {"prompt": "What is 360 / 9? Answer with just the number:", "answer": "40"},
        {"prompt": "What is 19 * 4? Answer with just the number:", "answer": "76"},
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
        {"prompt": "The boiling point of water is 100 degrees", "answer": "Celsius"},
        {"prompt": "The chemical formula for table salt is", "answer": "NaCl"},
        {"prompt": "The Earth orbits the", "answer": "Sun"},
        {"prompt": "Photosynthesis converts carbon dioxide and water into glucose and", "answer": "oxygen"},
        {"prompt": "The fastest land animal is the", "answer": "cheetah"},
        {"prompt": "The human body has 206", "answer": "bones"},
        {"prompt": "The chemical symbol for iron is", "answer": "Fe"},
        {"prompt": "The Great Wall is located in", "answer": "China"},
        {"prompt": "Pi is approximately equal to 3.14159 or simply", "answer": "3.14"},
        {"prompt": "Einstein's famous equation is E equals mc", "answer": "squared"},
        {"prompt": "The currency of Japan is the", "answer": "yen"},
        {"prompt": "The smallest prime number is", "answer": "2"},
        {"prompt": "Gravity on Earth is approximately 9.8 meters per second", "answer": "squared"},
        {"prompt": "The capital of France is", "answer": "Paris"},
        {"prompt": "The mitochondria is the powerhouse of the", "answer": "cell"},
        {"prompt": "The periodic table was created by Dmitri", "answer": "Mendeleev"},
    ],
}

# Perturbation config
LAYER_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 0.9]  # relative layer positions
STRENGTHS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SEEDS = [42, 137, 2049]
MAX_NEW_TOKENS = 10


# ── Perturbation hook ────────────────────────────────────────────────────
class JitterHook:
    """Inject scaled Gaussian noise into hidden states at a specific layer."""

    def __init__(self, strength: float, seed: int):
        self.strength = strength
        self.seed = seed
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Use CPU generator (CUDA generators have device constraints)
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        noise = torch.randn(hidden.shape, generator=rng, dtype=torch.float32).to(hidden.device)
        # Scale noise by activation norm so perturbation is relative
        act_norm = hidden.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        perturbation = self.strength * act_norm * noise / np.sqrt(hidden.shape[-1])

        hidden = hidden + perturbation.to(hidden.dtype)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ── Model utilities ──────────────────────────────────────────────────────
def load_model_and_tokenizer(model_id: str):
    """Load model in NF4 quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )
    model.eval()
    return model, tokenizer


def get_model_layers(model) -> List[nn.Module]:
    """Get the transformer/SSM layers from a model."""
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers",
                 "model.model.layers", "backbone.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList) and len(obj) > 2:
                return list(obj)
        except AttributeError:
            continue

    # Fallback: find first large ModuleList
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 2:
            return list(module)
    return []


def get_layer_indices(n_layers: int) -> List[int]:
    """Get layer indices at the standard fractional positions."""
    indices = []
    for frac in LAYER_FRACTIONS:
        idx = int(frac * (n_layers - 1))
        idx = max(0, min(idx, n_layers - 1))
        if idx not in indices:
            indices.append(idx)
    return sorted(indices)


# ── Inference with logit capture ─────────────────────────────────────────
def generate_with_logits(
    model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
) -> Tuple[str, Optional[np.ndarray]]:
    """Generate text and capture first-token logits.

    Returns (generated_text, first_token_logits_or_None).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # First token logits
    first_logits = None
    if hasattr(outputs, "scores") and outputs.scores and len(outputs.scores) > 0:
        first_logits = outputs.scores[0][0].float().cpu().numpy()

    return text, first_logits


# ── Metric computation ───────────────────────────────────────────────────
def check_correct(generated: str, answer: str) -> bool:
    return answer.lower() in generated.lower()


def edit_distance_normalized(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance."""
    if not s1 and not s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, m)


def logit_entropy(logits: np.ndarray) -> float:
    """Shannon entropy of softmax distribution."""
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log(probs)))


def logit_margin(logits: np.ndarray) -> float:
    """Difference between top-1 and top-2 logit values."""
    sorted_l = np.sort(logits)[::-1]
    if len(sorted_l) < 2:
        return 0.0
    return float(sorted_l[0] - sorted_l[1])


def kl_divergence(logits_p: np.ndarray, logits_q: np.ndarray) -> float:
    """KL(P || Q) from logit vectors."""
    lp = logits_p - logits_p.max()
    lq = logits_q - logits_q.max()
    p = np.exp(lp) / np.exp(lp).sum()
    q = np.exp(lq) / np.exp(lq).sum()
    q = np.maximum(q, 1e-12)
    p_pos = p[p > 1e-12]
    q_pos = q[p > 1e-12]
    return float(np.sum(p_pos * np.log(p_pos / q_pos)))


# ── Statistical utilities ────────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap mean with CI."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.mean(values)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def sensitivity_slope(strengths: np.ndarray, metric: np.ndarray) -> float:
    """Linear slope of metric vs strength at low alpha (first 3 points)."""
    n = min(3, len(strengths))
    if n < 2:
        return 0.0
    slope, _, _, _, _ = sp_stats.linregress(strengths[:n], metric[:n])
    return float(slope)


def critical_pressure(strengths: np.ndarray, accuracies: np.ndarray, threshold: float) -> float:
    """Smallest strength where accuracy drops below threshold. Returns inf if never."""
    for s, a in zip(strengths, accuracies):
        if a < threshold:
            return float(s)
    return float("inf")


# ── Main pipeline ────────────────────────────────────────────────────────
def run_baseline(model, tokenizer, prompts_by_domain: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Run clean baseline inference for all prompts. Returns dict of domain -> list of result dicts."""
    baseline = {}
    for domain, items in prompts_by_domain.items():
        domain_results = []
        for item in items:
            text, logits = generate_with_logits(model, tokenizer, item["prompt"])
            domain_results.append({
                "prompt": item["prompt"],
                "answer": item["answer"],
                "generated": text,
                "correct": check_correct(text, item["answer"]),
                "logits": logits,
                "entropy": logit_entropy(logits) if logits is not None else None,
                "margin": logit_margin(logits) if logits is not None else None,
            })
        baseline[domain] = domain_results
    return baseline


def run_perturbed(
    model, tokenizer, layers: List[nn.Module], layer_idx: int,
    strength: float, seed: int,
    prompts_by_domain: Dict[str, List[Dict]], baseline: Dict[str, List[Dict]],
) -> List[Dict]:
    """Run perturbed inference at one (layer, strength, seed). Returns list of per-prompt metrics."""
    hook = JitterHook(strength=strength, seed=seed)
    hook.attach(layers[layer_idx])

    rows = []
    for domain, items in prompts_by_domain.items():
        base_results = baseline[domain]
        for i, item in enumerate(items):
            text, logits = generate_with_logits(model, tokenizer, item["prompt"])
            base = base_results[i]

            correct = check_correct(text, item["answer"])
            flip = (text.strip().lower() != base["generated"].strip().lower())
            edit_dist = edit_distance_normalized(text, base["generated"])

            entropy = logit_entropy(logits) if logits is not None else None
            margin = logit_margin(logits) if logits is not None else None
            kl = None
            if logits is not None and base["logits"] is not None:
                min_len = min(len(logits), len(base["logits"]))
                kl = kl_divergence(base["logits"][:min_len], logits[:min_len])

            rows.append({
                "domain": domain,
                "prompt_idx": i,
                "layer_idx": layer_idx,
                "strength": strength,
                "seed": seed,
                "baseline_correct": base["correct"],
                "perturbed_correct": correct,
                "flip": flip,
                "edit_distance": edit_dist,
                "baseline_entropy": base["entropy"],
                "perturbed_entropy": entropy,
                "baseline_margin": base["margin"],
                "perturbed_margin": margin,
                "kl_divergence": kl,
            })

    hook.detach()
    return rows


def compute_summaries(df: pd.DataFrame, model_name: str, paradigm: str) -> pd.DataFrame:
    """Compute aggregate summaries per (model, layer, strength, domain)."""
    rows = []
    for (layer_idx, strength, domain), grp in df.groupby(["layer_idx", "strength", "domain"]):
        acc = grp["perturbed_correct"].mean()
        flip_rate = grp["flip"].mean()
        mean_edit = grp["edit_distance"].mean()
        mean_kl = grp["kl_divergence"].dropna().mean() if grp["kl_divergence"].notna().any() else None
        mean_entropy = grp["perturbed_entropy"].dropna().mean() if grp["perturbed_entropy"].notna().any() else None
        mean_margin = grp["perturbed_margin"].dropna().mean() if grp["perturbed_margin"].notna().any() else None

        rows.append({
            "model": model_name,
            "paradigm": paradigm,
            "layer_idx": layer_idx,
            "strength": strength,
            "domain": domain,
            "accuracy": acc,
            "flip_rate": flip_rate,
            "mean_edit_distance": mean_edit,
            "mean_kl_divergence": mean_kl,
            "mean_entropy": mean_entropy,
            "mean_margin": mean_margin,
            "n_prompts": len(grp),
        })
    return pd.DataFrame(rows)


# ── Visualization ────────────────────────────────────────────────────────
PARADIGM_COLORS = {
    "transformer": "#2196F3",
    "ssm": "#4CAF50",
    "hybrid": "#FF9800",
    "rwkv": "#9C27B0",
    "reasoning": "#FF5722",
    "custom": "#666666",
}


def plot_accuracy_vs_pressure(summary_df: pd.DataFrame, out_dir: Path):
    """Fig 1: Accuracy vs pressure strength, one line per model, faceted by domain."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, domain in enumerate(["math", "factual"]):
        ax = axes[ax_idx]
        sub = summary_df[summary_df["domain"] == domain]

        # Average over layers for model-level curve
        model_curves = sub.groupby(["model", "paradigm", "strength"]).agg(
            accuracy=("accuracy", "mean")
        ).reset_index()

        for model_name in model_curves["model"].unique():
            msub = model_curves[model_curves["model"] == model_name].sort_values("strength")
            paradigm = msub["paradigm"].iloc[0]
            color = PARADIGM_COLORS.get(paradigm, "#666")
            ax.plot(msub["strength"], msub["accuracy"], "o-",
                    color=color, linewidth=2, markersize=5,
                    label=f"{model_name}", alpha=0.8)

        ax.set_xlabel(r"Perturbation Strength ($\alpha$)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{domain.title()} Domain", fontsize=14)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Jitter-Pressure: Accuracy vs Perturbation Strength", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_pressure_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_flip_rate(summary_df: pd.DataFrame, out_dir: Path):
    """Fig 2: Flip rate vs pressure, faceted by layer position."""
    layer_indices = sorted(summary_df["layer_idx"].unique())
    n_layers_plot = min(5, len(layer_indices))
    fig, axes = plt.subplots(1, n_layers_plot, figsize=(4 * n_layers_plot, 6), sharey=True)
    if n_layers_plot == 1:
        axes = [axes]

    for ax_idx, layer_idx in enumerate(layer_indices[:n_layers_plot]):
        ax = axes[ax_idx]
        sub = summary_df[summary_df["layer_idx"] == layer_idx]

        # Average over domains
        model_curves = sub.groupby(["model", "paradigm", "strength"]).agg(
            flip_rate=("flip_rate", "mean")
        ).reset_index()

        for model_name in model_curves["model"].unique():
            msub = model_curves[model_curves["model"] == model_name].sort_values("strength")
            paradigm = msub["paradigm"].iloc[0]
            color = PARADIGM_COLORS.get(paradigm, "#666")
            ax.plot(msub["strength"], msub["flip_rate"], "o-",
                    color=color, linewidth=1.5, markersize=4, alpha=0.7,
                    label=f"{model_name}")

        ax.set_xlabel(r"Strength ($\alpha$)", fontsize=10)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Flip Rate", fontsize=11)

    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Response Flip Rate by Layer and Perturbation Strength", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "flip_rate_vs_pressure_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_critical_pressure_by_paradigm(model_summary_df: pd.DataFrame, out_dir: Path):
    """Fig 3: Critical pressure (50% accuracy threshold) by paradigm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = model_summary_df[model_summary_df["critical_pressure_50"] < float("inf")].copy()
    if plot_data.empty:
        # If no model drops below 50%, use critical_pressure_80
        plot_data = model_summary_df[model_summary_df["critical_pressure_80"] < float("inf")].copy()
        metric_col = "critical_pressure_80"
        title_suffix = "(80% threshold)"
    else:
        metric_col = "critical_pressure_50"
        title_suffix = "(50% threshold)"

    if not plot_data.empty:
        paradigm_order = ["transformer", "ssm", "hybrid", "rwkv", "reasoning"]
        paradigm_order = [p for p in paradigm_order if p in plot_data["paradigm"].values]

        sns.barplot(data=plot_data, x="paradigm", y=metric_col,
                    order=paradigm_order,
                    hue="paradigm", palette=PARADIGM_COLORS, legend=False, ax=ax)

        # Overlay individual model points
        for _, row in plot_data.iterrows():
            pidx = paradigm_order.index(row["paradigm"]) if row["paradigm"] in paradigm_order else 0
            ax.scatter(pidx, row[metric_col], color="black", zorder=5, s=40, alpha=0.7)
            ax.annotate(row["model"], (pidx, row[metric_col]),
                       fontsize=6, ha="center", va="bottom", rotation=15)

    ax.set_xlabel("Paradigm", fontsize=12)
    ax.set_ylabel(f"Critical Pressure {title_suffix}", fontsize=12)
    ax.set_title("Critical Perturbation Strength Before Accuracy Collapse", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "critical_pressure_by_paradigm.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_entropy_margin(summary_df: pd.DataFrame, out_dir: Path):
    """Fig 4: Entropy and margin vs pressure, colored by paradigm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Average over layers and domains
    model_curves = summary_df.groupby(["model", "paradigm", "strength"]).agg(
        mean_entropy=("mean_entropy", "mean"),
        mean_margin=("mean_margin", "mean"),
    ).reset_index()

    for model_name in model_curves["model"].unique():
        msub = model_curves[model_curves["model"] == model_name].sort_values("strength")
        paradigm = msub["paradigm"].iloc[0]
        color = PARADIGM_COLORS.get(paradigm, "#666")

        # Entropy
        if msub["mean_entropy"].notna().any():
            axes[0].plot(msub["strength"], msub["mean_entropy"], "o-",
                        color=color, linewidth=1.5, markersize=4,
                        label=model_name, alpha=0.8)
        # Margin
        if msub["mean_margin"].notna().any():
            axes[1].plot(msub["strength"], msub["mean_margin"], "o-",
                        color=color, linewidth=1.5, markersize=4,
                        label=model_name, alpha=0.8)

    axes[0].set_xlabel(r"Strength ($\alpha$)"); axes[0].set_ylabel("First-Token Entropy")
    axes[0].set_title("Prediction Entropy Under Pressure"); axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=7)

    axes[1].set_xlabel(r"Strength ($\alpha$)"); axes[1].set_ylabel("Top-1 Margin")
    axes[1].set_title("Logit Confidence Margin Under Pressure"); axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)

    fig.suptitle("Prediction Certainty Under Jitter Pressure", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_and_margin_vs_pressure.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_sensitivity_heatmap(summary_df: pd.DataFrame, out_dir: Path):
    """Fig 5: Heatmap of flip rate (model × layer) at a fixed moderate strength."""
    # Pick a moderate strength
    strengths = sorted(summary_df["strength"].unique())
    mid_strength = strengths[len(strengths) // 2] if strengths else 0.05

    sub = summary_df[summary_df["strength"] == mid_strength]
    pivot = sub.groupby(["model", "layer_idx"]).agg(
        flip_rate=("flip_rate", "mean")
    ).reset_index()

    if pivot.empty:
        return

    heatmap_data = pivot.pivot(index="model", columns="layer_idx", values="flip_rate")

    fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_data) * 0.6)))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(f"Layer Sensitivity Heatmap (alpha={mid_strength})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "layer_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Jitter-Pressure Inference Stability Analysis")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model IDs to test")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help="Override perturbation strengths")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override noise seeds")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--skip-models", nargs="+", default=None,
                        help="Model short names to skip (for resuming)")
    args = parser.parse_args()

    strengths = args.strengths or STRENGTHS
    seeds = args.seeds or SEEDS

    test_models = JITTER_MODELS
    if args.models:
        test_models = [(m, m.split("/")[-1], "custom") for m in args.models]

    skip_set = set(args.skip_models) if args.skip_models else set()

    # Output directories
    out_dir = Path("analysis/jitter_pressure")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "activation_cache").mkdir(exist_ok=True)

    all_raw_rows = []
    all_summaries = []

    # Resume support: load existing data for skipped models
    existing_csv = out_dir / "tables" / "raw_trials.csv"
    if skip_set and existing_csv.exists():
        existing_df = pd.read_csv(existing_csv)
        # Keep data for models that will be skipped (i.e., already completed)
        kept = existing_df[existing_df["model"].isin(skip_set)]
        if not kept.empty:
            all_raw_rows = kept.to_dict("records")
            print(f"  [RESUME] Loaded {len(all_raw_rows)} existing rows for {kept['model'].unique()}")

    print(f"{'=' * 70}")
    print(f"JITTER-PRESSURE INFERENCE STABILITY (JPIS)")
    print(f"Models: {len(test_models)}, Strengths: {strengths}")
    print(f"Seeds: {seeds}, Domains: {list(EVAL_PROMPTS.keys())}")
    print(f"{'=' * 70}\n")

    for model_id, model_name, paradigm in test_models:
        if model_name in skip_set:
            print(f"  [SKIP] {model_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name} ({paradigm})")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_id}: {e}")
            continue
        print(f"  Loaded in {time.time() - t0:.1f}s")

        layers = get_model_layers(model)
        if not layers:
            print(f"  [ERROR] Could not find model layers for {model_name}")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        n_layers = len(layers)
        layer_indices = get_layer_indices(n_layers)
        print(f"  Layers: {n_layers} total, testing at indices {layer_indices}")

        # ── Baseline ──────────────────────────────────────────────────
        print(f"  Running baseline...")
        t1 = time.time()
        baseline = run_baseline(model, tokenizer, EVAL_PROMPTS)
        baseline_time = time.time() - t1

        # Report baseline accuracy
        for domain in EVAL_PROMPTS:
            acc = sum(r["correct"] for r in baseline[domain]) / len(baseline[domain])
            print(f"    {domain}: {acc:.3f} ({sum(r['correct'] for r in baseline[domain])}/{len(baseline[domain])})")
        print(f"    Baseline took {baseline_time:.1f}s")

        # Cache baseline stats
        baseline_stats = {}
        for domain in EVAL_PROMPTS:
            entropies = [r["entropy"] for r in baseline[domain] if r["entropy"] is not None]
            margins = [r["margin"] for r in baseline[domain] if r["margin"] is not None]
            baseline_stats[domain] = {
                "accuracy": sum(r["correct"] for r in baseline[domain]) / len(baseline[domain]),
                "mean_entropy": np.mean(entropies) if entropies else None,
                "mean_margin": np.mean(margins) if margins else None,
            }

        np.savez_compressed(
            out_dir / "activation_cache" / f"{model_name.replace('/', '_')}_baseline.npz",
            **{f"{d}_stats": json.dumps(baseline_stats[d]) for d in baseline_stats}
        )

        # ── Perturbed runs ────────────────────────────────────────────
        n_configs = len(layer_indices) * len(strengths) * len(seeds)
        print(f"  Running {n_configs} perturbation configs...")

        config_idx = 0
        for layer_idx in layer_indices:
            for strength in strengths:
                for seed in seeds:
                    config_idx += 1
                    if config_idx % 10 == 0 or config_idx == 1:
                        print(f"    [{config_idx}/{n_configs}] layer={layer_idx}, a={strength}, seed={seed}")

                    rows = run_perturbed(
                        model, tokenizer, layers, layer_idx,
                        strength, seed, EVAL_PROMPTS, baseline,
                    )
                    for r in rows:
                        r["model"] = model_name
                        r["paradigm"] = paradigm
                        r["n_layers"] = n_layers
                        r["layer_frac"] = layer_idx / max(n_layers - 1, 1)
                    all_raw_rows.extend(rows)

        # Compute model-level summaries
        model_df = pd.DataFrame([r for r in all_raw_rows if r["model"] == model_name])
        if not model_df.empty:
            model_summary = compute_summaries(model_df, model_name, paradigm)
            all_summaries.append(model_summary)

        # Cleanup
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  {model_name} complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Save intermediate results after each model
        raw_df = pd.DataFrame(all_raw_rows)
        raw_df.to_csv(out_dir / "tables" / "raw_trials.csv", index=False)

    # ── Aggregate results ─────────────────────────────────────────────────
    if not all_raw_rows:
        print("\n[ERROR] No results collected. Exiting.")
        return

    raw_df = pd.DataFrame(all_raw_rows)
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    # Save raw and summary tables
    raw_df.to_csv(out_dir / "tables" / "raw_trials.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(out_dir / "tables" / "layer_curve_summary.csv", index=False)

    # ── Model-level summary with critical pressure & sensitivity ──────────
    model_rows = []
    for model_name in raw_df["model"].unique():
        mdf = raw_df[raw_df["model"] == model_name]
        paradigm = mdf["paradigm"].iloc[0]

        for domain in ["math", "factual"]:
            ddf = mdf[mdf["domain"] == domain]
            if ddf.empty:
                continue

            # Accuracy curve averaged over layers and seeds
            acc_curve = ddf.groupby("strength")["perturbed_correct"].mean()
            sorted_strengths = np.array(sorted(acc_curve.index))
            sorted_accs = np.array([acc_curve[s] for s in sorted_strengths])

            cp50 = critical_pressure(sorted_strengths, sorted_accs, 0.50)
            cp80 = critical_pressure(sorted_strengths, sorted_accs, 0.80)

            # Flip curve
            flip_curve = ddf.groupby("strength")["flip"].mean()
            sorted_flips = np.array([flip_curve[s] for s in sorted_strengths])

            # Sensitivity slope
            slope_acc = sensitivity_slope(sorted_strengths, sorted_accs)
            slope_flip = sensitivity_slope(sorted_strengths, sorted_flips)

            # Pressure AUC (1 - normalized area under accuracy curve)
            if len(sorted_strengths) > 1:
                auc = np.trapz(sorted_accs, sorted_strengths) / (sorted_strengths[-1] - sorted_strengths[0])
                pressure_auc = 1.0 - auc
            else:
                pressure_auc = 0.0

            # Bootstrap CI on mean accuracy at max strength
            max_str_accs = ddf[ddf["strength"] == sorted_strengths[-1]]["perturbed_correct"].values.astype(float)
            mean_acc, ci_lo, ci_hi = bootstrap_ci(max_str_accs)

            model_rows.append({
                "model": model_name,
                "paradigm": paradigm,
                "domain": domain,
                "critical_pressure_50": cp50,
                "critical_pressure_80": cp80,
                "sensitivity_slope_acc": slope_acc,
                "sensitivity_slope_flip": slope_flip,
                "pressure_auc": pressure_auc,
                "max_strength_accuracy": mean_acc,
                "max_strength_acc_ci_lo": ci_lo,
                "max_strength_acc_ci_hi": ci_hi,
            })

    model_summary_df = pd.DataFrame(model_rows)
    model_summary_df.to_csv(out_dir / "tables" / "model_summary.csv", index=False)

    # ── Cross-paradigm stats ──────────────────────────────────────────────
    stats_rows = []

    # Kruskal-Wallis: does paradigm predict pressure sensitivity?
    paradigm_groups = []
    paradigm_labels = []
    for paradigm in model_summary_df["paradigm"].unique():
        vals = model_summary_df[model_summary_df["paradigm"] == paradigm]["pressure_auc"].values
        if len(vals) > 0:
            paradigm_groups.append(vals)
            paradigm_labels.append(paradigm)

    if len(paradigm_groups) >= 2 and all(len(g) > 0 for g in paradigm_groups):
        try:
            h_stat, p_val = sp_stats.kruskal(*paradigm_groups)
            stats_rows.append({
                "test": "kruskal_wallis_paradigm_pressure_auc",
                "statistic": h_stat, "p_value": p_val,
                "groups": str(paradigm_labels),
                "n": sum(len(g) for g in paradigm_groups),
            })
        except Exception:
            pass

    # Wilcoxon: math vs factual critical pressure (paired by model)
    math_cp = model_summary_df[model_summary_df["domain"] == "math"].set_index("model")["pressure_auc"]
    fact_cp = model_summary_df[model_summary_df["domain"] == "factual"].set_index("model")["pressure_auc"]
    common = math_cp.index.intersection(fact_cp.index)
    if len(common) >= 5:
        try:
            stat, p = sp_stats.wilcoxon(math_cp[common].values, fact_cp[common].values)
            stats_rows.append({
                "test": "wilcoxon_math_vs_factual_pressure_auc",
                "statistic": stat, "p_value": p,
                "groups": "math vs factual", "n": len(common),
            })
        except Exception:
            pass

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df.to_csv(out_dir / "tables" / "stats_report.csv", index=False)

    # ── Visualizations ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Generating visualizations...")
    print(f"{'=' * 60}")

    fig_dir = out_dir / "figures"
    if not summary_df.empty:
        plot_accuracy_vs_pressure(summary_df, fig_dir)
        print("  -> accuracy_vs_pressure_by_model.png")

        plot_flip_rate(summary_df, fig_dir)
        print("  -> flip_rate_vs_pressure_by_layer.png")

        plot_entropy_margin(summary_df, fig_dir)
        print("  -> entropy_and_margin_vs_pressure.png")

        plot_layer_sensitivity_heatmap(summary_df, fig_dir)
        print("  -> layer_sensitivity_heatmap.png")

    if not model_summary_df.empty:
        plot_critical_pressure_by_paradigm(model_summary_df, fig_dir)
        print("  -> critical_pressure_by_paradigm.png")

    # ── Key findings ──────────────────────────────────────────────────────
    findings = []
    findings.append(f"models_tested: {len(raw_df['model'].unique())}")
    findings.append(f"paradigms: {', '.join(raw_df['paradigm'].unique())}")
    findings.append(f"total_trials: {len(raw_df)}")
    findings.append(f"strengths: {sorted(raw_df['strength'].unique())}")

    if not model_summary_df.empty:
        # Most robust model
        most_robust = model_summary_df.loc[model_summary_df["pressure_auc"].idxmin()]
        findings.append(f"most_robust_model: {most_robust['model']} (pressure_auc={most_robust['pressure_auc']:.4f})")

        # Least robust model
        least_robust = model_summary_df.loc[model_summary_df["pressure_auc"].idxmax()]
        findings.append(f"least_robust_model: {least_robust['model']} (pressure_auc={least_robust['pressure_auc']:.4f})")

        # Paradigm means
        paradigm_means = model_summary_df.groupby("paradigm")["pressure_auc"].mean()
        for p, v in paradigm_means.items():
            findings.append(f"paradigm_mean_pressure_auc_{p}: {v:.4f}")

        # Domain comparison
        domain_means = model_summary_df.groupby("domain")["pressure_auc"].mean()
        for d, v in domain_means.items():
            findings.append(f"domain_mean_pressure_auc_{d}: {v:.4f}")

    # Stats
    if not stats_df.empty:
        for _, row in stats_df.iterrows():
            findings.append(f"{row['test']}: stat={row['statistic']:.4f}, p={row['p_value']:.6f}, n={row['n']}")

    findings.append("business_implication: Pressure sensitivity reveals deployment risk — models appearing equivalent on benchmarks may have vastly different robustness to inference-time noise (quantization drift, temperature, stochastic decoding)")
    findings.append("scientific_implication: Dynamic stability under perturbation is a new axis orthogonal to representation geometry — 'compressed but stable' vs 'compact but hypersensitive' cannot be predicted from PR alone")

    (out_dir / "key_findings.txt").write_text("\n".join(findings), encoding="utf-8")

    # ── Final report ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("JPIS ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output: {out_dir}")
    print(f"Raw trials: {len(raw_df)} rows")
    print(f"Models: {len(raw_df['model'].unique())}")
    print(f"\nKey findings:")
    for f in findings:
        print(f"  {f}")
    print(f"\nTables: {list((out_dir / 'tables').glob('*.csv'))}")
    print(f"Figures: {list((out_dir / 'figures').glob('*.png'))}")


if __name__ == "__main__":
    main()
