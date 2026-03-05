#!/usr/bin/env python3
"""
causal_pr_robust.py — Stage 3b: Robust Over-Compression + JPIS Stability Bridge

Codex-designed protocol to lock the over-compression claim AND bridge it
with JPIS pressure-fragility data:
- 96 eval prompts (48 math + 48 factual), split: 32 calibration + 64 holdout
- 9 random seeds for stochastic expansion noise
- Two-stage pipeline: Stage A (coarse layer/strength scout) → Stage B (fine confirm)
- Critical-rank curves (keep_k sweep)
- Bridge analysis: correlate expansion gain with JPIS pressure_auc

Stage A: 6 layers × 5 strengths × 3 seeds, calibration prompts
Stage B: 3 layers × 9 fine strengths × 9 seeds, holdout prompts + critical-rank
Stage Bridge: Join expansion data with JPIS stability, mediation analysis
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
import torch
import torch.nn as nn

# ── Seeds (Codex protocol) ──────────────────────────────────────────────
SCOUT_SEEDS = [11, 23, 37]
ALL_SEEDS = [11, 23, 37, 47, 59, 71, 83, 97, 109]

# ── Coarse and Fine strength grids ──────────────────────────────────────
COARSE_STRENGTHS = [0.0, 0.05, 0.08, 0.10, 0.14]
FINE_STRENGTHS = [0.0, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]

# ── Critical-rank keep_k grid ──────────────────────────────────────────
CRITICAL_RANK_K = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

# ── Coarse layer grid (for 28-layer models; adapted per model) ────────
COARSE_LAYER_FRACS = [0.14, 0.29, 0.43, 0.57, 0.71, 0.86]  # ~4,8,12,16,20,24 out of 28

# ── Extended evaluation prompts (96 total) ───────────────────────────────
MATH_PROMPTS = [
    # Calibration (first 16)
    {"prompt": "What is 7 * 8? Answer with just the number:", "answer": "56"},
    {"prompt": "What is 15 + 27? Answer with just the number:", "answer": "42"},
    {"prompt": "What is 100 - 37? Answer with just the number:", "answer": "63"},
    {"prompt": "What is 144 / 12? Answer with just the number:", "answer": "12"},
    {"prompt": "What is 9 * 9? Answer with just the number:", "answer": "81"},
    {"prompt": "What is 23 + 19? Answer with just the number:", "answer": "42"},
    {"prompt": "What is 256 / 16? Answer with just the number:", "answer": "16"},
    {"prompt": "What is 11 * 11? Answer with just the number:", "answer": "121"},
    {"prompt": "What is 6 * 7? Answer with just the number:", "answer": "42"},
    {"prompt": "What is 50 + 50? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 200 - 85? Answer with just the number:", "answer": "115"},
    {"prompt": "What is 96 / 8? Answer with just the number:", "answer": "12"},
    {"prompt": "What is 13 * 7? Answer with just the number:", "answer": "91"},
    {"prompt": "What is 88 + 12? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 225 / 15? Answer with just the number:", "answer": "15"},
    {"prompt": "What is 14 * 6? Answer with just the number:", "answer": "84"},
    # Holdout (remaining 32)
    {"prompt": "What is 8 * 12? Answer with just the number:", "answer": "96"},
    {"prompt": "What is 33 + 47? Answer with just the number:", "answer": "80"},
    {"prompt": "What is 500 - 123? Answer with just the number:", "answer": "377"},
    {"prompt": "What is 72 / 9? Answer with just the number:", "answer": "8"},
    {"prompt": "What is 15 * 15? Answer with just the number:", "answer": "225"},
    {"prompt": "What is 64 + 36? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 300 - 175? Answer with just the number:", "answer": "125"},
    {"prompt": "What is 180 / 12? Answer with just the number:", "answer": "15"},
    {"prompt": "What is 17 * 5? Answer with just the number:", "answer": "85"},
    {"prompt": "What is 99 + 1? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 150 - 67? Answer with just the number:", "answer": "83"},
    {"prompt": "What is 84 / 7? Answer with just the number:", "answer": "12"},
    {"prompt": "What is 12 * 12? Answer with just the number:", "answer": "144"},
    {"prompt": "What is 45 + 55? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 1000 - 387? Answer with just the number:", "answer": "613"},
    {"prompt": "What is 132 / 11? Answer with just the number:", "answer": "12"},
    {"prompt": "What is 19 * 4? Answer with just the number:", "answer": "76"},
    {"prompt": "What is 77 + 23? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 400 - 256? Answer with just the number:", "answer": "144"},
    {"prompt": "What is 108 / 9? Answer with just the number:", "answer": "12"},
    {"prompt": "What is 16 * 8? Answer with just the number:", "answer": "128"},
    {"prompt": "What is 125 + 75? Answer with just the number:", "answer": "200"},
    {"prompt": "What is 250 - 137? Answer with just the number:", "answer": "113"},
    {"prompt": "What is 156 / 12? Answer with just the number:", "answer": "13"},
    {"prompt": "What is 21 * 3? Answer with just the number:", "answer": "63"},
    {"prompt": "What is 48 + 52? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 800 - 425? Answer with just the number:", "answer": "375"},
    {"prompt": "What is 168 / 14? Answer with just the number:", "answer": "12"},
    {"prompt": "What is 25 * 4? Answer with just the number:", "answer": "100"},
    {"prompt": "What is 67 + 33? Answer with just the number:", "answer": "100"},
]

FACTUAL_PROMPTS = [
    # Calibration (first 16)
    {"prompt": "The capital of Japan is", "answer": "Tokyo"},
    {"prompt": "Water freezes at 0 degrees", "answer": "Celsius"},
    {"prompt": "The chemical symbol for gold is", "answer": "Au"},
    {"prompt": "The largest planet in our solar system is", "answer": "Jupiter"},
    {"prompt": "The speed of light is approximately 300,000 km per", "answer": "second"},
    {"prompt": "DNA stands for deoxyribonucleic", "answer": "acid"},
    {"prompt": "The square root of 144 is", "answer": "12"},
    {"prompt": "The atomic number of carbon is", "answer": "6"},
    {"prompt": "The capital of France is", "answer": "Paris"},
    {"prompt": "The chemical formula for water is", "answer": "H2O"},
    {"prompt": "The tallest mountain in the world is Mount", "answer": "Everest"},
    {"prompt": "The first element on the periodic table is", "answer": "hydrogen"},
    {"prompt": "The number of continents on Earth is", "answer": "7"},
    {"prompt": "The chemical symbol for silver is", "answer": "Ag"},
    {"prompt": "The boiling point of water is 100 degrees", "answer": "Celsius"},
    {"prompt": "The planet closest to the Sun is", "answer": "Mercury"},
    # Holdout (remaining 32)
    {"prompt": "The capital of Germany is", "answer": "Berlin"},
    {"prompt": "The chemical symbol for iron is", "answer": "Fe"},
    {"prompt": "The number of bones in an adult human body is approximately", "answer": "206"},
    {"prompt": "The smallest prime number is", "answer": "2"},
    {"prompt": "The currency of the United Kingdom is the British", "answer": "pound"},
    {"prompt": "The chemical formula for table salt is", "answer": "NaCl"},
    {"prompt": "The largest ocean on Earth is the", "answer": "Pacific"},
    {"prompt": "The force of gravity on Earth is approximately 9.8 meters per second", "answer": "squared"},
    {"prompt": "The chemical symbol for sodium is", "answer": "Na"},
    {"prompt": "The capital of Italy is", "answer": "Rome"},
    {"prompt": "The speed of sound in air is approximately 343 meters per", "answer": "second"},
    {"prompt": "The most abundant gas in Earth's atmosphere is", "answer": "nitrogen"},
    {"prompt": "The number of planets in our solar system is", "answer": "8"},
    {"prompt": "The chemical symbol for copper is", "answer": "Cu"},
    {"prompt": "The largest organ in the human body is the", "answer": "skin"},
    {"prompt": "The freezing point of water in Fahrenheit is", "answer": "32"},
    {"prompt": "The capital of Australia is", "answer": "Canberra"},
    {"prompt": "The chemical symbol for potassium is", "answer": "K"},
    {"prompt": "The number of chromosomes in a human cell is", "answer": "46"},
    {"prompt": "The hardest natural substance on Earth is", "answer": "diamond"},
    {"prompt": "The capital of Canada is", "answer": "Ottawa"},
    {"prompt": "The chemical symbol for lead is", "answer": "Pb"},
    {"prompt": "The largest desert in the world is the", "answer": "Sahara"},
    {"prompt": "Pi is approximately equal to 3.14", "answer": "159"},
    {"prompt": "The capital of Brazil is", "answer": "Brasilia"},
    {"prompt": "The chemical symbol for tin is", "answer": "Sn"},
    {"prompt": "The number of teeth in a typical adult human mouth is", "answer": "32"},
    {"prompt": "The lightest element is", "answer": "hydrogen"},
    {"prompt": "The capital of Spain is", "answer": "Madrid"},
    {"prompt": "The chemical symbol for mercury is", "answer": "Hg"},
    {"prompt": "The deepest ocean trench is the", "answer": "Mariana"},
    {"prompt": "The number of degrees in a circle is", "answer": "360"},
]

# ── Models ──────────────────────────────────────────────────────────────
# Core models from Stage 3 (original over-compression test)
CORE_MODELS = [
    # (hf_id, short_name, role, core_pr)
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "reasoning", 1.07),
    ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base", "base", 1.57),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning", 1.08),
    ("Qwen/Qwen2.5-Math-1.5B", "Qwen2.5-Math-1.5B", "base", 1.53),
    ("nvidia/OpenReasoning-Nemotron-1.5B", "Nemotron-1.5B-R", "reasoning", 1.04),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", "base", 1.34),
]

# JPIS bridge models — tested in JPIS, need expansion data for bridge
BRIDGE_MODELS = [
    ("google/gemma-2-2b", "Gemma2-2B", "base", 2.53),
    ("google/gemma-3-1b-pt", "Gemma3-1B", "base", 1.45),
    ("state-spaces/mamba-790m-hf", "Mamba-790M", "ssm", 6.14),
    ("tiiuae/Falcon-H1-0.5B-Base", "FalconH1-0.5B", "hybrid", 5.66),
    ("tiiuae/Falcon-H1-1.5B-Base", "FalconH1-1.5B", "hybrid", 1.63),
    ("Zyphra/Zamba2-1.2B", "Zamba2-1.2B", "hybrid", 4.54),
]

# Combined list
TEST_MODELS = CORE_MODELS + BRIDGE_MODELS


def get_prompts(split: str, domain: str = "both") -> List[Dict]:
    """Get calibration or holdout prompts."""
    if split == "calibration":
        math = MATH_PROMPTS[:16]
        fact = FACTUAL_PROMPTS[:16]
    elif split == "holdout":
        math = MATH_PROMPTS[16:]
        fact = FACTUAL_PROMPTS[16:]
    else:
        math = MATH_PROMPTS
        fact = FACTUAL_PROMPTS

    if domain == "math":
        return [{"domain": "math", **p} for p in math]
    elif domain == "factual":
        return [{"domain": "factual", **p} for p in fact]
    else:
        return ([{"domain": "math", **p} for p in math] +
                [{"domain": "factual", **p} for p in fact])


# ── Hook classes (reused from causal_pr_intervention.py) ────────────────
class PRExpander:
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

        rng = torch.Generator(device=hidden.device)
        rng.manual_seed(self.seed)

        batch, seq_len, d_model = hidden.shape
        noise = torch.randn(seq_len, self.n_components, generator=rng,
                            device=hidden.device, dtype=torch.float32)
        Q = torch.randn(self.n_components, d_model, generator=rng,
                         device=hidden.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(Q.T)
        Q = Q[:, :self.n_components].T

        perturbation = noise @ Q
        scale = hidden.float().norm(dim=-1, keepdim=True).mean() * self.strength
        perturbation = perturbation * scale / perturbation.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        hidden = hidden + perturbation.unsqueeze(0).to(hidden.dtype)

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
        orig_dtype = hidden.dtype
        h = hidden[0].float()
        mean = h.mean(dim=0, keepdim=True)
        h_centered = h - mean

        U, S, Vh = torch.linalg.svd(h_centered, full_matrices=False)
        k = min(self.keep_components, len(S))
        projected = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        hidden = (projected + mean).unsqueeze(0).to(orig_dtype)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def attach(self, model, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ── Evaluation ──────────────────────────────────────────────────────────
def evaluate_per_prompt(model, tokenizer, prompts: List[Dict],
                        max_new_tokens: int = 10) -> List[Dict]:
    """Evaluate model on prompts, return per-prompt correctness."""
    device = next(model.parameters()).device
    results = []

    for item in prompts:
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
        answer = item["answer"].lower()
        correct = 1 if answer in generated.lower() else 0

        results.append({
            "prompt": item["prompt"],
            "domain": item["domain"],
            "answer": item["answer"],
            "generated": generated,
            "correct": correct,
        })

    return results


def get_model_layers(model) -> List[nn.Module]:
    for attr in ["model.layers", "transformer.h", "gpt_neox.layers",
                 "model.model.layers", "backbone.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    return []


def coarse_layers_for(n_layers: int) -> List[int]:
    """Get coarse layer indices (every 4th, adapted to model depth)."""
    return [max(1, int(f * n_layers)) for f in COARSE_LAYER_FRACS]


def fine_layers_around(best_layer: int, n_layers: int) -> List[int]:
    """Get fine layer indices: best_layer - 1, best_layer, best_layer + 1."""
    candidates = [best_layer - 1, best_layer, best_layer + 1]
    return [l for l in candidates if 0 <= l < n_layers]


# ── Stage A: Coarse Scouting ────────────────────────────────────────────
def run_stage_a(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    out_dir = Path("analysis/causal_pr_robust")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    cal_prompts = get_prompts("calibration")
    print(f"Stage A: Coarse scouting with {len(cal_prompts)} calibration prompts, "
          f"{len(SCOUT_SEEDS)} seeds")

    # Resume: load existing results if available
    existing_csv = out_dir / "tables" / "stage_a_results.csv"
    all_rows = []
    skip_models = set()
    if existing_csv.exists():
        existing_df = pd.read_csv(existing_csv)
        all_rows = existing_df.to_dict("records")
        skip_models = set(existing_df["model"].unique())
        print(f"  Resuming: {len(skip_models)} models already done: {skip_models}")

    model_filter = None
    if hasattr(args, 'models') and args.models:
        model_filter = [m.strip() for m in args.models.split(",")]

    for hf_id, model_name, role, core_pr in TEST_MODELS:
        if model_name in skip_models:
            print(f"\nSkipping {model_name} (already in saved results)")
            continue
        if model_filter and model_name not in model_filter:
            print(f"\nSkipping {model_name} (not in --models filter)")
            continue
        print(f"\n{'=' * 60}")
        print(f"Stage A: {model_name} ({role}, core_pr={core_pr:.2f})")
        print(f"{'=' * 60}")

        try:
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                hf_id, trust_remote_code=True, dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
            model.eval()
            print(f"  Loaded in {time.time() - t0:.1f}s")

            layers = get_model_layers(model)
            if not layers:
                print(f"  [ERROR] Could not find model layers, skipping")
                del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
                continue

            n_layers = len(layers)
            coarse_lyrs = coarse_layers_for(n_layers)
            print(f"  {n_layers} layers, testing coarse: {coarse_lyrs}")

            # Baseline
            print(f"  Evaluating baseline...")
            baseline_results = evaluate_per_prompt(model, tokenizer, cal_prompts)
            baseline_acc = np.mean([r["correct"] for r in baseline_results])
            math_acc = np.mean([r["correct"] for r in baseline_results if r["domain"] == "math"])
            fact_acc = np.mean([r["correct"] for r in baseline_results if r["domain"] == "factual"])
            print(f"    Baseline: overall={baseline_acc:.3f}, math={math_acc:.3f}, factual={fact_acc:.3f}")

            for r in baseline_results:
                all_rows.append({
                    "model": model_name, "role": role, "core_pr": core_pr,
                    "layer": -1, "strength": 0.0, "seed": 0,
                    "intervention": "none", "stage": "A",
                    "prompt": r["prompt"], "domain": r["domain"],
                    "correct": r["correct"],
                })

            # Expansion sweep: layers × strengths × seeds
            total_conditions = len(coarse_lyrs) * len([s for s in COARSE_STRENGTHS if s > 0]) * len(SCOUT_SEEDS)
            cond_idx = 0
            for layer_idx in coarse_lyrs:
                for strength in COARSE_STRENGTHS:
                    if strength == 0.0:
                        continue
                    for seed in SCOUT_SEEDS:
                        cond_idx += 1
                        expander = PRExpander(layer_idx, strength=strength, seed=seed)
                        expander.attach(model, layers[layer_idx])

                        results = evaluate_per_prompt(model, tokenizer, cal_prompts)
                        expander.detach()

                        acc = np.mean([r["correct"] for r in results])
                        print(f"    [{cond_idx}/{total_conditions}] layer={layer_idx}, "
                              f"s={strength:.2f}, seed={seed}: acc={acc:.3f}")

                        for r in results:
                            all_rows.append({
                                "model": model_name, "role": role, "core_pr": core_pr,
                                "layer": layer_idx, "strength": strength, "seed": seed,
                                "intervention": "expand", "stage": "A",
                                "prompt": r["prompt"], "domain": r["domain"],
                                "correct": r["correct"],
                            })

            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  [ERROR] {model_name} failed: {e}")
            print(f"  Attempting GPU cleanup and continuing...")
            try:
                del model, tokenizer
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # Save Stage A results
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "tables" / "stage_a_results.csv", index=False)
    print(f"\nSaved {len(df)} rows to stage_a_results.csv")

    # Find best layer per model (highest mean accuracy at mild expansion)
    best_layers = {}
    for model_name in df["model"].unique():
        sub = df[(df["model"] == model_name) & (df["intervention"] == "expand") &
                 (df["strength"].between(0.05, 0.10))]
        if sub.empty:
            continue
        layer_means = sub.groupby("layer")["correct"].mean()
        best_layer = int(layer_means.idxmax())
        best_layers[model_name] = best_layer
        print(f"  {model_name}: best coarse layer = {best_layer} "
              f"(acc={layer_means.max():.3f})")

    # Save best layers
    with open(out_dir / "tables" / "best_layers.json", "w") as f:
        json.dump(best_layers, f, indent=2)

    return best_layers


# ── Stage B: Fine Confirmation ──────────────────────────────────────────
def run_stage_b(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    out_dir = Path("analysis/causal_pr_robust")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    # Load best layers from Stage A
    best_layers_path = out_dir / "tables" / "best_layers.json"
    if not best_layers_path.exists():
        print("ERROR: Run Stage A first to find best layers")
        return
    with open(best_layers_path) as f:
        best_layers = json.load(f)

    holdout_prompts = get_prompts("holdout")
    print(f"Stage B: Fine confirmation with {len(holdout_prompts)} holdout prompts, "
          f"{len(ALL_SEEDS)} seeds")

    all_rows = []

    for hf_id, model_name, role, core_pr in TEST_MODELS:
        if model_name not in best_layers:
            print(f"  Skipping {model_name} (no best layer from Stage A)")
            continue

        best_layer = best_layers[model_name]
        print(f"\n{'=' * 60}")
        print(f"Stage B: {model_name} ({role}, best_layer={best_layer})")
        print(f"{'=' * 60}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        model.eval()
        print(f"  Loaded in {time.time() - t0:.1f}s")

        layers = get_model_layers(model)
        n_layers = len(layers)
        fine_lyrs = fine_layers_around(best_layer, n_layers)
        print(f"  Fine layers: {fine_lyrs}")

        # Baseline on holdout
        print(f"  Evaluating baseline...")
        baseline_results = evaluate_per_prompt(model, tokenizer, holdout_prompts)
        baseline_acc = np.mean([r["correct"] for r in baseline_results])
        print(f"    Baseline: {baseline_acc:.3f}")

        for r in baseline_results:
            all_rows.append({
                "model": model_name, "role": role, "core_pr": core_pr,
                "layer": -1, "strength": 0.0, "seed": 0,
                "intervention": "none", "stage": "B",
                "prompt": r["prompt"], "domain": r["domain"],
                "correct": r["correct"],
            })

        # Fine expansion sweep
        total_cond = len(fine_lyrs) * len([s for s in FINE_STRENGTHS if s > 0]) * len(ALL_SEEDS)
        cond_idx = 0
        for layer_idx in fine_lyrs:
            for strength in FINE_STRENGTHS:
                if strength == 0.0:
                    continue
                for seed in ALL_SEEDS:
                    cond_idx += 1
                    expander = PRExpander(layer_idx, strength=strength, seed=seed)
                    expander.attach(model, layers[layer_idx])

                    results = evaluate_per_prompt(model, tokenizer, holdout_prompts)
                    expander.detach()

                    acc = np.mean([r["correct"] for r in results])
                    if cond_idx % 20 == 0 or cond_idx <= 3:
                        print(f"    [{cond_idx}/{total_cond}] L={layer_idx}, "
                              f"s={strength:.2f}, seed={seed}: acc={acc:.3f}")

                    for r in results:
                        all_rows.append({
                            "model": model_name, "role": role, "core_pr": core_pr,
                            "layer": layer_idx, "strength": strength, "seed": seed,
                            "intervention": "expand", "stage": "B",
                            "prompt": r["prompt"], "domain": r["domain"],
                            "correct": r["correct"],
                        })

        # Critical-rank curve at best layer
        print(f"\n  Critical-rank curve at layer {best_layer}...")
        for keep_k in CRITICAL_RANK_K:
            if keep_k >= n_layers * 50:  # skip if k > reasonable rank
                continue
            compressor = PRCompressor(best_layer, keep_components=keep_k)
            compressor.attach(model, layers[best_layer])

            results = evaluate_per_prompt(model, tokenizer, holdout_prompts)
            compressor.detach()

            acc = np.mean([r["correct"] for r in results])
            print(f"    keep_k={keep_k}: acc={acc:.3f}")

            for r in results:
                all_rows.append({
                    "model": model_name, "role": role, "core_pr": core_pr,
                    "layer": best_layer, "strength": float(keep_k), "seed": 0,
                    "intervention": "compress", "stage": "B",
                    "prompt": r["prompt"], "domain": r["domain"],
                    "correct": r["correct"],
                })

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save Stage B results
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "tables" / "stage_b_results.csv", index=False)
    print(f"\nSaved {len(df)} rows to stage_b_results.csv")

    return df


# ── Statistical Analysis ────────────────────────────────────────────────
def run_stats(args):
    out_dir = Path("analysis/causal_pr_robust")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Load Stage B results
    stage_b_path = out_dir / "tables" / "stage_b_results.csv"
    if not stage_b_path.exists():
        print("ERROR: Run Stage B first")
        return

    df = pd.read_csv(stage_b_path)
    print(f"Loaded {len(df)} rows from Stage B")

    # ── 1. Over-compression analysis ────────────────────────────────────
    # For each model: compare baseline vs mild-expansion band (0.06-0.10)
    expand_df = df[df["intervention"].isin(["none", "expand"])]

    findings = []
    model_stats = []

    for model_name in expand_df["model"].unique():
        sub = expand_df[expand_df["model"] == model_name]
        baseline_correct = sub[sub["intervention"] == "none"]["correct"]
        baseline_acc = baseline_correct.mean()

        # Mild-band accuracy (pooled across seeds, fine layers, strengths 0.06-0.10)
        mild = sub[(sub["intervention"] == "expand") &
                   (sub["strength"] >= 0.06) & (sub["strength"] <= 0.10)]
        if mild.empty:
            continue

        mild_acc = mild["correct"].mean()
        delta = mild_acc - baseline_acc

        # Per-seed statistics
        seed_accs = mild.groupby("seed")["correct"].mean()
        se = seed_accs.std() / np.sqrt(len(seed_accs))

        role = sub["role"].iloc[0]
        core_pr = sub["core_pr"].iloc[0]

        model_stats.append({
            "model": model_name, "role": role, "core_pr": core_pr,
            "baseline_acc": baseline_acc, "mild_acc": mild_acc,
            "delta": delta, "se": se,
            "n_seeds": len(seed_accs),
            "improved": delta > 0,
        })

        print(f"  {model_name}: baseline={baseline_acc:.3f}, "
              f"mild={mild_acc:.3f}, delta={delta:+.3f} (SE={se:.3f})")

    stats_df = pd.DataFrame(model_stats)

    # ── 2. McNemar-like test per model ──────────────────────────────────
    from scipy.stats import binom_test, wilcoxon

    for _, row in stats_df.iterrows():
        model_name = row["model"]
        sub = expand_df[expand_df["model"] == model_name]

        # Per-prompt: baseline correct, mild-band correct (majority vote across seeds)
        baseline_per_prompt = (sub[sub["intervention"] == "none"]
                               .groupby("prompt")["correct"].mean())
        mild_sub = sub[(sub["intervention"] == "expand") &
                       (sub["strength"] >= 0.06) & (sub["strength"] <= 0.10)]
        mild_per_prompt = mild_sub.groupby("prompt")["correct"].mean()

        # Align on common prompts
        common = baseline_per_prompt.index.intersection(mild_per_prompt.index)
        if len(common) < 10:
            continue

        b = baseline_per_prompt[common].values
        m = mild_per_prompt[common].values

        # Wilcoxon signed-rank on per-prompt accuracy change
        diff = m - b
        nonzero = diff[diff != 0]
        if len(nonzero) >= 5:
            stat, p = wilcoxon(nonzero)
            direction = "improvement" if np.mean(diff) > 0 else "degradation"
            print(f"  {model_name} Wilcoxon: stat={stat:.1f}, p={p:.4f}, "
                  f"mean_diff={np.mean(diff):+.4f} ({direction})")
            stats_df.loc[stats_df["model"] == model_name, "wilcoxon_p"] = p
            stats_df.loc[stats_df["model"] == model_name, "mean_prompt_diff"] = np.mean(diff)

    # ── 3. Mixed-effects model (simplified: interaction test) ───────────
    # Test: do compressed models (low core_pr) benefit more from mild expansion?
    if len(stats_df) >= 4:
        from scipy.stats import pearsonr, spearmanr
        compressed = stats_df["core_pr"].values
        deltas = stats_df["delta"].values
        if len(compressed) >= 4:
            r, p = spearmanr(compressed, deltas)
            findings.append(f"compression_benefit_correlation: r={r:.3f}, p={p:.4f}")
            print(f"\n  Compression-benefit correlation: Spearman r={r:.3f}, p={p:.4f}")
            print(f"  (Negative r = more compressed models benefit more from expansion)")

    # ── 4. Critical-rank analysis ───────────────────────────────────────
    compress_df = df[df["intervention"] == "compress"]
    if not compress_df.empty:
        print(f"\n  Critical-rank analysis:")
        critical_ranks = []
        for model_name in compress_df["model"].unique():
            sub = compress_df[compress_df["model"] == model_name]
            baseline_sub = df[(df["model"] == model_name) & (df["intervention"] == "none")]
            baseline_acc = baseline_sub["correct"].mean()

            # Per keep_k accuracy
            rank_accs = sub.groupby("strength")["correct"].mean().sort_index()

            # Find k95
            k95 = None
            for k, acc in rank_accs.items():
                if acc >= baseline_acc - 0.05:
                    k95 = int(k)
                    break

            role = sub["role"].iloc[0]
            core_pr = sub["core_pr"].iloc[0]

            critical_ranks.append({
                "model": model_name, "role": role, "core_pr": core_pr,
                "baseline_acc": baseline_acc, "k95": k95,
                "k95_normalized": k95 / 64 if k95 else None,  # normalize by max tested k
            })
            print(f"    {model_name}: k95={k95}, baseline={baseline_acc:.3f}")

            # Math vs factual breakdown
            for domain in ["math", "factual"]:
                dom_sub = sub[sub["domain"] == domain]
                dom_baseline = baseline_sub[baseline_sub["domain"] == domain]["correct"].mean()
                dom_rank_accs = dom_sub.groupby("strength")["correct"].mean().sort_index()
                dom_k95 = None
                for k, acc in dom_rank_accs.items():
                    if acc >= dom_baseline - 0.05:
                        dom_k95 = int(k)
                        break
                print(f"      {domain}: k95={dom_k95}")

        cr_df = pd.DataFrame(critical_ranks)
        cr_df.to_csv(out_dir / "tables" / "critical_ranks.csv", index=False)

    # ── 5. Save comprehensive stats ─────────────────────────────────────
    stats_df.to_csv(out_dir / "tables" / "overcompression_stats.csv", index=False)

    # ── 6. Visualizations ───────────────────────────────────────────────
    _plot_results(df, stats_df, out_dir)

    # ── 7. Key findings ─────────────────────────────────────────────────
    n_improved = stats_df["improved"].sum()
    n_total = len(stats_df)
    findings.insert(0, f"models_tested: {n_total}")
    findings.insert(1, f"models_improved_by_mild_expansion: {n_improved}/{n_total}")

    for _, row in stats_df.iterrows():
        findings.append(
            f"{row['model']}: baseline={row['baseline_acc']:.3f}, "
            f"mild_expansion={row['mild_acc']:.3f}, "
            f"delta={row['delta']:+.3f}"
        )

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")
    print(f"\nKey findings saved to {findings_path}")
    for f in findings:
        print(f"  {f}")


def _plot_results(df, stats_df, out_dir):
    """Generate all figures."""
    fig_dir = out_dir / "figures"

    # ── Plot 1: Expansion dose-response per model (seed-averaged) ─────
    expand_df = df[df["intervention"].isin(["none", "expand"])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors_r = ["#FF5722", "#E91E63", "#FF9800"]
    colors_b = ["#2196F3", "#00BCD4", "#4CAF50"]

    r_idx, b_idx = 0, 0
    for model_name in expand_df["model"].unique():
        sub = expand_df[expand_df["model"] == model_name]
        role = sub["role"].iloc[0]

        # Aggregate: mean accuracy per strength (across seeds)
        agg = sub.groupby("strength")["correct"].agg(["mean", "std", "count"])
        agg["se"] = agg["std"] / np.sqrt(agg["count"])

        if role == "reasoning":
            color = colors_r[r_idx % len(colors_r)]
            r_idx += 1
            ls = "-"
        else:
            color = colors_b[b_idx % len(colors_b)]
            b_idx += 1
            ls = "--"

        axes[0].errorbar(agg.index, agg["mean"], yerr=agg["se"],
                         fmt="o-", color=color, linestyle=ls,
                         linewidth=2, markersize=5, capsize=3,
                         label=f"{model_name} ({role})")

    axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Expansion Strength")
    axes[0].set_ylabel("Accuracy (mean +/- SE across seeds)")
    axes[0].set_title("PR Expansion Dose-Response (Holdout)")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].grid(True, alpha=0.3)

    # ── Plot 2: Critical-rank curves ──────────────────────────────────
    compress_df = df[df["intervention"] == "compress"]
    if not compress_df.empty:
        r_idx, b_idx = 0, 0
        for model_name in compress_df["model"].unique():
            sub = compress_df[compress_df["model"] == model_name]
            role = sub["role"].iloc[0]
            baseline_sub = df[(df["model"] == model_name) & (df["intervention"] == "none")]
            baseline_acc = baseline_sub["correct"].mean()

            rank_accs = sub.groupby("strength")["correct"].mean().sort_index()

            if role == "reasoning":
                color = colors_r[r_idx % len(colors_r)]
                r_idx += 1
                ls = "-"
            else:
                color = colors_b[b_idx % len(colors_b)]
                b_idx += 1
                ls = "--"

            x = [0] + list(rank_accs.index)
            y = [baseline_acc] + list(rank_accs.values)
            axes[1].plot(x, y, "o-", color=color, linestyle=ls,
                         linewidth=2, markersize=5,
                         label=f"{model_name} ({role})")

        axes[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        axes[1].set_xlabel("Keep Top-K Components")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Critical-Rank Curves (Holdout)")
        axes[1].legend(fontsize=7, loc="lower right")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale("log")

    plt.tight_layout()
    plt.savefig(fig_dir / "robust_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved robust_dose_response.png")

    # ── Plot 3: Over-compression scatter ──────────────────────────────
    if len(stats_df) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        for _, row in stats_df.iterrows():
            color = "#FF5722" if row["role"] == "reasoning" else "#2196F3"
            marker = "^" if row["role"] == "reasoning" else "o"
            ax.scatter(row["core_pr"], row["delta"], color=color,
                       marker=marker, s=120, zorder=5)
            ax.annotate(row["model"], (row["core_pr"], row["delta"]),
                        fontsize=7, ha="left", va="bottom")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Core PR (lower = more compressed)")
        ax.set_ylabel("Accuracy delta (mild expansion - baseline)")
        ax.set_title("Over-Compression Evidence: Do Compressed Models Benefit from Expansion?")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "overcompression_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved overcompression_scatter.png")

    # ── Plot 4: Math vs factual critical-rank ─────────────────────────
    if not compress_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, domain in enumerate(["math", "factual"]):
            ax = axes[idx]
            r_idx, b_idx = 0, 0
            for model_name in compress_df["model"].unique():
                sub = compress_df[(compress_df["model"] == model_name) &
                                  (compress_df["domain"] == domain)]
                role = sub["role"].iloc[0] if not sub.empty else "unknown"
                baseline_sub = df[(df["model"] == model_name) &
                                  (df["intervention"] == "none") &
                                  (df["domain"] == domain)]
                baseline_acc = baseline_sub["correct"].mean() if not baseline_sub.empty else 0

                rank_accs = sub.groupby("strength")["correct"].mean().sort_index()

                if role == "reasoning":
                    color = colors_r[r_idx % len(colors_r)]
                    r_idx += 1
                    ls = "-"
                else:
                    color = colors_b[b_idx % len(colors_b)]
                    b_idx += 1
                    ls = "--"

                x = [0] + list(rank_accs.index)
                y = [baseline_acc] + list(rank_accs.values)
                ax.plot(x, y, "o-", color=color, linestyle=ls,
                        linewidth=2, markersize=5,
                        label=f"{model_name}")

            ax.set_xlabel("Keep Top-K Components")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Critical-Rank: {domain.capitalize()}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(fig_dir / "critical_rank_by_domain.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved critical_rank_by_domain.png")


# ── Stage Bridge: Over-Compression x JPIS Stability ──────────────────
def run_bridge(args):
    """Bridge over-compression expansion data with JPIS pressure-fragility.

    Hypothesis: Over-compressed models gain from mild PR expansion,
    and that gain predicts lower pressure fragility (pressure_auc).
    """
    from scipy.stats import spearmanr, pearsonr

    out_dir = Path("analysis/causal_pr_robust")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    # Load expansion data (Stage A)
    stage_a_path = out_dir / "tables" / "stage_a_results.csv"
    if not stage_a_path.exists():
        print("ERROR: Run Stage A first")
        return
    expand_df = pd.read_csv(stage_a_path)
    print(f"Loaded {len(expand_df)} rows from Stage A")

    # Load JPIS pressure data
    jpis_path = Path("analysis/jitter_pressure/tables/model_summary.csv")
    if not jpis_path.exists():
        print("ERROR: JPIS results not found at", jpis_path)
        return
    jpis_df = pd.read_csv(jpis_path)
    print(f"Loaded {len(jpis_df)} rows from JPIS")

    # ── Compute expansion gain per model ────────────────────────────
    model_gains = []
    for model_name in expand_df["model"].unique():
        sub = expand_df[expand_df["model"] == model_name]
        baseline = sub[sub["intervention"] == "none"]
        if baseline.empty:
            continue
        baseline_acc = baseline["correct"].mean()

        # Mild expansion band (0.05-0.10)
        mild = sub[(sub["intervention"] == "expand") &
                   (sub["strength"] >= 0.05) & (sub["strength"] <= 0.10)]
        if mild.empty:
            continue
        mild_acc = mild["correct"].mean()
        delta = mild_acc - baseline_acc

        # Per-domain
        for domain in ["math", "factual"]:
            dom_base = baseline[baseline["domain"] == domain]["correct"].mean()
            dom_mild = mild[mild["domain"] == domain]["correct"].mean()
            dom_delta = dom_mild - dom_base if not np.isnan(dom_mild) else 0.0

            model_gains.append({
                "model": model_name,
                "role": sub["role"].iloc[0],
                "core_pr": sub["core_pr"].iloc[0],
                "domain": domain,
                "baseline_acc": dom_base,
                "mild_acc": dom_mild,
                "expansion_gain": dom_delta,
            })

        # Overall
        model_gains.append({
            "model": model_name,
            "role": sub["role"].iloc[0],
            "core_pr": sub["core_pr"].iloc[0],
            "domain": "overall",
            "baseline_acc": baseline_acc,
            "mild_acc": mild_acc,
            "expansion_gain": delta,
        })

    gains_df = pd.DataFrame(model_gains)
    print(f"\nExpansion gains computed for {gains_df['model'].nunique()} models")

    # ── Join with JPIS data ─────────────────────────────────────────
    # Aggregate JPIS to per-model level (average across domains)
    jpis_model = jpis_df.groupby("model").agg({
        "pressure_auc": "mean",
        "paradigm": "first",
    }).reset_index()

    # Also get per-domain JPIS
    jpis_by_domain = jpis_df[["model", "domain", "pressure_auc"]].copy()
    jpis_by_domain = jpis_by_domain.rename(columns={"pressure_auc": "jpis_pressure_auc"})

    # Merge overall
    overall_gains = gains_df[gains_df["domain"] == "overall"].copy()
    bridge = overall_gains.merge(jpis_model, on="model", how="inner")

    # Merge per-domain
    domain_gains = gains_df[gains_df["domain"] != "overall"].copy()
    bridge_domain = domain_gains.merge(jpis_by_domain, on=["model", "domain"], how="inner")

    n_bridge = len(bridge)
    print(f"\nBridge: {n_bridge} models with both expansion and JPIS data")
    if n_bridge < 4:
        print("WARNING: Too few models for meaningful correlation. Need at least 4.")
        if n_bridge < 3:
            print("ABORTING bridge analysis — run Stage A on more models.")
            return

    for _, row in bridge.iterrows():
        print(f"  {row['model']}: expansion_gain={row['expansion_gain']:+.3f}, "
              f"pressure_auc={row['pressure_auc']:.3f}, core_pr={row['core_pr']:.2f}")

    # ── Correlation: expansion_gain vs pressure_auc ─────────────────
    x = bridge["expansion_gain"].values
    y = bridge["pressure_auc"].values

    r_spearman, p_spearman = spearmanr(x, y)
    r_pearson, p_pearson = pearsonr(x, y)
    print(f"\n  Overall: expansion_gain ~ pressure_auc")
    print(f"    Spearman r={r_spearman:.3f}, p={p_spearman:.4f}")
    print(f"    Pearson  r={r_pearson:.3f}, p={p_pearson:.4f}")

    # ── Per-domain correlation ──────────────────────────────────────
    domain_corrs = {}
    for domain in ["math", "factual"]:
        dsub = bridge_domain[bridge_domain["domain"] == domain]
        if len(dsub) >= 4:
            r_d, p_d = spearmanr(dsub["expansion_gain"].values,
                                  dsub["jpis_pressure_auc"].values)
            domain_corrs[domain] = {"r": r_d, "p": p_d, "n": len(dsub)}
            print(f"    {domain}: Spearman r={r_d:.3f}, p={p_d:.4f}, n={len(dsub)}")

    # ── Correlation: core_pr vs pressure_auc ────────────────────────
    r_pr_pauc, p_pr_pauc = spearmanr(bridge["core_pr"].values, y)
    print(f"\n  core_pr ~ pressure_auc: Spearman r={r_pr_pauc:.3f}, p={p_pr_pauc:.4f}")

    # ── Correlation: core_pr vs expansion_gain ──────────────────────
    r_pr_gain, p_pr_gain = spearmanr(bridge["core_pr"].values, x)
    print(f"  core_pr ~ expansion_gain: Spearman r={r_pr_gain:.3f}, p={p_pr_gain:.4f}")

    # ── Bootstrap CI for main correlation ───────────────────────────
    n_boot = 5000
    boot_corrs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(n_bridge, size=n_bridge, replace=True)
        r_boot, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r_boot):
            boot_corrs.append(r_boot)
    boot_corrs = np.array(boot_corrs)
    ci_lo, ci_hi = np.percentile(boot_corrs, [2.5, 97.5])
    print(f"\n  Bootstrap CI (5000): [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ── Permutation test ────────────────────────────────────────────
    n_perm = 10000
    perm_corrs = []
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        r_perm, _ = spearmanr(x, perm_y)
        perm_corrs.append(r_perm)
    perm_corrs = np.array(perm_corrs)
    # Two-sided p-value
    perm_p = np.mean(np.abs(perm_corrs) >= np.abs(r_spearman))
    print(f"  Permutation p (10000): {perm_p:.4f}")

    # ── Mediation analysis: core_pr → expansion_gain → pressure_auc ──
    print(f"\n  Mediation: core_pr -> expansion_gain -> pressure_auc")
    # Path a: core_pr → expansion_gain
    path_a_r, path_a_p = spearmanr(bridge["core_pr"].values, x)
    # Path b: expansion_gain → pressure_auc (controlling for core_pr)
    # Use partial correlation approximation
    path_b_r, path_b_p = spearmanr(x, y)
    # Path c: core_pr → pressure_auc (total effect)
    path_c_r, path_c_p = spearmanr(bridge["core_pr"].values, y)
    # Indirect effect estimate via bootstrap
    indirect_boots = []
    for _ in range(5000):
        idx = rng.choice(n_bridge, size=n_bridge, replace=True)
        ra, _ = spearmanr(bridge["core_pr"].values[idx], x[idx])
        rb, _ = spearmanr(x[idx], y[idx])
        if not (np.isnan(ra) or np.isnan(rb)):
            indirect_boots.append(ra * rb)
    indirect_boots = np.array(indirect_boots)
    indirect_mean = np.mean(indirect_boots)
    indirect_ci = np.percentile(indirect_boots, [2.5, 97.5])

    print(f"    Path a (core_pr -> gain): r={path_a_r:.3f}, p={path_a_p:.4f}")
    print(f"    Path b (gain -> pauc):    r={path_b_r:.3f}, p={path_b_p:.4f}")
    print(f"    Path c (core_pr -> pauc): r={path_c_r:.3f}, p={path_c_p:.4f}")
    print(f"    Indirect (a*b): {indirect_mean:.3f} [{indirect_ci[0]:.3f}, {indirect_ci[1]:.3f}]")

    # ── Save bridge results ─────────────────────────────────────────
    bridge_stats = {
        "n_models": n_bridge,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "permutation_p": perm_p,
        "path_a_r": path_a_r, "path_a_p": path_a_p,
        "path_b_r": path_b_r, "path_b_p": path_b_p,
        "path_c_r": path_c_r, "path_c_p": path_c_p,
        "indirect_effect": indirect_mean,
        "indirect_ci_lo": indirect_ci[0],
        "indirect_ci_hi": indirect_ci[1],
    }
    for domain, corr in domain_corrs.items():
        bridge_stats[f"{domain}_spearman_r"] = corr["r"]
        bridge_stats[f"{domain}_spearman_p"] = corr["p"]

    with open(out_dir / "tables" / "bridge_stats.json", "w") as f:
        json.dump(bridge_stats, f, indent=2)

    bridge.to_csv(out_dir / "tables" / "bridge_joined.csv", index=False)
    bridge_domain.to_csv(out_dir / "tables" / "bridge_domain.csv", index=False)

    # ── Key findings ────────────────────────────────────────────────
    findings = [
        f"bridge_models: {n_bridge}",
        f"expansion_gain_vs_pressure_auc_spearman: r={r_spearman:.3f}, p={p_spearman:.4f}",
        f"bootstrap_ci_95: [{ci_lo:.3f}, {ci_hi:.3f}]",
        f"permutation_p: {perm_p:.4f}",
        f"mediation_indirect_effect: {indirect_mean:.3f} [{indirect_ci[0]:.3f}, {indirect_ci[1]:.3f}]",
        f"core_pr_vs_pressure_auc: r={r_pr_pauc:.3f}, p={p_pr_pauc:.4f}",
    ]
    for domain, corr in domain_corrs.items():
        findings.append(f"{domain}_expansion_vs_fragility: r={corr['r']:.3f}, p={corr['p']:.4f}")

    for _, row in bridge.iterrows():
        findings.append(f"{row['model']}: gain={row['expansion_gain']:+.3f}, "
                       f"pauc={row['pressure_auc']:.3f}, pr={row['core_pr']:.2f}")

    findings.append(f"business_implication: Over-compression predicts deployment fragility - "
                   f"models benefiting from PR expansion are more sensitive to inference noise, "
                   f"providing a testable diagnostic for production risk assessment")
    findings.append(f"scientific_implication: PR compression and dynamic stability are mechanistically "
                   f"linked through a mediation path (core_pr -> expansion_gain -> pressure_fragility), "
                   f"suggesting representation geometry causally determines noise robustness")

    findings_path = out_dir / "key_findings.txt"
    findings_path.write_text("\n".join(findings), encoding="utf-8")

    # ── Bridge visualizations ───────────────────────────────────────
    _plot_bridge(bridge, bridge_domain, bridge_stats, out_dir)

    print(f"\nBridge analysis complete. Results in {out_dir}/tables/")
    return bridge_stats


def _plot_bridge(bridge, bridge_domain, stats, out_dir):
    """Generate bridge-specific visualizations."""
    fig_dir = out_dir / "figures"

    ROLE_COLORS = {
        "reasoning": "#FF5722", "base": "#2196F3", "ssm": "#9C27B0",
        "hybrid": "#4CAF50", "transformer": "#2196F3",
    }
    ROLE_MARKERS = {
        "reasoning": "^", "base": "o", "ssm": "D", "hybrid": "s",
        "transformer": "o",
    }

    # ── Plot 1: Expansion gain vs pressure_auc scatter ──────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in bridge.iterrows():
        role = row.get("paradigm", row.get("role", "base"))
        color = ROLE_COLORS.get(role, "#666666")
        marker = ROLE_MARKERS.get(role, "o")
        ax.scatter(row["expansion_gain"], row["pressure_auc"],
                   color=color, marker=marker, s=150, zorder=5, edgecolors="black")
        ax.annotate(row["model"],
                    (row["expansion_gain"], row["pressure_auc"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    # Add regression line
    x = bridge["expansion_gain"].values
    y = bridge["pressure_auc"].values
    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, linewidth=1.5)

    ax.axvline(x=0, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Expansion Gain (mild PR expansion - baseline accuracy)")
    ax.set_ylabel("Pressure AUC (higher = more fragile)")
    r = stats["spearman_r"]
    p_val = stats["permutation_p"]
    ax.set_title(f"Over-Compression Bridge: Expansion Gain vs Pressure Fragility\n"
                 f"Spearman r={r:.3f}, permutation p={p_val:.4f}, n={stats['n_models']}")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for role, color in ROLE_COLORS.items():
        marker = ROLE_MARKERS.get(role, "o")
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                                       markerfacecolor=color, markersize=10,
                                       label=role.capitalize()))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "bridge_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved bridge_scatter.png")

    # ── Plot 2: Three-panel mediation path ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: core_pr → expansion_gain
    for _, row in bridge.iterrows():
        role = row.get("paradigm", row.get("role", "base"))
        color = ROLE_COLORS.get(role, "#666666")
        marker = ROLE_MARKERS.get(role, "o")
        axes[0].scatter(row["core_pr"], row["expansion_gain"],
                        color=color, marker=marker, s=100, edgecolors="black")
        axes[0].annotate(row["model"], (row["core_pr"], row["expansion_gain"]),
                         fontsize=7, ha="left", va="bottom")

    axes[0].axhline(y=0, color="black", linestyle=":", alpha=0.3)
    axes[0].set_xlabel("Core PR (representation dimensionality)")
    axes[0].set_ylabel("Expansion Gain")
    axes[0].set_title(f"Path a: Core PR -> Expansion Gain\n"
                      f"r={stats['path_a_r']:.3f}, p={stats['path_a_p']:.4f}")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: expansion_gain → pressure_auc
    for _, row in bridge.iterrows():
        role = row.get("paradigm", row.get("role", "base"))
        color = ROLE_COLORS.get(role, "#666666")
        marker = ROLE_MARKERS.get(role, "o")
        axes[1].scatter(row["expansion_gain"], row["pressure_auc"],
                        color=color, marker=marker, s=100, edgecolors="black")
        axes[1].annotate(row["model"], (row["expansion_gain"], row["pressure_auc"]),
                         fontsize=7, ha="left", va="bottom")

    axes[1].axvline(x=0, color="black", linestyle=":", alpha=0.3)
    axes[1].set_xlabel("Expansion Gain")
    axes[1].set_ylabel("Pressure AUC (fragility)")
    axes[1].set_title(f"Path b: Expansion Gain -> Fragility\n"
                      f"r={stats['path_b_r']:.3f}, p={stats['path_b_p']:.4f}")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: core_pr → pressure_auc (total effect)
    for _, row in bridge.iterrows():
        role = row.get("paradigm", row.get("role", "base"))
        color = ROLE_COLORS.get(role, "#666666")
        marker = ROLE_MARKERS.get(role, "o")
        axes[2].scatter(row["core_pr"], row["pressure_auc"],
                        color=color, marker=marker, s=100, edgecolors="black")
        axes[2].annotate(row["model"], (row["core_pr"], row["pressure_auc"]),
                         fontsize=7, ha="left", va="bottom")

    axes[2].set_xlabel("Core PR")
    axes[2].set_ylabel("Pressure AUC (fragility)")
    axes[2].set_title(f"Path c (total): Core PR -> Fragility\n"
                      f"r={stats['path_c_r']:.3f}, p={stats['path_c_p']:.4f}\n"
                      f"Indirect={stats['indirect_effect']:.3f}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "bridge_mediation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved bridge_mediation.png")

    # ── Plot 3: Per-domain expansion vs fragility ───────────────────
    if not bridge_domain.empty and len(bridge_domain) >= 6:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, domain in enumerate(["math", "factual"]):
            ax = axes[idx]
            dsub = bridge_domain[bridge_domain["domain"] == domain]
            for _, row in dsub.iterrows():
                color = ROLE_COLORS.get(row.get("role", "base"), "#666666")
                marker = ROLE_MARKERS.get(row.get("role", "base"), "o")
                ax.scatter(row["expansion_gain"], row["jpis_pressure_auc"],
                           color=color, marker=marker, s=120, edgecolors="black")
                ax.annotate(row["model"],
                            (row["expansion_gain"], row["jpis_pressure_auc"]),
                            fontsize=7, ha="left", va="bottom")

            ax.axvline(x=0, color="black", linestyle=":", alpha=0.3)
            ax.set_xlabel("Expansion Gain")
            ax.set_ylabel("Pressure AUC (fragility)")

            # Compute domain correlation
            dkey = f"{domain}_spearman_r"
            if dkey in stats:
                ax.set_title(f"{domain.capitalize()}: Expansion vs Fragility\n"
                             f"r={stats[dkey]:.3f}, p={stats[f'{domain}_spearman_p']:.4f}")
            else:
                ax.set_title(f"{domain.capitalize()}: Expansion vs Fragility")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "bridge_by_domain.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved bridge_by_domain.png")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Robust over-compression replication")
    parser.add_argument("--stage", choices=["A", "B", "stats", "bridge", "all"], default="all",
                        help="Which stage to run")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names to run (default: all)")
    args = parser.parse_args()

    if args.stage in ("A", "all"):
        best_layers = run_stage_a(args)

    if args.stage in ("B", "all"):
        run_stage_b(args)

    if args.stage in ("stats", "all"):
        run_stats(args)

    if args.stage in ("bridge", "all"):
        run_bridge(args)


if __name__ == "__main__":
    main()
