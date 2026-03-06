#!/usr/bin/env python3
"""
orthogonality_cross_paradigm_015.py — Exp-015: 7B+ Cross-Paradigm Orthogonality Resolution

Codex-directed follow-up to Exp-014. Tests whether the interaction detected at
7B+ scale is a universal scale law or transformer-specific heterogeneity by
adding SSM, hybrid, and RWKV paradigm coverage at 7B+.

Design:
  10 primary models × 1 layer (mid=0.5) × 2×2 factorial × 5 seeds × 64 prompts
  = 12,800 trials (if all 10 complete)

  Reuses exp-014 data for 5 completed models (5 seeds from [11,23,37,43,59])
  Only runs new models; appends to combined CSV.

  Fixed strengths: surgery=0.08, jitter=0.08
  Layer: mid-layer only (0.5 of total layers)
  Seeds: [11, 23, 37, 43, 59] (5 seeds, reduced from 7 to spend budget on models)
  Prompts: Same 64 from exp-014 for direct comparability

Completion gate: >=8 models with >=1 hybrid and >=1 SSM/RWKV

Decision rules (pre-registered):
  - Pooled effect stays negative AND permutation p<0.05 → breakthrough strengthens
  - Transformers negative, SSM/hybrid/RWKV near zero → "large-transformer-specific coupling"
  - Pooled effect returns to ROPE → exp-014 was false positive / overfit risk
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
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
from scipy import stats as sp_stats

# ── Pre-registered constants ─────────────────────────────────────────────
SESOI_PROB = 0.012      # smallest effect of interest on probability scale
ROPE_OR = (0.90, 1.11)  # region of practical equivalence on OR scale

LAYER_FRACTION = 0.5     # mid-layer only
SURGERY_STRENGTH = 0.08
JITTER_STRENGTH = 0.08
SEEDS = [11, 23, 37, 43, 59]  # 5 seeds (reduced from 7 for more model budget)
MAX_NEW_TOKENS = 10

# Batch size by paradigm (conservative for VRAM on RTX 5090 25.7GB)
BATCH_SIZES = {"transformer": 4, "ssm": 4, "hybrid": 1, "reasoning": 1, "rwkv": 1}

# ── Model registry (Codex-specified 10 primary + 3 fallback) ─────────────
MODELS: List[Tuple[str, str, str, float]] = [
    # Transformers (4) — 3 cached from exp-014
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B", "transformer", 7.0),
    ("Qwen/Qwen3-8B", "Qwen3-8B", "transformer", 8.0),
    ("allenai/OLMo-2-1124-7B", "OLMo2-7B", "transformer", 7.0),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama3.1-8B-I", "transformer", 8.0),
    # Reasoning (2) — cached from exp-014
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DSR1-7B", "reasoning", 7.0),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DSR1-Llama-8B", "reasoning", 8.0),
    # SSM / recurrent (2)
    ("tiiuae/falcon-mamba-7b-instruct", "FalconMamba-7B-I", "ssm", 7.0),
    ("RWKV/v5-Eagle-7B-HF", "RWKVv5-Eagle-7B", "ssm", 7.0),
    # Hybrid (2)
    ("togethercomputer/StripedHyena-Nous-7B", "StripedHyena-7B", "hybrid", 7.0),
    ("Zyphra/Zamba2-7B-Instruct", "Zamba2-7B-I", "hybrid", 7.0),
]

FALLBACK_MODELS: List[Tuple[str, str, str, float]] = [
    ("tiiuae/falcon-mamba-7b", "FalconMamba-7B", "ssm", 7.0),
    ("RWKV/v6-Finch-7B-HF", "RWKVv6-Finch-7B", "ssm", 7.0),
    ("togethercomputer/StripedHyena-Hessian-7B", "StripedHyena-H-7B", "hybrid", 7.0),
]

# ── Prompts: 64 total (32 cal + 32 holdout) — identical to exp-014 ──────
PROMPTS = {
    "math": [
        {"id": "m01", "prompt": "What is 7 * 8? Answer with just the number:", "answer": "56", "split": "cal"},
        {"id": "m02", "prompt": "What is 15 + 27? Answer with just the number:", "answer": "42", "split": "cal"},
        {"id": "m03", "prompt": "What is 100 - 37? Answer with just the number:", "answer": "63", "split": "cal"},
        {"id": "m04", "prompt": "What is 144 / 12? Answer with just the number:", "answer": "12", "split": "cal"},
        {"id": "m05", "prompt": "What is 9 * 9? Answer with just the number:", "answer": "81", "split": "cal"},
        {"id": "m06", "prompt": "What is 23 + 19? Answer with just the number:", "answer": "42", "split": "cal"},
        {"id": "m07", "prompt": "What is 256 / 16? Answer with just the number:", "answer": "16", "split": "cal"},
        {"id": "m08", "prompt": "What is 11 * 11? Answer with just the number:", "answer": "121", "split": "cal"},
        {"id": "m09", "prompt": "What is 225 / 15? Answer with just the number:", "answer": "15", "split": "hold"},
        {"id": "m10", "prompt": "What is 37 + 48? Answer with just the number:", "answer": "85", "split": "hold"},
        {"id": "m11", "prompt": "What is 16 * 7? Answer with just the number:", "answer": "112", "split": "hold"},
        {"id": "m12", "prompt": "What is 400 - 213? Answer with just the number:", "answer": "187", "split": "hold"},
    ],
    "factual": [
        {"id": "f01", "prompt": "The capital of Japan is", "answer": "Tokyo", "split": "cal"},
        {"id": "f02", "prompt": "Water freezes at 0 degrees", "answer": "Celsius", "split": "cal"},
        {"id": "f03", "prompt": "The chemical symbol for gold is", "answer": "Au", "split": "cal"},
        {"id": "f04", "prompt": "The largest planet in our solar system is", "answer": "Jupiter", "split": "cal"},
        {"id": "f05", "prompt": "The speed of light is approximately 300,000 km per", "answer": "second", "split": "cal"},
        {"id": "f06", "prompt": "DNA stands for deoxyribonucleic", "answer": "acid", "split": "cal"},
        {"id": "f07", "prompt": "The square root of 144 is", "answer": "12", "split": "cal"},
        {"id": "f08", "prompt": "The atomic number of carbon is", "answer": "6", "split": "cal"},
        {"id": "f09", "prompt": "The largest ocean on Earth is the", "answer": "Pacific", "split": "hold"},
        {"id": "f10", "prompt": "The hardest natural substance is", "answer": "diamond", "split": "hold"},
        {"id": "f11", "prompt": "The chemical symbol for sodium is", "answer": "Na", "split": "hold"},
        {"id": "f12", "prompt": "The closest star to Earth is the", "answer": "Sun", "split": "hold"},
    ],
    "logic": [
        {"id": "l01", "prompt": "If all roses are flowers and all flowers need water, then all roses need", "answer": "water", "split": "cal"},
        {"id": "l02", "prompt": "If today is Monday, then tomorrow is", "answer": "Tuesday", "split": "cal"},
        {"id": "l03", "prompt": "A dozen eggs is exactly", "answer": "12", "split": "cal"},
        {"id": "l04", "prompt": "If a triangle has angles of 60, 60, and 60 degrees, it is called", "answer": "equilateral", "split": "cal"},
        {"id": "l05", "prompt": "The opposite of 'increase' is", "answer": "decrease", "split": "cal"},
        {"id": "l06", "prompt": "Half of 200 is", "answer": "100", "split": "cal"},
        {"id": "l07", "prompt": "If 3x = 21, then x equals", "answer": "7", "split": "cal"},
        {"id": "l08", "prompt": "The next number in the sequence 2, 4, 8, 16 is", "answer": "32", "split": "cal"},
        {"id": "l09", "prompt": "If a car travels at 60 km/h for 2 hours, the total distance is", "answer": "120", "split": "hold"},
        {"id": "l10", "prompt": "The number of sides in a hexagon is", "answer": "6", "split": "hold"},
        {"id": "l11", "prompt": "If 5 + x = 13, then x equals", "answer": "8", "split": "hold"},
        {"id": "l12", "prompt": "The next prime after 7 is", "answer": "11", "split": "hold"},
    ],
    "hard": [
        {"id": "h01", "prompt": "What is 17 * 19? Answer with just the number:", "answer": "323", "split": "cal"},
        {"id": "h02", "prompt": "What is the cube root of 27?", "answer": "3", "split": "cal"},
        {"id": "h03", "prompt": "What is 15% of 200? Answer with just the number:", "answer": "30", "split": "cal"},
        {"id": "h04", "prompt": "If f(x) = 2x + 3, what is f(5)? Answer with just the number:", "answer": "13", "split": "cal"},
        {"id": "h05", "prompt": "The element with atomic number 79 is", "answer": "gold", "split": "cal"},
        {"id": "h06", "prompt": "What is 2^10? Answer with just the number:", "answer": "1024", "split": "cal"},
        {"id": "h07", "prompt": "The derivative of x^2 is", "answer": "2x", "split": "cal"},
        {"id": "h08", "prompt": "How many degrees in a right angle?", "answer": "90", "split": "cal"},
        {"id": "h09", "prompt": "What is 23 * 17? Answer with just the number:", "answer": "391", "split": "hold"},
        {"id": "h10", "prompt": "What is the square root of 225?", "answer": "15", "split": "hold"},
        {"id": "h11", "prompt": "What is 3^4? Answer with just the number:", "answer": "81", "split": "hold"},
        {"id": "h12", "prompt": "What is 7^3? Answer with just the number:", "answer": "343", "split": "hold"},
    ],
    "multistep": [
        {"id": "ms01", "prompt": "If 3x + 7 = 22, what is x? Answer with just the number:", "answer": "5", "split": "hold"},
        {"id": "ms02", "prompt": "A shop sells apples for $2 each. If you buy 7 and pay with $20, how much change? Answer with just the number:", "answer": "6", "split": "hold"},
        {"id": "ms03", "prompt": "What is 13 * 13? Answer with just the number:", "answer": "169", "split": "hold"},
        {"id": "ms04", "prompt": "If a rectangle has length 12 and width 5, what is its area? Answer with just the number:", "answer": "60", "split": "hold"},
        {"id": "ms05", "prompt": "What is 2^8? Answer with just the number:", "answer": "256", "split": "hold"},
        {"id": "ms06", "prompt": "The sum of angles in a triangle is how many degrees? Answer with just the number:", "answer": "180", "split": "hold"},
        {"id": "ms07", "prompt": "If you travel 150 km in 3 hours, what is your speed in km/h? Answer with just the number:", "answer": "50", "split": "hold"},
        {"id": "ms08", "prompt": "What is 999 + 1? Answer with just the number:", "answer": "1000", "split": "hold"},
        {"id": "ms09", "prompt": "How many seconds are in one hour? Answer with just the number:", "answer": "3600", "split": "hold"},
        {"id": "ms10", "prompt": "What is 25% of 80? Answer with just the number:", "answer": "20", "split": "hold"},
        {"id": "ms11", "prompt": "If f(x) = x^2 - 1, what is f(4)? Answer with just the number:", "answer": "15", "split": "hold"},
        {"id": "ms12", "prompt": "A coin is flipped 3 times. Total number of possible outcomes? Answer with just the number:", "answer": "8", "split": "hold"},
        {"id": "ms13", "prompt": "What is the GCD of 12 and 18? Answer with just the number:", "answer": "6", "split": "hold"},
        {"id": "ms14", "prompt": "A right triangle has legs 3 and 4. What is the hypotenuse? Answer with just the number:", "answer": "5", "split": "hold"},
        {"id": "ms15", "prompt": "What is 7! / 5!? Answer with just the number:", "answer": "42", "split": "hold"},
        {"id": "ms16", "prompt": "How many prime numbers are between 1 and 20? Answer with just the number:", "answer": "8", "split": "hold"},
    ],
}


# ── Strict answer parsing (identical to exp-013/014) ─────────────────────
def parse_numeric(text: str) -> Optional[str]:
    text = text.strip().lower()
    text = re.sub(r'[.,;:!?\s]+$', '', text)
    match = re.search(r'-?\d[\d,]*\.?\d*', text)
    if match:
        return match.group().replace(',', '')
    return None


def check_correct(generated: str, answer: str, domain: str) -> Tuple[bool, float]:
    gen_lower = generated.strip().lower()
    ans_lower = answer.lower()

    if domain in ("math", "hard", "logic", "multistep") and answer.replace('.', '').replace('-', '').isdigit():
        parsed = parse_numeric(gen_lower)
        if parsed is not None:
            try:
                if abs(float(parsed) - float(answer)) < 0.01:
                    return True, 1.0
            except ValueError:
                pass
        if ans_lower in gen_lower:
            return True, 0.5
        return False, 0.0

    if ans_lower in gen_lower:
        return True, 1.0
    synonyms = {
        "celsius": ["celsius", "centigrade", "\u00b0c"],
        "squared": ["squared", "\u00b2", "square"],
    }
    if ans_lower in synonyms:
        for syn in synonyms[ans_lower]:
            if syn in gen_lower:
                return True, 0.8
    return False, 0.0


# ── Hooks (identical to exp-012/013/014) ──────────────────────────────────
class PRExpansionHook:
    def __init__(self, strength: float, seed: int, n_components: int = 5):
        self.strength = strength
        self.seed = seed
        self.n_components = n_components
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

    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


class JitterHook:
    def __init__(self, strength: float, seed: int):
        self.strength = strength
        self.seed = seed
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        noise = torch.randn(hidden.shape, generator=rng, dtype=torch.float32).to(hidden.device)
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
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ),
    )
    model.eval()
    return model, tokenizer


def get_model_layers(model) -> List[nn.Module]:
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
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 2:
            return list(module)
    return []


def get_mid_layer_idx(n_layers: int) -> int:
    return int(LAYER_FRACTION * (n_layers - 1))


# ── Generation ───────────────────────────────────────────────────────────
def generate_single(model, tokenizer, prompt_item, device):
    inp = tokenizer(
        prompt_item["prompt"], return_tensors="pt",
        truncation=True, max_length=128
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inp["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    correct, confidence = check_correct(text, prompt_item["answer"], prompt_item["domain"])
    return text, correct, confidence


def generate_batch(model, tokenizer, prompts_batch, device):
    texts = [p["prompt"] for p in prompts_batch]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    results = []
    for i, p in enumerate(prompts_batch):
        input_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
        gen_ids = outputs[i][input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        correct, confidence = check_correct(text, p["answer"], p["domain"])
        results.append((text, correct, confidence))
    return results


def generate_prompts(model, tokenizer, prompts, device, batch_size):
    """Generate with automatic fallback from batch to single-prompt."""
    all_results = []
    if batch_size > 1:
        try:
            for batch_start in range(0, len(prompts), batch_size):
                batch = prompts[batch_start:batch_start + batch_size]
                results = generate_batch(model, tokenizer, batch, device)
                all_results.extend(zip(batch, results))
            return all_results
        except Exception as e:
            print(f"    [BATCH FAIL] Falling back to single-prompt: {type(e).__name__}")
            torch.cuda.empty_cache()
            all_results = []

    # Single-prompt fallback
    for p in prompts:
        try:
            torch.cuda.empty_cache()
            text, correct, confidence = generate_single(model, tokenizer, p, device)
            all_results.append((p, (text, correct, confidence)))
        except Exception:
            all_results.append((p, ("ERROR", False, 0.0)))
    return all_results


# ── Data import from exp-014 ─────────────────────────────────────────────
def import_exp014_data(out_dir: Path) -> pd.DataFrame:
    """Import exp-014 data, filtering to seeds used in exp-015."""
    exp014_csv = Path("analysis/orthogonality_scale_014/tables/raw_trials.csv")
    if not exp014_csv.exists():
        print("  [WARN] exp-014 data not found, starting fresh")
        return pd.DataFrame()

    df = pd.read_csv(exp014_csv)
    # Filter to seeds used in this experiment
    df = df[df["seed"].isin(SEEDS)]
    print(f"  [IMPORT] Loaded {len(df)} rows from exp-014 "
          f"({df['model_name'].nunique()} models, seeds {sorted(df['seed'].unique())})")
    return df


# ── Data collection ─────────────────────────────────────────────────────
def collect_data(out_dir: Path) -> pd.DataFrame:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = tables_dir / "raw_trials.csv"

    all_rows = []
    failures = []
    completed_models = set()

    if raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        all_rows = existing.to_dict("records")
        completed_models = set(existing["model_name"].unique())
        print(f"  [RESUME] Loaded {len(all_rows)} rows for {completed_models}")
    else:
        # Import exp-014 data on first run
        exp014_df = import_exp014_data(out_dir)
        if len(exp014_df) > 0:
            all_rows = exp014_df.to_dict("records")
            completed_models = set(exp014_df["model_name"].unique())
            # Save immediately
            pd.DataFrame(all_rows).to_csv(raw_csv, index=False)
            print(f"  [IMPORT] Seeded with {len(all_rows)} exp-014 rows ({completed_models})")

    # Flatten prompts
    all_prompts = []
    for domain, items in PROMPTS.items():
        for item in items:
            all_prompts.append({**item, "domain": domain})

    # 2×2 factorial conditions
    conditions = [
        (0.0, 0.0, "control"),
        (SURGERY_STRENGTH, 0.0, "surgery_only"),
        (0.0, JITTER_STRENGTH, "jitter_only"),
        (SURGERY_STRENGTH, JITTER_STRENGTH, "both"),
    ]

    n_conditions = len(conditions) * len(SEEDS)

    # Build full model list (primary + fallback)
    active_models = list(MODELS)
    # Check if we need fallbacks (SSM/hybrid coverage)
    completed_paradigms = set()
    for model_id, model_name, paradigm, params_b in MODELS:
        if model_name in completed_models:
            completed_paradigms.add(paradigm)

    total_trials = len(active_models) * n_conditions * len(all_prompts)
    print(f"\n{'=' * 70}")
    print(f"EXP-015: 7B+ CROSS-PARADIGM ORTHOGONALITY RESOLUTION")
    print(f"Models: {len(active_models)} primary + {len(FALLBACK_MODELS)} fallback")
    print(f"Prompts: {len(all_prompts)}, Seeds: {len(SEEDS)}")
    print(f"Layer: mid ({LAYER_FRACTION}), Surgery: {SURGERY_STRENGTH}, Jitter: {JITTER_STRENGTH}")
    print(f"Conditions per model: {n_conditions}")
    print(f"Already completed: {completed_models}")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, paradigm, params_b) in enumerate(active_models):
        if model_name in completed_models:
            print(f"  [{model_idx+1}/{len(active_models)}] SKIP {model_name} (completed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{model_idx+1}/{len(active_models)}] {model_name} ({paradigm}, {params_b}B)")
        print(f"{'=' * 60}")

        t_model = time.time()
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"  [FAIL] {model_id}: {e}")
            failures.append({"model_id": model_id, "model_name": model_name,
                             "paradigm": paradigm, "error": str(e)})
            continue

        layers = get_model_layers(model)
        if not layers:
            print(f"  [FAIL] No layers for {model_name}")
            failures.append({"model_id": model_id, "model_name": model_name,
                             "paradigm": paradigm, "error": "no_layers"})
            try:
                del model, tokenizer
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

        n_layers = len(layers)
        mid_idx = get_mid_layer_idx(n_layers)
        layer_frac = mid_idx / max(n_layers - 1, 1)
        print(f"  Layers: {n_layers}, target: layer {mid_idx} (frac={layer_frac:.2f})")
        print(f"  Loaded in {time.time() - t_model:.1f}s")

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM: {mem_gb:.1f}GB allocated")

        device = next(model.parameters()).device
        batch_size = BATCH_SIZES.get(paradigm, 1)
        model_rows = []
        cond_count = 0
        model_failed = False

        try:
            for surg_s, jit_s, cond_label in conditions:
                for seed in SEEDS:
                    cond_count += 1
                    hooks = []

                    if surg_s > 0:
                        h = PRExpansionHook(strength=surg_s, seed=seed)
                        h.attach(layers[mid_idx])
                        hooks.append(h)
                    if jit_s > 0:
                        h = JitterHook(strength=jit_s, seed=seed + 10000)
                        h.attach(layers[mid_idx])
                        hooks.append(h)

                    results = generate_prompts(model, tokenizer, all_prompts, device, batch_size)

                    for p, (text, correct, confidence) in results:
                        model_rows.append({
                            "model_name": model_name, "paradigm": paradigm,
                            "params_b": params_b, "seed": seed,
                            "layer_idx": mid_idx, "layer_frac": round(layer_frac, 2),
                            "surgery_strength": surg_s, "jitter_strength": jit_s,
                            "condition": cond_label,
                            "domain": p["domain"], "prompt_id": p["id"],
                            "split": p["split"],
                            "answer": p["answer"], "generated": text,
                            "correct": int(correct), "confidence": confidence,
                        })

                    for h in hooks:
                        h.detach()

                    elapsed = time.time() - t_model
                    rate = len(model_rows) / elapsed if elapsed > 0 else 0
                    print(f"    [{cond_count}/{n_conditions}] {cond_label} seed={seed} "
                          f"({elapsed:.0f}s, {rate:.1f} trials/s)")

        except Exception as model_err:
            print(f"  [MODEL FAIL] {model_name}: {type(model_err).__name__}: {model_err}")
            failures.append({"model_id": model_id, "model_name": model_name,
                             "paradigm": paradigm, "error": str(model_err)})
            model_failed = True

        if not model_failed and model_rows:
            all_rows.extend(model_rows)
            completed_models.add(model_name)
            pd.DataFrame(all_rows).to_csv(raw_csv, index=False)

        try:
            del model, tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        elapsed = time.time() - t_model
        print(f"  {model_name}: {len(model_rows)} trials in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Check completion gate: try fallbacks if needed
    current_paradigms = set()
    df_current = pd.DataFrame(all_rows)
    if len(df_current) > 0:
        for mn in df_current["model_name"].unique():
            p = df_current[df_current["model_name"] == mn]["paradigm"].iloc[0]
            current_paradigms.add(p)

    need_ssm = "ssm" not in current_paradigms
    need_hybrid = "hybrid" not in current_paradigms

    if (need_ssm or need_hybrid) and FALLBACK_MODELS:
        print(f"\n{'=' * 60}")
        print(f"FALLBACK MODE: need_ssm={need_ssm}, need_hybrid={need_hybrid}")
        print(f"{'=' * 60}")

        for model_id, model_name, paradigm, params_b in FALLBACK_MODELS:
            if model_name in completed_models:
                continue
            if paradigm == "ssm" and not need_ssm:
                continue
            if paradigm == "hybrid" and not need_hybrid:
                continue

            print(f"\n  [FALLBACK] Trying {model_name} ({paradigm})")
            t_model = time.time()
            try:
                model, tokenizer = load_model_and_tokenizer(model_id)
            except Exception as e:
                print(f"  [FAIL] {model_id}: {e}")
                failures.append({"model_id": model_id, "model_name": model_name,
                                 "paradigm": paradigm, "error": str(e)})
                continue

            layers = get_model_layers(model)
            if not layers:
                print(f"  [FAIL] No layers for {model_name}")
                failures.append({"model_id": model_id, "model_name": model_name,
                                 "paradigm": paradigm, "error": "no_layers"})
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                continue

            n_layers = len(layers)
            mid_idx = get_mid_layer_idx(n_layers)
            layer_frac = mid_idx / max(n_layers - 1, 1)
            device = next(model.parameters()).device
            batch_size = BATCH_SIZES.get(paradigm, 1)
            model_rows = []
            cond_count = 0
            model_failed = False

            try:
                for surg_s, jit_s, cond_label in conditions:
                    for seed in SEEDS:
                        cond_count += 1
                        hooks = []
                        if surg_s > 0:
                            h = PRExpansionHook(strength=surg_s, seed=seed)
                            h.attach(layers[mid_idx])
                            hooks.append(h)
                        if jit_s > 0:
                            h = JitterHook(strength=jit_s, seed=seed + 10000)
                            h.attach(layers[mid_idx])
                            hooks.append(h)

                        results = generate_prompts(model, tokenizer, all_prompts, device, batch_size)
                        for p, (text, correct, confidence) in results:
                            model_rows.append({
                                "model_name": model_name, "paradigm": paradigm,
                                "params_b": params_b, "seed": seed,
                                "layer_idx": mid_idx, "layer_frac": round(layer_frac, 2),
                                "surgery_strength": surg_s, "jitter_strength": jit_s,
                                "condition": cond_label,
                                "domain": p["domain"], "prompt_id": p["id"],
                                "split": p["split"],
                                "answer": p["answer"], "generated": text,
                                "correct": int(correct), "confidence": confidence,
                            })
                        for h in hooks:
                            h.detach()

                        elapsed = time.time() - t_model
                        rate = len(model_rows) / elapsed if elapsed > 0 else 0
                        print(f"    [{cond_count}/{n_conditions}] {cond_label} seed={seed} "
                              f"({elapsed:.0f}s, {rate:.1f} trials/s)")

            except Exception as model_err:
                print(f"  [MODEL FAIL] {model_name}: {type(model_err).__name__}: {model_err}")
                failures.append({"model_id": model_id, "model_name": model_name,
                                 "paradigm": paradigm, "error": str(model_err)})
                model_failed = True

            if not model_failed and model_rows:
                all_rows.extend(model_rows)
                completed_models.add(model_name)
                pd.DataFrame(all_rows).to_csv(raw_csv, index=False)
                if paradigm == "ssm":
                    need_ssm = False
                elif paradigm == "hybrid":
                    need_hybrid = False

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            elapsed = time.time() - t_model
            print(f"  {model_name}: {len(model_rows)} trials in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if failures:
        pd.DataFrame(failures).to_csv(tables_dir / "failures.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(raw_csv, index=False)
    print(f"\nData collection complete: {len(df)} rows, {df['model_name'].nunique()} models")
    print(f"Paradigms covered: {sorted(df['paradigm'].unique())}")

    # Check completion gate
    n_models_final = df["model_name"].nunique()
    paradigms_final = set(df["paradigm"].unique())
    gate_pass = n_models_final >= 8 and "ssm" in paradigms_final and "hybrid" in paradigms_final
    print(f"Completion gate: {n_models_final} models, paradigms={paradigms_final}, PASS={gate_pass}")

    return df


# ── Analysis ─────────────────────────────────────────────────────────────
def run_analysis(out_dir: Path):
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables_dir / "raw_trials.csv")
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {len(df)} trials, {df['model_name'].nunique()} models")
    print(f"Paradigms: {sorted(df['paradigm'].unique())}")
    print(f"{'=' * 70}")

    # Check completion gate
    paradigms = set(df["paradigm"].unique())
    n_models = df["model_name"].nunique()
    has_ssm = "ssm" in paradigms
    has_hybrid = "hybrid" in paradigms
    gate_pass = n_models >= 8 and has_ssm and has_hybrid
    print(f"  Completion gate: {n_models} models, SSM={has_ssm}, hybrid={has_hybrid}, PASS={gate_pass}")
    if not gate_pass:
        print(f"  WARNING: Completion gate NOT met. Proceeding with available data.")

    # ── 1. Per-model interaction ──
    model_interactions = _compute_model_interactions(df)
    model_int_df = pd.DataFrame(model_interactions)
    model_int_df.to_csv(tables_dir / "interaction_by_model.csv", index=False)

    global_int = model_int_df["interaction"].mean()
    print(f"\nGlobal mean interaction: {global_int:.4f}")

    # ── 2. Mixed-effects model ──
    mixed = _fit_mixed_model(df, tables_dir)

    # ── 3. Bootstrap ──
    boot = _bootstrap_interaction(model_int_df, tables_dir)

    # ── 4. Permutation test ──
    perm = _permutation_test(df, tables_dir)

    # ── 5. Paradigm-specific interactions ──
    paradigm_int = model_int_df.groupby("paradigm")["interaction"].agg(["mean", "std", "count"])
    paradigm_int.to_csv(tables_dir / "interaction_by_paradigm.csv")
    print(f"\nInteraction by paradigm:\n{paradigm_int}")

    # ── 6. Reasoning-only interaction ──
    reasoning_df = df[df["paradigm"] == "reasoning"]
    if len(reasoning_df) > 0:
        reasoning_int = _compute_global_interaction(reasoning_df)
        print(f"\nReasoning-only interaction: {reasoning_int:.4f}")
    else:
        reasoning_int = np.nan

    # ── 7. Cal vs holdout ──
    cal_int = _compute_global_interaction(df[df["split"] == "cal"])
    hold_int = _compute_global_interaction(df[df["split"] == "hold"])
    split_results = {
        "calibration_interaction": float(cal_int),
        "holdout_interaction": float(hold_int),
        "gap_pp": float(abs(cal_int - hold_int) * 100),
    }
    with open(tables_dir / "split_comparison.json", "w") as f:
        json.dump(split_results, f, indent=2)
    print(f"\nCal: {cal_int:.4f}, Holdout: {hold_int:.4f} (gap: {split_results['gap_pp']:.1f}pp)")

    # ── 8. LOO and LOPO ──
    loo = _leave_one_out(model_int_df, tables_dir)
    lopo = _leave_one_paradigm_out(model_int_df, tables_dir)

    # ── 9. Main effects ──
    main_effects = _compute_main_effects(df, tables_dir)

    # ── 10. Scale-moderation meta-regression ──
    scale_mod = _scale_moderation_meta(model_int_df, tables_dir)

    # ── 11. Cochran's Q heterogeneity test (NEW for exp-015) ──
    hetero = _heterogeneity_test(model_int_df, tables_dir)

    # ── 12. Figures ──
    _plot_paradigm_forest(model_int_df, figures_dir)
    _plot_model_forest(model_int_df, figures_dir)
    _plot_scale_moderation(model_int_df, scale_mod, figures_dir)

    # ── 13. Key findings ──
    _write_findings(df, model_int_df, mixed, boot, perm, loo, lopo,
                    split_results, reasoning_int, main_effects, scale_mod,
                    hetero, gate_pass, out_dir)


def _compute_model_interactions(df: pd.DataFrame) -> List[Dict]:
    interactions = []
    for model_name in df["model_name"].unique():
        mdf = df[df["model_name"] == model_name]
        paradigm = mdf["paradigm"].iloc[0]
        params_b = mdf["params_b"].iloc[0]
        acc = {}
        for (ss, js), grp in mdf.groupby(["surgery_strength", "jitter_strength"]):
            key = ("surg" if ss > 0 else "ctrl", "jit" if js > 0 else "off")
            acc[key] = grp["correct"].mean()
        c00 = acc.get(("ctrl", "off"), np.nan)
        c10 = acc.get(("surg", "off"), np.nan)
        c01 = acc.get(("ctrl", "jit"), np.nan)
        c11 = acc.get(("surg", "jit"), np.nan)
        interaction = (c11 - c01) - (c10 - c00)
        interactions.append({
            "model_name": model_name, "paradigm": paradigm,
            "params_b": params_b, "interaction": interaction,
            "acc_control": c00, "acc_surgery": c10,
            "acc_jitter": c01, "acc_both": c11,
        })
    return interactions


def _compute_global_interaction(df: pd.DataFrame) -> float:
    ints = []
    for mn in df["model_name"].unique():
        mdf = df[df["model_name"] == mn]
        acc = {}
        for (ss, js), grp in mdf.groupby(["surgery_strength", "jitter_strength"]):
            key = ("surg" if ss > 0 else "ctrl", "jit" if js > 0 else "off")
            acc[key] = grp["correct"].mean()
        c00 = acc.get(("ctrl", "off"), np.nan)
        c10 = acc.get(("surg", "off"), np.nan)
        c01 = acc.get(("ctrl", "jit"), np.nan)
        c11 = acc.get(("surg", "jit"), np.nan)
        interaction = (c11 - c01) - (c10 - c00)
        if not np.isnan(interaction):
            ints.append(interaction)
    return np.mean(ints) if ints else 0.0


def _fit_mixed_model(df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Mixed-effects model ---")
    df = df.copy()
    df["surgery_bin"] = (df["surgery_strength"] > 0).astype(int)
    df["jitter_bin"] = (df["jitter_strength"] > 0).astype(int)
    df["interaction"] = df["surgery_bin"] * df["jitter_bin"]
    df["domain_bin"] = (df["domain"].isin(["math", "hard", "multistep"])).astype(int)
    df["log10_params"] = np.log10(df["params_b"])

    results = {}
    try:
        import statsmodels.api as sm
        exog = df[["surgery_bin", "jitter_bin", "interaction", "domain_bin",
                    "log10_params"]].copy()
        exog.insert(0, "intercept", 1)
        logit = sm.Logit(df["correct"].values, exog.values)
        fit = logit.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["model_name"]})
        names = list(exog.columns)
        for i, n in enumerate(names):
            results[f"coef_{n}"] = float(fit.params[i])
            results[f"se_{n}"] = float(fit.bse[i])
        idx = names.index("interaction")
        results["interaction_beta"] = float(fit.params[idx])
        results["interaction_se"] = float(fit.bse[idx])
        results["interaction_z"] = float(fit.tvalues[idx])
        results["interaction_p"] = float(fit.pvalues[idx])
        results["interaction_or"] = float(np.exp(fit.params[idx]))
        results["method"] = "Logit_ClusterRobust"
        print(f"  Interaction: beta={results['interaction_beta']:.4f}, "
              f"p={results['interaction_p']:.4f}, OR={results['interaction_or']:.4f}")
    except Exception as e:
        print(f"  Mixed model failed: {e}")
        results["method"] = "failed"

    with open(tables_dir / "mixed_effects_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _bootstrap_interaction(model_int_df: pd.DataFrame, tables_dir: Path, n_boot: int = 5000) -> Dict:
    print("\n--- Bootstrap (5000) ---")
    rng = np.random.default_rng(42)
    models = model_int_df["model_name"].values
    model_ints = model_int_df.set_index("model_name")["interaction"]
    n_models = len(models)

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(models, size=n_models, replace=True)
        boot_means.append(model_ints[sample].mean())

    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    mean_int = model_ints.mean()
    in_rope = np.mean((boot_means >= -SESOI_PROB) & (boot_means <= SESOI_PROB))

    results = {
        "mean_interaction": float(mean_int),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
        "sesoi_rope_fraction": float(in_rope),
        "sesoi_bound": SESOI_PROB,
        "n_boot": n_boot,
    }

    with open(tables_dir / "bootstrap_interaction.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Mean: {mean_int:.4f}, CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  ROPE ({-SESOI_PROB}, {SESOI_PROB}) fraction: {in_rope:.3f}")
    return results


def _permutation_test(df: pd.DataFrame, tables_dir: Path, n_perm: int = 5000) -> Dict:
    print("\n--- Permutation test (5000) ---")
    rng = np.random.default_rng(42)
    observed = _compute_global_interaction(df)

    null_dist = []
    for i in range(n_perm):
        df_perm = df.copy()
        for mn in df_perm["model_name"].unique():
            mask = df_perm["model_name"] == mn
            idx = df_perm.index[mask]
            df_perm.loc[idx, "surgery_strength"] = rng.permutation(df_perm.loc[idx, "surgery_strength"].values)
            df_perm.loc[idx, "jitter_strength"] = rng.permutation(df_perm.loc[idx, "jitter_strength"].values)
        null_dist.append(_compute_global_interaction(df_perm))
        if (i + 1) % 1000 == 0:
            print(f"    [{i+1}/{n_perm}]")

    null_dist = np.array(null_dist)
    p_perm = np.mean(np.abs(null_dist) >= np.abs(observed))

    results = {"observed": float(observed), "p_perm": float(p_perm),
               "null_mean": float(null_dist.mean()), "null_std": float(null_dist.std())}

    with open(tables_dir / "permutation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Observed: {observed:.4f}, p_perm: {p_perm:.4f}")
    return results


def _leave_one_out(model_int_df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Leave-one-model-out ---")
    model_ints = model_int_df.set_index("model_name")["interaction"]
    full_mean = model_ints.mean()

    loo_rows = []
    for mn in model_ints.index:
        loo_mean = model_ints.drop(mn).mean()
        loo_rows.append({"excluded": mn, "loo_mean": loo_mean, "delta": loo_mean - full_mean})

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(tables_dir / "loo_interaction.csv", index=False)

    results = {"full_mean": float(full_mean), "loo_std": float(loo_df["loo_mean"].std()),
               "most_influential": str(loo_df.loc[loo_df["delta"].abs().idxmax(), "excluded"])}
    print(f"  LOO std: {results['loo_std']:.4f}, most influential: {results['most_influential']}")
    return results


def _leave_one_paradigm_out(model_int_df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Leave-one-paradigm-out ---")
    full_mean = model_int_df["interaction"].mean()
    lopo_rows = []
    for paradigm in model_int_df["paradigm"].unique():
        subset = model_int_df[model_int_df["paradigm"] != paradigm]
        lopo_mean = subset["interaction"].mean()
        lopo_rows.append({"excluded_paradigm": paradigm, "lopo_mean": lopo_mean,
                          "delta": lopo_mean - full_mean})

    lopo_df = pd.DataFrame(lopo_rows)
    lopo_df.to_csv(tables_dir / "lopo_interaction.csv", index=False)

    results = {"full_mean": float(full_mean), "lopo_std": float(lopo_df["lopo_mean"].std()),
               "most_influential": str(lopo_df.loc[lopo_df["delta"].abs().idxmax(), "excluded_paradigm"])}
    print(f"  LOPO std: {results['lopo_std']:.4f}, most influential: {results['most_influential']}")
    return results


def _compute_main_effects(df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Main effects ---")
    acc_control = df[(df["surgery_strength"] == 0) & (df["jitter_strength"] == 0)]["correct"].mean()
    acc_surgery = df[(df["surgery_strength"] > 0) & (df["jitter_strength"] == 0)]["correct"].mean()
    acc_jitter = df[(df["surgery_strength"] == 0) & (df["jitter_strength"] > 0)]["correct"].mean()
    acc_both = df[(df["surgery_strength"] > 0) & (df["jitter_strength"] > 0)]["correct"].mean()

    surgery_effect = acc_surgery - acc_control
    jitter_effect = acc_jitter - acc_control
    baseline_acc = acc_control

    results = {
        "baseline_accuracy": float(baseline_acc),
        "surgery_main_effect": float(surgery_effect),
        "jitter_main_effect": float(jitter_effect),
        "acc_control": float(acc_control),
        "acc_surgery_only": float(acc_surgery),
        "acc_jitter_only": float(acc_jitter),
        "acc_both": float(acc_both),
    }

    with open(tables_dir / "main_effects.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Baseline: {baseline_acc:.3f}, Surgery: {surgery_effect:+.3f}, Jitter: {jitter_effect:+.3f}")
    return results


def _heterogeneity_test(model_int_df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Cochran's Q test for between-model heterogeneity in interaction effects."""
    print("\n--- Heterogeneity test (Cochran's Q) ---")
    interactions = model_int_df["interaction"].values
    k = len(interactions)
    mean_int = np.mean(interactions)

    # Cochran's Q: sum of squared deviations from mean, weighted
    # For equal-weight case: Q = k * sum((yi - ybar)^2) / var_within
    # Approximate with model-level variance
    if k < 3:
        print("  Too few models for heterogeneity test")
        results = {"cochrans_q": np.nan, "q_p": np.nan, "i_squared": np.nan,
                   "tau_squared": np.nan, "k": k}
    else:
        # Variance of model interactions
        var_between = np.var(interactions, ddof=1)
        # Approximate within-model SE (from trial counts)
        q_stat = (k - 1) * var_between / (np.mean(interactions ** 2) - mean_int ** 2 + 1e-10)
        # More direct: chi-squared test on deviations
        ss_between = np.sum((interactions - mean_int) ** 2)
        # Under null, each model interaction ~ same, so deviations are chi-sq
        # Use direct sum of squared z-scores
        se_approx = np.std(interactions, ddof=1) / np.sqrt(k)
        z_scores = (interactions - mean_int) / (np.std(interactions, ddof=1) + 1e-10)
        q_direct = np.sum(z_scores ** 2)
        q_p = 1 - sp_stats.chi2.cdf(q_direct, df=k - 1)

        # I-squared
        i_squared = max(0, (q_direct - (k - 1)) / q_direct) * 100 if q_direct > 0 else 0

        # Tau-squared (DerSimonian-Laird)
        tau_sq = max(0, (q_direct - (k - 1)) / k)

        results = {
            "cochrans_q": float(q_direct),
            "q_p": float(q_p),
            "i_squared": float(i_squared),
            "tau_squared": float(tau_sq),
            "k": k,
            "mean_interaction": float(mean_int),
            "sd_interaction": float(np.std(interactions, ddof=1)),
        }
        print(f"  Q={q_direct:.2f}, p={q_p:.4f}, I²={i_squared:.1f}%, τ²={tau_sq:.4f}")

    with open(tables_dir / "heterogeneity_test.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _scale_moderation_meta(model_int_df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Pool exp-012, exp-013, exp-014, exp-015 data for scale-moderation analysis."""
    print("\n--- Scale-moderation meta-regression ---")

    all_model_data = []

    prior_paths = [
        ("exp-012", Path("analysis/orthogonality_decomposition_012/tables/raw_trials.csv")),
        ("exp-013", Path("analysis/orthogonality_grid_013/tables/raw_trials.csv")),
        ("exp-014", Path("analysis/orthogonality_scale_014/tables/raw_trials.csv")),
    ]

    for exp_id, csv_path in prior_paths:
        if csv_path.exists():
            prior_df = pd.read_csv(csv_path)
            # Normalize column names
            if "surgery" in prior_df.columns and "surgery_strength" not in prior_df.columns:
                prior_df["surgery_strength"] = (prior_df["surgery"] != "control").astype(float)
                prior_df["jitter_strength"] = (prior_df["jitter"] != "off").astype(float)
            for mn in prior_df["model_name"].unique():
                mdf = prior_df[prior_df["model_name"] == mn]
                paradigm = mdf["paradigm"].iloc[0]
                params_b = mdf["params_b"].iloc[0]
                acc = {}
                for (ss, js), grp in mdf.groupby(["surgery_strength", "jitter_strength"]):
                    key = ("surg" if ss > 0 else "ctrl", "jit" if js > 0 else "off")
                    if key not in acc:
                        acc[key] = []
                    acc[key].append(grp["correct"].mean())
                c00 = np.mean(acc.get(("ctrl", "off"), [np.nan]))
                c10 = np.mean(acc.get(("surg", "off"), [np.nan]))
                c01 = np.mean(acc.get(("ctrl", "jit"), [np.nan]))
                c11 = np.mean(acc.get(("surg", "jit"), [np.nan]))
                interaction = (c11 - c01) - (c10 - c00)
                if not np.isnan(interaction):
                    all_model_data.append({
                        "experiment": exp_id, "model_name": mn,
                        "paradigm": paradigm, "params_b": params_b,
                        "interaction": interaction,
                    })
            print(f"  Loaded {exp_id}: {len(prior_df['model_name'].unique())} models")
        else:
            print(f"  {exp_id} data not found at {csv_path}")

    # Add current experiment (exp-015) — only models NOT already in exp-014
    exp014_models = {d["model_name"] for d in all_model_data if d["experiment"] == "exp-014"}
    for _, row in model_int_df.iterrows():
        if row["model_name"] not in exp014_models:
            all_model_data.append({
                "experiment": "exp-015", "model_name": row["model_name"],
                "paradigm": row["paradigm"], "params_b": row["params_b"],
                "interaction": row["interaction"],
            })

    meta_df = pd.DataFrame(all_model_data)
    meta_df["log10_params"] = np.log10(meta_df["params_b"])
    meta_df.to_csv(tables_dir / "scale_meta_data.csv", index=False)

    results = {"n_experiments": len(meta_df["experiment"].unique()),
               "n_models_total": len(meta_df)}

    if len(meta_df) >= 5:
        slope, intercept, r, p, se = sp_stats.linregress(
            meta_df["log10_params"], meta_df["interaction"])
        results["scale_slope"] = float(slope)
        results["scale_intercept"] = float(intercept)
        results["scale_r"] = float(r)
        results["scale_p"] = float(p)
        results["scale_se"] = float(se)
        print(f"  Scale moderation: slope={slope:.4f}, r={r:.3f}, p={p:.4f}")
    else:
        print(f"  Insufficient data for meta-regression ({len(meta_df)} models)")

    with open(tables_dir / "scale_moderation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Figures ──────────────────────────────────────────────────────────────
PARADIGM_COLORS = {"transformer": "#2196F3", "ssm": "#4CAF50",
                   "hybrid": "#FF9800", "reasoning": "#FF5722",
                   "rwkv": "#9C27B0"}


def _plot_paradigm_forest(model_int_df: pd.DataFrame, fig_dir: Path):
    paradigm_data = []
    for paradigm in sorted(model_int_df["paradigm"].unique()):
        psub = model_int_df[model_int_df["paradigm"] == paradigm]
        mean = psub["interaction"].mean()
        se = psub["interaction"].std() / np.sqrt(len(psub)) if len(psub) > 1 else 0
        paradigm_data.append({"paradigm": paradigm, "mean": mean, "se": se, "n": len(psub)})

    pdf = pd.DataFrame(paradigm_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(pdf))
    colors = [PARADIGM_COLORS.get(p, "#666") for p in pdf["paradigm"]]

    ax.barh(list(y_pos), pdf["mean"], xerr=1.96 * pdf["se"],
            color=colors, alpha=0.8, capsize=5, height=0.6)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.axvspan(-SESOI_PROB, SESOI_PROB, alpha=0.2, color="green", label="SESOI")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{p} (n={n})" for p, n in zip(pdf["paradigm"], pdf["n"])])
    ax.set_xlabel("Mean Interaction Effect")
    ax.set_title("Exp-015: Interaction by Paradigm at 7B+ (95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_by_paradigm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_by_paradigm.png")


def _plot_model_forest(model_int_df: pd.DataFrame, fig_dir: Path):
    sorted_df = model_int_df.sort_values("interaction")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(sorted_df))
    colors = [PARADIGM_COLORS.get(p, "#666") for p in sorted_df["paradigm"]]

    ax.barh(list(y_pos), sorted_df["interaction"],
            color=colors, alpha=0.8, height=0.6)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.axvspan(-SESOI_PROB, SESOI_PROB, alpha=0.2, color="green", label="SESOI")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{mn} ({p})" for mn, p in
                        zip(sorted_df["model_name"], sorted_df["paradigm"])])
    ax.set_xlabel("Interaction Effect")
    ax.set_title("Exp-015: Per-Model Interaction at 7B+ Scale")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_by_model.png")


def _plot_scale_moderation(model_int_df: pd.DataFrame, scale_mod: Dict, fig_dir: Path):
    meta_csv = Path("analysis/orthogonality_cross_paradigm_015/tables/scale_meta_data.csv")
    if meta_csv.exists():
        meta_df = pd.read_csv(meta_csv)
    else:
        meta_df = model_int_df.copy()
        meta_df["experiment"] = "exp-015"

    fig, ax = plt.subplots(figsize=(10, 6))
    exp_markers = {"exp-012": "s", "exp-013": "^", "exp-014": "o", "exp-015": "D"}
    for exp_id in sorted(meta_df["experiment"].unique()):
        edf = meta_df[meta_df["experiment"] == exp_id]
        colors = [PARADIGM_COLORS.get(p, "#666") for p in edf["paradigm"]]
        marker = exp_markers.get(exp_id, "o")
        ax.scatter(np.log10(edf["params_b"]), edf["interaction"],
                   c=colors, marker=marker, s=80, alpha=0.8, label=exp_id,
                   edgecolors="black", linewidth=0.5)

    if "scale_slope" in scale_mod:
        x_range = np.linspace(meta_df["log10_params"].min() - 0.1,
                              meta_df["log10_params"].max() + 0.1, 100)
        y_pred = scale_mod["scale_slope"] * x_range + scale_mod["scale_intercept"]
        ax.plot(x_range, y_pred, "k--", alpha=0.5,
                label=f"slope={scale_mod['scale_slope']:.3f}, p={scale_mod['scale_p']:.3f}")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhspan(-SESOI_PROB, SESOI_PROB, alpha=0.15, color="green", label="SESOI")
    ax.set_xlabel("log10(Parameters, B)")
    ax.set_ylabel("Interaction Effect")
    ax.set_title("Scale Moderation: Pooled Exp-012/013/014/015")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "scale_moderation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> scale_moderation.png")


def _write_findings(df, model_int_df, mixed, boot, perm, loo, lopo,
                    split, reasoning_int, main_effects, scale_mod,
                    hetero, gate_pass, out_dir):
    lines = []
    lines.append(f"models_tested: {df['model_name'].nunique()}")
    lines.append(f"total_trials: {len(df)}")
    lines.append(f"paradigms: {', '.join(sorted(df['paradigm'].unique()))}")
    lines.append(f"domains: {', '.join(sorted(df['domain'].unique()))}")
    lines.append(f"layer_position: {LAYER_FRACTION}")
    lines.append(f"surgery_strength: {SURGERY_STRENGTH}")
    lines.append(f"jitter_strength: {JITTER_STRENGTH}")
    lines.append(f"seeds: {len(SEEDS)}")
    lines.append(f"prompts: {len(df['prompt_id'].unique())}")
    lines.append(f"completion_gate: {gate_pass}")

    mean_int = model_int_df["interaction"].mean()
    lines.append(f"global_mean_interaction: {mean_int:.5f}")

    if "interaction_beta" in mixed:
        lines.append(f"mixed_model_interaction_beta: {mixed['interaction_beta']:.4f}")
        lines.append(f"mixed_model_interaction_p: {mixed['interaction_p']:.4f}")
        lines.append(f"mixed_model_interaction_or: {mixed['interaction_or']:.4f}")

    lines.append(f"bootstrap_ci: [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")
    lines.append(f"bootstrap_ci_includes_zero: {boot['ci_includes_zero']}")
    lines.append(f"sesoi_rope_fraction: {boot['sesoi_rope_fraction']:.3f}")
    lines.append(f"sesoi_bound: {SESOI_PROB}")
    lines.append(f"permutation_p: {perm['p_perm']:.4f}")
    lines.append(f"loo_std: {loo['loo_std']:.4f}")
    lines.append(f"lopo_std: {lopo['lopo_std']:.4f}")
    lines.append(f"calibration_interaction: {split['calibration_interaction']:.4f}")
    lines.append(f"holdout_interaction: {split['holdout_interaction']:.4f}")
    lines.append(f"cal_holdout_gap_pp: {split['gap_pp']:.1f}")

    if not np.isnan(reasoning_int):
        lines.append(f"reasoning_only_interaction: {reasoning_int:.4f}")

    lines.append(f"baseline_accuracy: {main_effects['baseline_accuracy']:.3f}")
    lines.append(f"surgery_main_effect: {main_effects['surgery_main_effect']:+.3f}")
    lines.append(f"jitter_main_effect: {main_effects['jitter_main_effect']:+.3f}")

    # Heterogeneity
    if "cochrans_q" in hetero and not np.isnan(hetero.get("cochrans_q", np.nan)):
        lines.append(f"cochrans_q: {hetero['cochrans_q']:.2f}")
        lines.append(f"cochrans_q_p: {hetero['q_p']:.4f}")
        lines.append(f"i_squared: {hetero['i_squared']:.1f}")
        lines.append(f"tau_squared: {hetero['tau_squared']:.4f}")

    for _, row in model_int_df.iterrows():
        lines.append(f"model_{row['model_name']}_interaction: {row['interaction']:.4f}")

    for p in sorted(model_int_df["paradigm"].unique()):
        psub = model_int_df[model_int_df["paradigm"] == p]
        lines.append(f"paradigm_{p}_interaction: {psub['interaction'].mean():.4f}")

    if "scale_slope" in scale_mod:
        lines.append(f"scale_moderation_slope: {scale_mod['scale_slope']:.4f}")
        lines.append(f"scale_moderation_p: {scale_mod['scale_p']:.4f}")
        lines.append(f"scale_moderation_r: {scale_mod['scale_r']:.3f}")

    # Verdict using Codex decision rules
    transformer_int = model_int_df[model_int_df["paradigm"] == "transformer"]["interaction"].mean()
    non_transformer_int = model_int_df[model_int_df["paradigm"] != "transformer"]["interaction"]
    has_non_transformer = len(non_transformer_int) > 0

    orthogonal = boot["ci_includes_zero"] and perm["p_perm"] > 0.05 and boot["sesoi_rope_fraction"] > 0.5
    transformer_specific = (not orthogonal and has_non_transformer and
                           abs(non_transformer_int.mean()) < SESOI_PROB and
                           transformer_int < -SESOI_PROB)

    if orthogonal:
        verdict = "ORTHOGONALITY_RESTORED (exp-014 was false positive)"
    elif transformer_specific:
        verdict = "TRANSFORMER_SPECIFIC_COUPLING (not universal scale law)"
    else:
        verdict = "INTERACTION_CONFIRMED_AT_SCALE"

    lines.append(f"verdict: {verdict}")

    if verdict == "ORTHOGONALITY_RESTORED (exp-014 was false positive)":
        lines.append(
            "business_implication: Adding SSM/hybrid models restored orthogonality at 7B+. "
            "The exp-014 interaction was a false positive from limited paradigm coverage. "
            "Independent deployment testing validated across scale and paradigms."
        )
        lines.append(
            "scientific_implication: Cross-paradigm replication at 7B+ shows the exp-014 "
            "interaction was driven by insufficient model diversity. With SSM/hybrid coverage, "
            "orthogonality between geometric and dynamic axes holds universally."
        )
    elif verdict == "TRANSFORMER_SPECIFIC_COUPLING (not universal scale law)":
        lines.append(
            "business_implication: The surgery-jitter interaction is transformer-specific, not "
            "a universal scale law. SSM/hybrid models maintain orthogonality at 7B+. "
            "Joint testing required only for large transformers; other paradigms can test independently."
        )
        lines.append(
            "scientific_implication: Cross-paradigm resolution reveals architecture-specific "
            "heterogeneity: large transformers couple geometric and dynamic perturbations, "
            "while SSM/hybrid/reasoning models maintain independence. This is paradigm-specific "
            "coupling, not a general scale collapse of orthogonality."
        )
    else:
        lines.append(
            "business_implication: Interaction confirmed at 7B+ scale across paradigms. "
            "Orthogonality breaks are not transformer-specific. Joint geometric-dynamic "
            "validation required for all deployment-scale models."
        )
        lines.append(
            "scientific_implication: Cross-paradigm replication confirms scale-dependent "
            "coupling between representation geometry and dynamic stability. The interaction "
            "is not limited to transformers — it is a universal property of large models."
        )

    (out_dir / "key_findings.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFindings written. Verdict: {verdict}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp-015: 7B+ Cross-Paradigm Orthogonality Resolution")
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--outdir", default="analysis/orthogonality_cross_paradigm_015")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("collect", "all"):
        collect_data(out_dir)

    if args.stage in ("analyze", "all"):
        run_analysis(out_dir)


if __name__ == "__main__":
    main()
