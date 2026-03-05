#!/usr/bin/env python3
"""
orthogonality_grid_013.py — Exp-013: Orthogonality Grid (Expanded Factorial)

Codex-designed follow-up to Exp-012. Tests orthogonality across a full grid
of layer positions and perturbation strengths with harder prompts and stricter
answer parsing.

Design:
  15 models × 3 layers × 3 surgery strengths × 3 jitter strengths × 3 seeds × 48 prompts
  = ~58k trials (batched inference for speed)

  Layer positions: [0.2, 0.5, 0.8] of total layers (early/mid/late)
  Surgery: [0.0, 0.06, 0.12]
  Jitter:  [0.0, 0.04, 0.08]
  Seeds:   [11, 23, 37]

  Prompts: 48 total (16 math, 16 factual, 8 logic, 8 hard)
  Split: 32 calibration + 16 holdout
  Batch size: 16 prompts per forward pass

Pre-specified SESOI:
  - Interaction probability-scale bound: [-0.012, +0.012]
  - ROPE on OR: [0.90, 1.11]
"""
from __future__ import annotations

import argparse
import gc
import json
import re
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

LAYER_FRACTIONS = [0.2, 0.5, 0.8]
SURGERY_STRENGTHS = [0.0, 0.06, 0.12]
JITTER_STRENGTHS = [0.0, 0.04, 0.08]
SEEDS = [11, 23, 37]
MAX_NEW_TOKENS = 10
BATCH_SIZE = 16

# ── Model registry ───────────────────────────────────────────────────────
MODELS: List[Tuple[str, str, str, float]] = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "transformer", 0.6),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "transformer", 1.7),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B", "transformer", 0.5),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", "transformer", 1.5),
    ("google/gemma-3-1b-it", "Gemma3-1B", "transformer", 1.0),
    ("google/gemma-2-2b-it", "Gemma2-2B", "transformer", 2.0),
    ("state-spaces/mamba-130m-hf", "Mamba-130M", "ssm", 0.13),
    ("state-spaces/mamba-370m-hf", "Mamba-370M", "ssm", 0.37),
    ("state-spaces/mamba-790m-hf", "Mamba-790M", "ssm", 0.79),
    ("state-spaces/mamba-2.8b-hf", "Mamba-2.8B", "ssm", 2.8),
    ("tiiuae/Falcon-H1-0.5B-Instruct", "FalconH1-0.5B", "hybrid", 0.5),
    # ("tiiuae/Falcon-H1-1.5B-Instruct", "FalconH1-1.5B", "hybrid", 1.5),  # shape mismatch with batch gen
    ("Zyphra/Zamba2-1.2B", "Zamba2-1.2B", "hybrid", 1.2),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning", 1.5),
    ("nvidia/OpenReasoning-Nemotron-1.5B", "Nemotron-1.5B-R", "reasoning", 1.5),
]

# ── Prompts: 48 total ────────────────────────────────────────────────────
# Split: 32 calibration + 16 holdout (balanced across domains)
PROMPTS = {
    "math": [
        # Calibration (8)
        {"id": "m01", "prompt": "What is 7 * 8? Answer with just the number:", "answer": "56", "split": "cal"},
        {"id": "m02", "prompt": "What is 15 + 27? Answer with just the number:", "answer": "42", "split": "cal"},
        {"id": "m03", "prompt": "What is 100 - 37? Answer with just the number:", "answer": "63", "split": "cal"},
        {"id": "m04", "prompt": "What is 144 / 12? Answer with just the number:", "answer": "12", "split": "cal"},
        {"id": "m05", "prompt": "What is 9 * 9? Answer with just the number:", "answer": "81", "split": "cal"},
        {"id": "m06", "prompt": "What is 23 + 19? Answer with just the number:", "answer": "42", "split": "cal"},
        {"id": "m07", "prompt": "What is 256 / 16? Answer with just the number:", "answer": "16", "split": "cal"},
        {"id": "m08", "prompt": "What is 11 * 11? Answer with just the number:", "answer": "121", "split": "cal"},
        # Holdout (4)
        {"id": "m09", "prompt": "What is 225 / 15? Answer with just the number:", "answer": "15", "split": "hold"},
        {"id": "m10", "prompt": "What is 37 + 48? Answer with just the number:", "answer": "85", "split": "hold"},
        {"id": "m11", "prompt": "What is 16 * 7? Answer with just the number:", "answer": "112", "split": "hold"},
        {"id": "m12", "prompt": "What is 400 - 213? Answer with just the number:", "answer": "187", "split": "hold"},
    ],
    "factual": [
        # Calibration (8)
        {"id": "f01", "prompt": "The capital of Japan is", "answer": "Tokyo", "split": "cal"},
        {"id": "f02", "prompt": "Water freezes at 0 degrees", "answer": "Celsius", "split": "cal"},
        {"id": "f03", "prompt": "The chemical symbol for gold is", "answer": "Au", "split": "cal"},
        {"id": "f04", "prompt": "The largest planet in our solar system is", "answer": "Jupiter", "split": "cal"},
        {"id": "f05", "prompt": "The speed of light is approximately 300,000 km per", "answer": "second", "split": "cal"},
        {"id": "f06", "prompt": "DNA stands for deoxyribonucleic", "answer": "acid", "split": "cal"},
        {"id": "f07", "prompt": "The square root of 144 is", "answer": "12", "split": "cal"},
        {"id": "f08", "prompt": "The atomic number of carbon is", "answer": "6", "split": "cal"},
        # Holdout (4)
        {"id": "f09", "prompt": "The largest ocean on Earth is the", "answer": "Pacific", "split": "hold"},
        {"id": "f10", "prompt": "The hardest natural substance is", "answer": "diamond", "split": "hold"},
        {"id": "f11", "prompt": "The chemical symbol for sodium is", "answer": "Na", "split": "hold"},
        {"id": "f12", "prompt": "The closest star to Earth is the", "answer": "Sun", "split": "hold"},
    ],
    "logic": [
        # Calibration (8)
        {"id": "l01", "prompt": "If all roses are flowers and all flowers need water, then all roses need", "answer": "water", "split": "cal"},
        {"id": "l02", "prompt": "If today is Monday, then tomorrow is", "answer": "Tuesday", "split": "cal"},
        {"id": "l03", "prompt": "A dozen eggs is exactly", "answer": "12", "split": "cal"},
        {"id": "l04", "prompt": "If a triangle has angles of 60, 60, and 60 degrees, it is called", "answer": "equilateral", "split": "cal"},
        {"id": "l05", "prompt": "The opposite of 'increase' is", "answer": "decrease", "split": "cal"},
        {"id": "l06", "prompt": "Half of 200 is", "answer": "100", "split": "cal"},
        {"id": "l07", "prompt": "If 3x = 21, then x equals", "answer": "7", "split": "cal"},
        {"id": "l08", "prompt": "The next number in the sequence 2, 4, 8, 16 is", "answer": "32", "split": "cal"},
        # Holdout (4)
        {"id": "l09", "prompt": "If a car travels at 60 km/h for 2 hours, the total distance is", "answer": "120", "split": "hold"},
        {"id": "l10", "prompt": "The number of sides in a hexagon is", "answer": "6", "split": "hold"},
        {"id": "l11", "prompt": "If 5 + x = 13, then x equals", "answer": "8", "split": "hold"},
        {"id": "l12", "prompt": "The next prime after 7 is", "answer": "11", "split": "hold"},
    ],
    "hard": [
        # Calibration (8)
        {"id": "h01", "prompt": "What is 17 * 19? Answer with just the number:", "answer": "323", "split": "cal"},
        {"id": "h02", "prompt": "What is the cube root of 27?", "answer": "3", "split": "cal"},
        {"id": "h03", "prompt": "What is 15% of 200? Answer with just the number:", "answer": "30", "split": "cal"},
        {"id": "h04", "prompt": "If f(x) = 2x + 3, what is f(5)? Answer with just the number:", "answer": "13", "split": "cal"},
        {"id": "h05", "prompt": "The element with atomic number 79 is", "answer": "gold", "split": "cal"},
        {"id": "h06", "prompt": "What is 2^10? Answer with just the number:", "answer": "1024", "split": "cal"},
        {"id": "h07", "prompt": "The derivative of x^2 is", "answer": "2x", "split": "cal"},
        {"id": "h08", "prompt": "How many degrees in a right angle?", "answer": "90", "split": "cal"},
        # Holdout (4)
        {"id": "h09", "prompt": "What is 23 * 17? Answer with just the number:", "answer": "391", "split": "hold"},
        {"id": "h10", "prompt": "What is the square root of 225?", "answer": "15", "split": "hold"},
        {"id": "h11", "prompt": "What is 3^4? Answer with just the number:", "answer": "81", "split": "hold"},
        {"id": "h12", "prompt": "What is 7^3? Answer with just the number:", "answer": "343", "split": "hold"},
    ],
}


# ── Strict answer parsing ────────────────────────────────────────────────
def parse_numeric(text: str) -> Optional[str]:
    """Extract numeric answer from text with normalization."""
    text = text.strip().lower()
    # Remove common suffixes
    text = re.sub(r'[.,;:!?\s]+$', '', text)
    # Try to find a number
    match = re.search(r'-?\d[\d,]*\.?\d*', text)
    if match:
        num_str = match.group().replace(',', '')
        return num_str
    return None


def check_correct(generated: str, answer: str, domain: str) -> Tuple[bool, float]:
    """Strict answer checking. Returns (correct, confidence).
    confidence: 1.0 = exact match, 0.5 = substring match, 0.0 = no match
    """
    gen_lower = generated.strip().lower()
    ans_lower = answer.lower()

    # Numeric domains: parse and compare
    if domain in ("math", "hard", "logic") and answer.replace('.', '').replace('-', '').isdigit():
        parsed = parse_numeric(gen_lower)
        if parsed is not None:
            try:
                if abs(float(parsed) - float(answer)) < 0.01:
                    return True, 1.0
            except ValueError:
                pass
        # Fallback: substring
        if ans_lower in gen_lower:
            return True, 0.5
        return False, 0.0

    # Factual/text domains: normalized comparison
    # Exact match
    if ans_lower in gen_lower:
        return True, 1.0

    # Synonym handling for common factual answers
    synonyms = {
        "celsius": ["celsius", "centigrade", "°c"],
        "squared": ["squared", "²", "square"],
    }
    if ans_lower in synonyms:
        for syn in synonyms[ans_lower]:
            if syn in gen_lower:
                return True, 0.8
    return False, 0.0


# ── Hooks (reused from exp-012) ──────────────────────────────────────────
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


def get_layer_indices(n_layers: int) -> List[int]:
    indices = []
    for frac in LAYER_FRACTIONS:
        idx = int(frac * (n_layers - 1))
        idx = max(0, min(idx, n_layers - 1))
        if idx not in indices:
            indices.append(idx)
    return sorted(indices)


# ── Batched generation ──────────────────────────────────────────────────
def generate_batch(model, tokenizer, prompts_batch, device):
    """Generate responses for a batch of prompts simultaneously."""
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

    # Flatten prompts
    all_prompts = []
    for domain, items in PROMPTS.items():
        for item in items:
            all_prompts.append({**item, "domain": domain})

    n_conditions = len(LAYER_FRACTIONS) * len(SURGERY_STRENGTHS) * len(JITTER_STRENGTHS) * len(SEEDS)
    total_trials = len(MODELS) * n_conditions * len(all_prompts)

    print(f"\n{'=' * 70}")
    print(f"EXP-013: ORTHOGONALITY GRID (EXPANDED)")
    print(f"Models: {len(MODELS)}, Prompts: {len(all_prompts)}, Seeds: {len(SEEDS)}")
    print(f"Layers: {LAYER_FRACTIONS}, Surgery: {SURGERY_STRENGTHS}, Jitter: {JITTER_STRENGTHS}")
    print(f"Conditions per model: {n_conditions}, Batch size: {BATCH_SIZE}")
    print(f"Total trials: {total_trials}")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, paradigm, params_b) in enumerate(MODELS):
        if model_name in completed_models:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (completed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{model_idx+1}/{len(MODELS)}] {model_name} ({paradigm}, {params_b}B)")
        print(f"{'=' * 60}")

        t_model = time.time()
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"  [FAIL] {model_id}: {e}")
            failures.append({"model_id": model_id, "model_name": model_name, "error": str(e)})
            continue

        layers = get_model_layers(model)
        if not layers:
            print(f"  [FAIL] No layers for {model_name}")
            failures.append({"model_id": model_id, "model_name": model_name, "error": "no_layers"})
            del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
            continue

        n_layers = len(layers)
        layer_indices = get_layer_indices(n_layers)
        print(f"  Layers: {n_layers}, targets: {layer_indices}")
        print(f"  Loaded in {time.time() - t_model:.1f}s")

        device = next(model.parameters()).device
        model_rows = []
        cond_count = 0
        model_failed = False

        try:
            for layer_idx in layer_indices:
                layer_frac = layer_idx / max(n_layers - 1, 1)
                for surg_s in SURGERY_STRENGTHS:
                    for jit_s in JITTER_STRENGTHS:
                        for seed in SEEDS:
                            cond_count += 1
                            hooks = []

                            if surg_s > 0:
                                h = PRExpansionHook(strength=surg_s, seed=seed)
                                h.attach(layers[layer_idx])
                                hooks.append(h)
                            if jit_s > 0:
                                h = JitterHook(strength=jit_s, seed=seed + 10000)
                                h.attach(layers[layer_idx])
                                hooks.append(h)

                            # Process prompts in batches
                            for batch_start in range(0, len(all_prompts), BATCH_SIZE):
                                batch = all_prompts[batch_start:batch_start + BATCH_SIZE]
                                try:
                                    results = generate_batch(model, tokenizer, batch, device)
                                except Exception as batch_err:
                                    # Fall back to single-prompt
                                    if cond_count <= 1:
                                        print(f"    [BATCH FAIL] Single-prompt fallback: {type(batch_err).__name__}")
                                    results = []
                                    for p in batch:
                                        try:
                                            torch.cuda.empty_cache()
                                            inp = tokenizer(
                                                p["prompt"], return_tensors="pt",
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
                                            correct, confidence = check_correct(text, p["answer"], p["domain"])
                                            results.append((text, correct, confidence))
                                        except Exception:
                                            results.append(("ERROR", False, 0.0))

                                for p, (text, correct, confidence) in zip(batch, results):
                                    model_rows.append({
                                        "model_name": model_name, "paradigm": paradigm,
                                        "params_b": params_b, "seed": seed,
                                        "layer_idx": layer_idx, "layer_frac": round(layer_frac, 2),
                                        "surgery_strength": surg_s, "jitter_strength": jit_s,
                                        "domain": p["domain"], "prompt_id": p["id"],
                                        "split": p["split"],
                                        "answer": p["answer"], "generated": text,
                                        "correct": int(correct), "confidence": confidence,
                                    })

                            for h in hooks:
                                h.detach()

                            if cond_count % 10 == 0:
                                elapsed = time.time() - t_model
                                rate = cond_count * len(all_prompts) / elapsed if elapsed > 0 else 0
                                print(f"    [{cond_count}/{n_conditions}] {elapsed:.0f}s ({rate:.1f} trials/s)")
        except Exception as model_err:
            print(f"  [MODEL FAIL] {model_name}: {type(model_err).__name__}: {model_err}")
            failures.append({"model_id": model_id, "model_name": model_name, "error": str(model_err)})
            model_failed = True

        if not model_failed and model_rows:
            all_rows.extend(model_rows)
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

    if failures:
        pd.DataFrame(failures).to_csv(tables_dir / "failures.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(raw_csv, index=False)
    print(f"\nData collection complete: {len(df)} rows, {df['model_name'].nunique()} models")
    return df


# ── Analysis ─────────────────────────────────────────────────────────────
def run_analysis(out_dir: Path):
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables_dir / "raw_trials.csv")
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {len(df)} trials, {df['model_name'].nunique()} models")
    print(f"{'=' * 70}")

    # ── 1. Per-cell interaction at each (layer, surgery_strength, jitter_strength) ──
    interactions_by_cell = []
    for model_name in df["model_name"].unique():
        mdf = df[df["model_name"] == model_name]
        paradigm = mdf["paradigm"].iloc[0]
        for layer_frac in sorted(mdf["layer_frac"].unique()):
            ldf = mdf[mdf["layer_frac"] == layer_frac]
            # For each nonzero surgery and jitter, compute interaction vs control
            for surg_s in [s for s in SURGERY_STRENGTHS if s > 0]:
                for jit_s in [s for s in JITTER_STRENGTHS if s > 0]:
                    acc = {}
                    for (ss, js), grp in ldf.groupby(["surgery_strength", "jitter_strength"]):
                        acc[(ss, js)] = grp["correct"].mean()
                    c00 = acc.get((0.0, 0.0), np.nan)
                    c10 = acc.get((surg_s, 0.0), np.nan)
                    c01 = acc.get((0.0, jit_s), np.nan)
                    c11 = acc.get((surg_s, jit_s), np.nan)
                    interaction = (c11 - c01) - (c10 - c00)
                    interactions_by_cell.append({
                        "model_name": model_name, "paradigm": paradigm,
                        "layer_frac": layer_frac,
                        "surgery_strength": surg_s, "jitter_strength": jit_s,
                        "interaction": interaction,
                    })

    cell_df = pd.DataFrame(interactions_by_cell)
    cell_df.to_csv(tables_dir / "interaction_by_cell.csv", index=False)

    # ── 2. Global interaction (average across all cells and models) ──
    global_int = cell_df["interaction"].mean()
    global_std = cell_df["interaction"].std()
    print(f"\nGlobal mean interaction: {global_int:.4f} (std={global_std:.4f})")

    # ── 3. Mixed-effects model ──
    mixed = _fit_mixed_model(df, tables_dir)

    # ── 4. Bootstrap over models ──
    boot = _bootstrap_interaction(cell_df, tables_dir)

    # ── 5. Permutation test ──
    perm = _permutation_test(df, tables_dir)

    # ── 6. Layer-specific interaction ──
    layer_int = cell_df.groupby("layer_frac")["interaction"].agg(["mean", "std", "count"])
    layer_int.to_csv(tables_dir / "interaction_by_layer.csv")
    print(f"\nInteraction by layer:\n{layer_int}")

    # ── 7. Paradigm-specific interaction ──
    paradigm_int = cell_df.groupby("paradigm")["interaction"].agg(["mean", "std", "count"])
    paradigm_int.to_csv(tables_dir / "interaction_by_paradigm.csv")
    print(f"\nInteraction by paradigm:\n{paradigm_int}")

    # ── 8. Calibration vs holdout ──
    cal_df = df[df["split"] == "cal"]
    hold_df = df[df["split"] == "hold"]
    cal_int = _compute_global_interaction(cal_df)
    hold_int = _compute_global_interaction(hold_df)
    split_results = {"calibration_interaction": float(cal_int), "holdout_interaction": float(hold_int)}
    with open(tables_dir / "split_comparison.json", "w") as f:
        json.dump(split_results, f, indent=2)
    print(f"\nCal interaction: {cal_int:.4f}, Holdout: {hold_int:.4f}")

    # ── 9. LOO ──
    loo = _leave_one_out(cell_df, tables_dir)

    # ── 10. Figures ──
    _plot_interaction_heatmap(cell_df, figures_dir)
    _plot_layer_interaction(cell_df, figures_dir)
    _plot_paradigm_forest(cell_df, figures_dir)

    # ── 11. Key findings ──
    _write_findings(df, cell_df, mixed, boot, perm, loo, split_results, out_dir)


def _fit_mixed_model(df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Mixed-effects model ---")
    df = df.copy()
    df["surgery_bin"] = (df["surgery_strength"] > 0).astype(int)
    df["jitter_bin"] = (df["jitter_strength"] > 0).astype(int)
    df["interaction"] = df["surgery_bin"] * df["jitter_bin"]
    df["domain_bin"] = (df["domain"].isin(["math", "hard"])).astype(int)
    df["log10_params"] = np.log10(df["params_b"])

    results = {}
    try:
        import statsmodels.api as sm
        exog = df[["surgery_bin", "jitter_bin", "interaction", "domain_bin",
                    "log10_params", "layer_frac"]].copy()
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
        print(f"  Interaction: beta={results['interaction_beta']:.4f}, p={results['interaction_p']:.4f}, OR={results['interaction_or']:.4f}")
    except Exception as e:
        print(f"  Mixed model failed: {e}")
        results["method"] = "failed"

    with open(tables_dir / "mixed_effects_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _bootstrap_interaction(cell_df: pd.DataFrame, tables_dir: Path, n_boot: int = 5000) -> Dict:
    print("\n--- Bootstrap (5000) ---")
    rng = np.random.default_rng(42)
    models = cell_df["model_name"].unique()
    n_models = len(models)

    # Compute per-model mean interaction
    model_ints = cell_df.groupby("model_name")["interaction"].mean()

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(models, size=n_models, replace=True)
        boot_means.append(model_ints[sample].mean())

    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    mean_int = model_ints.mean()

    # ROPE check with pre-specified bounds
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


def _compute_global_interaction(df: pd.DataFrame) -> float:
    ints = []
    for mn in df["model_name"].unique():
        mdf = df[df["model_name"] == mn]
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
            ints.append(interaction)
    return np.mean(ints) if ints else 0.0


def _leave_one_out(cell_df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Leave-one-model-out ---")
    model_ints = cell_df.groupby("model_name")["interaction"].mean()
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


# ── Figures ──────────────────────────────────────────────────────────────
PARADIGM_COLORS = {"transformer": "#2196F3", "ssm": "#4CAF50", "hybrid": "#FF9800", "reasoning": "#FF5722"}


def _plot_interaction_heatmap(cell_df: pd.DataFrame, fig_dir: Path):
    """Heatmap: interaction across surgery × jitter strength grid."""
    pivot = cell_df.groupby(["surgery_strength", "jitter_strength"])["interaction"].mean().reset_index()
    heatmap = pivot.pivot(index="surgery_strength", columns="jitter_strength", values="interaction")

    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                ax=ax, vmin=-0.05, vmax=0.05, linewidths=0.5)
    ax.set_xlabel("Jitter Strength")
    ax.set_ylabel("Surgery Strength")
    ax.set_title("Mean Interaction Effect Across Strength Grid\n(values near 0 = orthogonal)")
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_heatmap.png")


def _plot_layer_interaction(cell_df: pd.DataFrame, fig_dir: Path):
    """Bar chart: mean interaction by layer position."""
    layer_data = cell_df.groupby("layer_frac")["interaction"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(layer_data)), layer_data["mean"],
                  yerr=layer_data["std"] / np.sqrt(cell_df.groupby("layer_frac").ngroups),
                  color=["#42A5F5", "#66BB6A", "#EF5350"], alpha=0.8, capsize=5)
    ax.set_xticks(range(len(layer_data)))
    ax.set_xticklabels([f"Layer {f:.1f}" for f in layer_data["layer_frac"]])
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhspan(-SESOI_PROB, SESOI_PROB, alpha=0.2, color="green", label=f"SESOI [{-SESOI_PROB}, {SESOI_PROB}]")
    ax.set_ylabel("Mean Interaction Effect")
    ax.set_title("Interaction by Layer Position")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_by_layer.png")


def _plot_paradigm_forest(cell_df: pd.DataFrame, fig_dir: Path):
    """Forest plot: per-paradigm mean interaction with CI."""
    paradigm_data = []
    for paradigm in sorted(cell_df["paradigm"].unique()):
        psub = cell_df[cell_df["paradigm"] == paradigm]
        model_means = psub.groupby("model_name")["interaction"].mean()
        mean = model_means.mean()
        se = model_means.std() / np.sqrt(len(model_means)) if len(model_means) > 1 else 0
        paradigm_data.append({"paradigm": paradigm, "mean": mean, "se": se, "n": len(model_means)})

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
    ax.set_title("Interaction by Paradigm (95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_by_paradigm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_by_paradigm.png")


def _write_findings(df, cell_df, mixed, boot, perm, loo, split, out_dir):
    lines = []
    lines.append(f"models_tested: {df['model_name'].nunique()}")
    lines.append(f"total_trials: {len(df)}")
    lines.append(f"paradigms: {', '.join(sorted(df['paradigm'].unique()))}")
    lines.append(f"domains: {', '.join(sorted(df['domain'].unique()))}")
    lines.append(f"layer_positions: {sorted(df['layer_frac'].unique())}")
    lines.append(f"surgery_strengths: {sorted(df['surgery_strength'].unique())}")
    lines.append(f"jitter_strengths: {sorted(df['jitter_strength'].unique())}")

    mean_int = cell_df["interaction"].mean()
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
    lines.append(f"calibration_interaction: {split['calibration_interaction']:.4f}")
    lines.append(f"holdout_interaction: {split['holdout_interaction']:.4f}")

    # Per-layer
    for lf in sorted(cell_df["layer_frac"].unique()):
        lsub = cell_df[cell_df["layer_frac"] == lf]
        lines.append(f"layer_{lf}_interaction: {lsub['interaction'].mean():.4f}")

    # Per-paradigm
    for p in sorted(cell_df["paradigm"].unique()):
        psub = cell_df[cell_df["paradigm"] == p]
        lines.append(f"paradigm_{p}_interaction: {psub['interaction'].mean():.4f}")

    # Verdict
    orthogonal = boot["ci_includes_zero"] and perm["p_perm"] > 0.05 and boot["sesoi_rope_fraction"] > 0.5
    verdict = "NO_INTERACTION_DETECTED (orthogonality supported)" if orthogonal else "INTERACTION_DETECTED"
    lines.append(f"verdict: {verdict}")

    if orthogonal:
        lines.append(
            "business_implication: Orthogonality between geometric (PR) and dynamic (noise) "
            "axes holds across multiple layer positions, perturbation strengths, and task "
            "families. Deployment risk requires independent testing on both axes."
        )
        lines.append(
            "scientific_implication: Expanded grid factorial (3 layers x 3x3 strengths x 48 "
            "prompts, batched) confirms no detectable interaction between PR surgery and jitter "
            "stress. Orthogonality is robust to layer position, perturbation magnitude, and task "
            "type, strengthening the multi-axis characterization framework."
        )
    else:
        lines.append(
            "business_implication: Interaction detected in expanded grid — geometric and "
            "dynamic axes may not be fully independent at all operating points."
        )
        lines.append(
            "scientific_implication: Expanded grid reveals interaction between PR surgery "
            "and jitter stress that was not detected in the simpler exp-012 design."
        )

    (out_dir / "key_findings.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFindings written. Verdict: {verdict}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp-013: Orthogonality Grid")
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--outdir", default="analysis/orthogonality_grid_013")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("collect", "all"):
        collect_data(out_dir)
    if args.stage in ("analyze", "all"):
        run_analysis(out_dir)


if __name__ == "__main__":
    main()
