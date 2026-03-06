#!/usr/bin/env python3
"""
orthogonality_submodule_017.py — Exp-017: Submodule Locus of 7B+ Coupling

Codex-directed follow-up to Exp-016. Exp-016 ruled out layer localization,
training regularization, and geometric prediction as mechanisms for the 7B+
surgery x jitter coupling. This experiment tests WHERE in the transformer block
the coupling lives: attention path, MLP path, or residual integration.

Design:
  4 models x 2 layers x 3 targets x 2x2 factorial x 5 seeds x 64 prompts
  = 30,720 eval trials + branch diagnostics

  Models (all cached from exp-016, excluding DSR1-Llama-8B):
    base_transformer: Qwen2.5-7B, Qwen3-8B, OLMo2-7B
    reasoning_tuned: DSR1-7B

  Layer positions: [0.50 (primary), 0.80 (confirmation)]
  Hook targets: [block, self_attn, mlp]
  Fixed strengths: surgery=0.08, jitter=0.08
  Seeds: [11, 23, 37, 43, 59]
  Prompts: Same 64 from exp-014/015/016

  Phase 1: Factorial evaluation across targets (30,720 trials)
  Phase 2: Branch diagnostics (clean pass: norms, cosine alignment)
  Phase 3: Analysis (wild-cluster bootstrap, decision rules)

Hypotheses (pre-registered):
  H1: The coupling is carried by the attention path (self_attn)
  H2: The coupling is carried by the MLP path
  H3: The coupling arises from residual integration (neither branch alone)

Decision rules (Codex-specified):
  - self_attn negative + mlp in ROPE -> ATTENTION_PATH_COUPLING
  - block negative + both subbranches weak -> RESIDUAL_STREAM_INTEGRATION
  - both subbranches negative -> GENERIC_BLOCK_COUPLING
  - no clean separation -> STOP_MECHANISM_CHASING

Inference:
  - Primary: wild-cluster bootstrap (exact enumeration, 2^4=16 sign combos)
  - CIs: model-clustered bootstrap (5000 resamples)
  - Sensitivity: within-model permutation (5000 shuffles)
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
ROPE_OR = (0.90, 1.11)

LAYER_FRACTIONS = [0.50, 0.80]  # mid (primary) + late (confirmation)
HOOK_TARGETS = ["block", "self_attn", "mlp"]
SURGERY_STRENGTH = 0.08
JITTER_STRENGTH = 0.08
SEEDS = [11, 23, 37, 43, 59]
MAX_NEW_TOKENS = 10

BATCH_SIZES = {"base_transformer": 4, "reasoning_tuned": 1}

# ── Model registry (4 models from exp-016, excluding DSR1-Llama-8B) ─────
MODELS: List[Tuple[str, str, str, float]] = [
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B", "base_transformer", 7.0),
    ("Qwen/Qwen3-8B", "Qwen3-8B", "base_transformer", 8.0),
    ("allenai/OLMo-2-1124-7B", "OLMo2-7B", "base_transformer", 7.0),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DSR1-7B", "reasoning_tuned", 7.0),
]

# ── Prompts: 64 total (32 cal + 32 holdout) — identical to exp-014/015/016
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


# ── Answer parsing (identical to exp-014/015/016) ─────────────────────────
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


# ── Hooks (identical to exp-012-016) ──────────────────────────────────────
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

    def attach(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

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

    def attach(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


class BranchCaptureHook:
    """Captures branch output norms and vectors for diagnostics."""
    def __init__(self):
        self.norms = []
        self.vectors = []  # last-token vectors for cosine alignment
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # L2 norm of last token
        last_tok = hidden[:, -1, :].detach().float()
        self.norms.append(last_tok.norm(dim=-1).cpu())
        self.vectors.append(last_tok.cpu())

    def attach(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    def get_and_clear(self):
        norms = torch.cat(self.norms) if self.norms else torch.tensor([])
        vectors = torch.cat(self.vectors) if self.vectors else torch.tensor([])
        self.norms = []
        self.vectors = []
        return norms, vectors


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


def get_layer_idx(n_layers: int, fraction: float) -> int:
    return int(fraction * (n_layers - 1))


def get_target_module(layer_module: nn.Module, target: str) -> nn.Module:
    """Get the submodule to hook based on the target name."""
    if target == "block":
        return layer_module
    elif target == "self_attn":
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer_module, attr):
                return getattr(layer_module, attr)
        raise ValueError(f"No attention submodule in {type(layer_module).__name__}")
    elif target == "mlp":
        for attr in ["mlp", "feed_forward", "ffn"]:
            if hasattr(layer_module, attr):
                return getattr(layer_module, attr)
        raise ValueError(f"No MLP submodule in {type(layer_module).__name__}")
    raise ValueError(f"Unknown target: {target}")


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

    for p in prompts:
        try:
            torch.cuda.empty_cache()
            text, correct, confidence = generate_single(model, tokenizer, p, device)
            all_results.append((p, (text, correct, confidence)))
        except Exception:
            all_results.append((p, ("ERROR", False, 0.0)))
    return all_results


# ── Phase 1: Factorial evaluation across targets ─────────────────────────
def collect_factorial(out_dir: Path) -> pd.DataFrame:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = tables_dir / "raw_trials.csv"

    all_rows = []
    completed_keys = set()

    if raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        all_rows = existing.to_dict("records")
        completed_keys = set(
            existing.groupby(["model_name", "layer_frac", "target"]).groups.keys()
        )
        completed_models = set(existing["model_name"].unique())
        print(f"  [RESUME] Loaded {len(all_rows)} rows, {len(completed_keys)} combos done")
    else:
        completed_models = set()

    all_prompts = []
    for domain, items in PROMPTS.items():
        for item in items:
            all_prompts.append({**item, "domain": domain})

    conditions = [
        (0.0, 0.0, "control"),
        (SURGERY_STRENGTH, 0.0, "surgery_only"),
        (0.0, JITTER_STRENGTH, "jitter_only"),
        (SURGERY_STRENGTH, JITTER_STRENGTH, "both"),
    ]

    n_combos = len(LAYER_FRACTIONS) * len(HOOK_TARGETS)
    n_per_combo = len(conditions) * len(SEEDS)
    total_expected = len(MODELS) * n_combos * n_per_combo * len(all_prompts)

    print(f"\n{'=' * 70}")
    print(f"EXP-017 PHASE 1: SUBMODULE FACTORIAL EVALUATION")
    print(f"Models: {len(MODELS)}")
    print(f"Layer fractions: {LAYER_FRACTIONS}")
    print(f"Hook targets: {HOOK_TARGETS}")
    print(f"Prompts: {len(all_prompts)}, Seeds: {len(SEEDS)}, Conditions: {len(conditions)}")
    print(f"Expected trials: {total_expected}")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, group, params_b) in enumerate(MODELS):
        # Check if all combos for this model are done
        model_combos = {
            (model_name, lf, t)
            for lf in LAYER_FRACTIONS for t in HOOK_TARGETS
        }
        if model_combos <= completed_keys:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (all combos done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{model_idx+1}/{len(MODELS)}] {model_name} ({group}, {params_b}B)")
        print(f"{'=' * 60}")

        t_model = time.time()
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"  [FAIL] {model_id}: {e}")
            continue

        layers = get_model_layers(model)
        if not layers:
            print(f"  [FAIL] No layers for {model_name}")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        total_layers = len(layers)
        print(f"  Total layers: {total_layers}")

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM: {mem_gb:.1f}GB allocated")

        device = next(model.parameters()).device
        batch_size = BATCH_SIZES.get(group, 1)
        model_rows = []
        model_failed = False

        # Verify submodule availability
        test_layer = layers[get_layer_idx(total_layers, 0.50)]
        for target in HOOK_TARGETS:
            try:
                mod = get_target_module(test_layer, target)
                print(f"  Target '{target}': {type(mod).__name__}")
            except ValueError as e:
                print(f"  [WARN] {e}")

        try:
            combo_num = 0
            for layer_frac in LAYER_FRACTIONS:
                layer_idx = get_layer_idx(total_layers, layer_frac)
                actual_frac = layer_idx / max(total_layers - 1, 1)
                layer_module = layers[layer_idx]

                for target in HOOK_TARGETS:
                    combo_num += 1
                    key = (model_name, layer_frac, target)
                    if key in completed_keys:
                        print(f"  [{combo_num}/{n_combos}] SKIP L={layer_frac} "
                              f"target={target} (done)")
                        continue

                    target_module = get_target_module(layer_module, target)
                    print(f"\n  [{combo_num}/{n_combos}] L={layer_frac:.2f} -> "
                          f"idx {layer_idx}/{total_layers} target={target} "
                          f"({type(target_module).__name__})")

                    cond_count = 0
                    for surg_s, jit_s, cond_label in conditions:
                        for seed in SEEDS:
                            cond_count += 1
                            hooks = []

                            if surg_s > 0:
                                h = PRExpansionHook(strength=surg_s, seed=seed)
                                h.attach(target_module)
                                hooks.append(h)
                            if jit_s > 0:
                                h = JitterHook(strength=jit_s, seed=seed + 10000)
                                h.attach(target_module)
                                hooks.append(h)

                            results = generate_prompts(
                                model, tokenizer, all_prompts, device, batch_size
                            )

                            for p, (text, correct, confidence) in results:
                                model_rows.append({
                                    "model_name": model_name, "group": group,
                                    "params_b": params_b, "seed": seed,
                                    "layer_idx": layer_idx,
                                    "layer_frac": round(actual_frac, 3),
                                    "layer_target": layer_frac,
                                    "target": target,
                                    "surgery_strength": surg_s,
                                    "jitter_strength": jit_s,
                                    "condition": cond_label,
                                    "domain": p["domain"], "prompt_id": p["id"],
                                    "split": p["split"],
                                    "answer": p["answer"], "generated": text,
                                    "correct": int(correct),
                                    "confidence": confidence,
                                })

                            for h in hooks:
                                h.detach()

                            elapsed = time.time() - t_model
                            rate = len(model_rows) / elapsed if elapsed > 0 else 0
                            print(f"    [{cond_count}/{n_per_combo}] "
                                  f"L={layer_frac:.2f} {target} "
                                  f"{cond_label} seed={seed} "
                                  f"({elapsed:.0f}s, {rate:.1f} t/s)")

        except Exception as model_err:
            print(f"  [MODEL FAIL] {model_name}: {type(model_err).__name__}: "
                  f"{model_err}")
            model_failed = True

        if not model_failed and model_rows:
            all_rows.extend(model_rows)
            pd.DataFrame(all_rows).to_csv(raw_csv, index=False)
            print(f"  SAVED {len(model_rows)} trials for {model_name}")

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_model
        print(f"  {model_name}: {len(model_rows)} trials in {elapsed:.0f}s "
              f"({elapsed/60:.1f} min)")

    df = pd.DataFrame(all_rows)
    df.to_csv(raw_csv, index=False)
    print(f"\nPhase 1 complete: {len(df)} rows, {df['model_name'].nunique()} models")
    return df


# ── Phase 2: Branch diagnostics (clean pass) ─────────────────────────────
def collect_diagnostics(out_dir: Path) -> pd.DataFrame:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = tables_dir / "branch_diagnostics.csv"

    if diag_csv.exists():
        existing = pd.read_csv(diag_csv)
        completed_models = set(existing["model_name"].unique())
        if len(completed_models) >= len(MODELS):
            print(f"  [RESUME] Branch diagnostics complete")
            return existing
        print(f"  [RESUME] Diagnostics done for {completed_models}")
    else:
        completed_models = set()

    # Build diagnostic prompts (same 64 eval prompts)
    diag_prompts = []
    for domain, items in PROMPTS.items():
        for item in items:
            diag_prompts.append(item["prompt"])

    all_diag_rows = []
    if diag_csv.exists():
        all_diag_rows = pd.read_csv(diag_csv).to_dict("records")

    print(f"\n{'=' * 70}")
    print(f"EXP-017 PHASE 2: BRANCH DIAGNOSTICS (CLEAN PASS)")
    print(f"Models: {len(MODELS)}, Layers: {len(LAYER_FRACTIONS)}, "
          f"Prompts: {len(diag_prompts)}")
    print(f"Metrics: self_attn norm, mlp norm, cosine alignment")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, group, params_b) in enumerate(MODELS):
        if model_name in completed_models:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (done)")
            continue

        print(f"\n  [{model_idx+1}/{len(MODELS)}] {model_name} ({group})")
        t_model = time.time()

        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"  [FAIL] {model_id}: {e}")
            continue

        layers = get_model_layers(model)
        if not layers:
            print(f"  [FAIL] No layers for {model_name}")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        total_layers = len(layers)
        device = next(model.parameters()).device

        for layer_frac in LAYER_FRACTIONS:
            layer_idx = get_layer_idx(total_layers, layer_frac)
            actual_frac = layer_idx / max(total_layers - 1, 1)
            layer_module = layers[layer_idx]

            # Attach capture hooks on all three targets simultaneously
            block_cap = BranchCaptureHook()
            attn_cap = BranchCaptureHook()
            mlp_cap = BranchCaptureHook()

            block_cap.attach(layer_module)
            attn_cap.attach(get_target_module(layer_module, "self_attn"))
            mlp_cap.attach(get_target_module(layer_module, "mlp"))

            # Run all prompts (no perturbation)
            for prompt_text in diag_prompts:
                inp = tokenizer(
                    prompt_text, return_tensors="pt",
                    truncation=True, max_length=128
                ).to(device)
                with torch.no_grad():
                    model(**inp)

            block_cap.detach()
            attn_cap.detach()
            mlp_cap.detach()

            block_norms, block_vecs = block_cap.get_and_clear()
            attn_norms, attn_vecs = attn_cap.get_and_clear()
            mlp_norms, mlp_vecs = mlp_cap.get_and_clear()

            if len(block_norms) < 2:
                print(f"    Layer {layer_frac:.2f}: no captures")
                continue

            # Compute metrics
            mean_block_norm = float(block_norms.mean())
            mean_attn_norm = float(attn_norms.mean())
            mean_mlp_norm = float(mlp_norms.mean())
            attn_mlp_ratio = mean_attn_norm / (mean_mlp_norm + 1e-10)

            # Cosine similarity between attn and mlp outputs
            if attn_vecs.shape[0] > 0 and mlp_vecs.shape[0] > 0:
                attn_normed = attn_vecs / attn_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                mlp_normed = mlp_vecs / mlp_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cos_attn_mlp = float((attn_normed * mlp_normed).sum(dim=1).mean())
            else:
                cos_attn_mlp = float('nan')

            # Cosine of attn output with block output
            if attn_vecs.shape[0] > 0 and block_vecs.shape[0] > 0:
                block_normed = block_vecs / block_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cos_attn_block = float((attn_normed * block_normed).sum(dim=1).mean())
                cos_mlp_block = float((mlp_normed * block_normed).sum(dim=1).mean())
            else:
                cos_attn_block = float('nan')
                cos_mlp_block = float('nan')

            all_diag_rows.append({
                "model_name": model_name, "group": group, "params_b": params_b,
                "layer_idx": layer_idx,
                "layer_frac": round(actual_frac, 3),
                "layer_target": layer_frac,
                "block_norm": round(mean_block_norm, 4),
                "attn_norm": round(mean_attn_norm, 4),
                "mlp_norm": round(mean_mlp_norm, 4),
                "attn_mlp_ratio": round(attn_mlp_ratio, 4),
                "cos_attn_mlp": round(cos_attn_mlp, 6),
                "cos_attn_block": round(cos_attn_block, 6),
                "cos_mlp_block": round(cos_mlp_block, 6),
                "n_prompts": int(len(block_norms)),
            })
            print(f"    Layer {layer_frac:.2f}: attn_norm={mean_attn_norm:.1f}, "
                  f"mlp_norm={mean_mlp_norm:.1f}, ratio={attn_mlp_ratio:.3f}, "
                  f"cos(attn,mlp)={cos_attn_mlp:.4f}")

            del block_norms, block_vecs, attn_norms, attn_vecs, mlp_norms, mlp_vecs
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        completed_models.add(model_name)

        pd.DataFrame(all_diag_rows).to_csv(diag_csv, index=False)
        print(f"  {model_name}: diagnostics in {time.time() - t_model:.0f}s")

    diag_df = pd.DataFrame(all_diag_rows)
    diag_df.to_csv(diag_csv, index=False)
    print(f"\nPhase 2 complete: {len(diag_df)} measurements")
    return diag_df


# ── Analysis ─────────────────────────────────────────────────────────────
def run_analysis(out_dir: Path):
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables_dir / "raw_trials.csv")
    diag_df = pd.read_csv(tables_dir / "branch_diagnostics.csv")

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {len(df)} trials, {df['model_name'].nunique()} models, "
          f"{len(diag_df)} diagnostic measurements")
    print(f"Targets: {sorted(df['target'].unique())}")
    print(f"{'=' * 70}")

    # 1. Per-model per-layer per-target interaction
    interactions = _compute_target_interactions(df)
    int_df = pd.DataFrame(interactions)
    int_df.to_csv(tables_dir / "interaction_by_model_layer_target.csv", index=False)

    # 2. Target-level summary
    target_summary = _compute_target_summary(int_df, tables_dir)

    # 3. Wild-cluster bootstrap (exact, per target)
    wild = _wild_cluster_bootstrap(int_df, tables_dir)

    # 4. Model-clustered bootstrap CIs (per target)
    boot = _clustered_bootstrap(int_df, tables_dir)

    # 5. Within-model permutation (sensitivity, per target)
    perm = _permutation_test(df, tables_dir)

    # 6. Decision rules
    decision = _apply_decision_rules(int_df, wild, boot, tables_dir)

    # 7. Mixed-effects model with target interaction
    mixed = _fit_target_mixed_model(df, tables_dir)

    # 8. LOO sensitivity
    loo = _leave_one_out(int_df, tables_dir)

    # 9. Cal vs holdout
    split = _split_comparison(df, tables_dir)

    # 10. Figures
    _plot_target_interaction(int_df, boot, figures_dir)
    _plot_target_heatmap(int_df, figures_dir)
    _plot_model_target_profiles(int_df, figures_dir)
    _plot_branch_diagnostics(diag_df, figures_dir)
    _plot_diagnostics_vs_interaction(int_df, diag_df, figures_dir)

    # 11. Key findings + verdict
    _write_findings(df, int_df, diag_df, target_summary, wild, boot, perm,
                    decision, mixed, loo, split, out_dir)


def _compute_target_interactions(df: pd.DataFrame) -> List[Dict]:
    interactions = []
    for model_name in df["model_name"].unique():
        mdf = df[df["model_name"] == model_name]
        group = mdf["group"].iloc[0]
        params_b = mdf["params_b"].iloc[0]

        for layer_target in mdf["layer_target"].unique():
            ldf = mdf[mdf["layer_target"] == layer_target]
            for target in ldf["target"].unique():
                tdf = ldf[ldf["target"] == target]
                acc = {}
                for (ss, js), grp in tdf.groupby(
                    ["surgery_strength", "jitter_strength"]
                ):
                    key = ("surg" if ss > 0 else "ctrl",
                           "jit" if js > 0 else "off")
                    acc[key] = grp["correct"].mean()
                c00 = acc.get(("ctrl", "off"), np.nan)
                c10 = acc.get(("surg", "off"), np.nan)
                c01 = acc.get(("ctrl", "jit"), np.nan)
                c11 = acc.get(("surg", "jit"), np.nan)
                interaction = (c11 - c01) - (c10 - c00)
                interactions.append({
                    "model_name": model_name, "group": group,
                    "params_b": params_b,
                    "layer_target": float(layer_target),
                    "target": target,
                    "interaction": interaction,
                    "acc_control": c00, "acc_surgery": c10,
                    "acc_jitter": c01, "acc_both": c11,
                    "surgery_effect": c10 - c00,
                    "jitter_effect": c01 - c00,
                })
    return interactions


def _compute_target_summary(int_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    print("\n--- Target-level interaction summary ---")
    rows = []
    for target in HOOK_TARGETS:
        tdf = int_df[int_df["target"] == target]
        mean_int = tdf["interaction"].mean()
        std_int = tdf["interaction"].std()
        n = len(tdf)
        rows.append({
            "target": target,
            "mean_interaction": mean_int,
            "std_interaction": std_int,
            "n": n,
            "mean_surgery_effect": tdf["surgery_effect"].mean(),
            "mean_jitter_effect": tdf["jitter_effect"].mean(),
        })
        print(f"  {target:10s}: int={mean_int:+.4f} +/- {std_int:.4f} (n={n})")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(tables_dir / "target_summary.csv", index=False)
    return summary_df


def _wild_cluster_bootstrap(int_df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Exact wild-cluster bootstrap with Rademacher weights (2^K combos)."""
    print("\n--- Wild-cluster bootstrap (exact) ---")
    results = {}
    models = sorted(int_df["model_name"].unique())
    K = len(models)
    n_combos = 2 ** K

    for target in HOOK_TARGETS:
        tdf = int_df[int_df["target"] == target]
        model_ints = tdf.groupby("model_name")["interaction"].mean()

        # Align to model order
        ints = np.array([model_ints.get(m, np.nan) for m in models])
        observed_mean = np.nanmean(ints)

        count_extreme = 0
        for i in range(n_combos):
            weights = np.array(
                [1 if (i >> j) & 1 else -1 for j in range(K)]
            )
            boot_mean = np.nanmean(weights * ints)
            if abs(boot_mean) >= abs(observed_mean):
                count_extreme += 1

        p_wild = count_extreme / n_combos

        results[target] = {
            "observed_mean": float(observed_mean),
            "p_wild": float(p_wild),
            "n_models": K,
            "n_combos": n_combos,
            "model_interactions": {m: float(v) for m, v in model_ints.items()},
        }
        print(f"  {target:10s}: mean={observed_mean:+.4f}, p_wild={p_wild:.4f} "
              f"({n_combos} combos)")

    with open(tables_dir / "wild_cluster_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _clustered_bootstrap(int_df: pd.DataFrame, tables_dir: Path,
                         n_boot: int = 5000) -> Dict:
    """Model-clustered bootstrap CIs per target."""
    print("\n--- Clustered bootstrap CIs (5000) ---")
    rng = np.random.default_rng(42)
    results = {}

    for target in HOOK_TARGETS:
        tdf = int_df[int_df["target"] == target]
        model_ints = tdf.groupby("model_name")["interaction"].mean()
        models = model_ints.index.values
        values = model_ints.values
        K = len(models)

        boot_means = []
        for _ in range(n_boot):
            idx = rng.integers(0, K, size=K)
            boot_means.append(np.mean(values[idx]))

        boot_means = np.array(boot_means)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        results[target] = {
            "mean": float(values.mean()),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
            "in_rope": bool(ci_lo >= -SESOI_PROB and ci_hi <= SESOI_PROB),
        }
        print(f"  {target:10s}: mean={values.mean():+.4f}, "
              f"CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
              f"excl_zero={results[target]['ci_excludes_zero']}")

    # Also compute per layer-target
    results["by_layer"] = {}
    for lt in sorted(int_df["layer_target"].unique()):
        results["by_layer"][str(lt)] = {}
        for target in HOOK_TARGETS:
            subset = int_df[
                (int_df["layer_target"] == lt) & (int_df["target"] == target)
            ]
            model_ints = subset.groupby("model_name")["interaction"].mean()
            values = model_ints.values
            K = len(values)

            boot_means = []
            for _ in range(n_boot):
                idx = rng.integers(0, K, size=K)
                boot_means.append(np.mean(values[idx]))

            boot_means = np.array(boot_means)
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            results["by_layer"][str(lt)][target] = {
                "mean": float(values.mean()),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
            }

    with open(tables_dir / "clustered_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _permutation_test(df: pd.DataFrame, tables_dir: Path,
                      n_perm: int = 5000) -> Dict:
    """Within-model permutation test per target."""
    print("\n--- Permutation test (5000) ---")
    rng = np.random.default_rng(42)
    results = {}

    for target in HOOK_TARGETS:
        tdf = df[df["target"] == target]
        obs_li = _compute_target_interactions(tdf)
        observed = np.mean([x["interaction"] for x in obs_li])

        null_dist = []
        for i in range(n_perm):
            df_perm = tdf.copy()
            for mn in df_perm["model_name"].unique():
                mask = df_perm["model_name"] == mn
                idx = df_perm.index[mask]
                df_perm.loc[idx, "surgery_strength"] = rng.permutation(
                    df_perm.loc[idx, "surgery_strength"].values)
                df_perm.loc[idx, "jitter_strength"] = rng.permutation(
                    df_perm.loc[idx, "jitter_strength"].values)
            perm_li = _compute_target_interactions(df_perm)
            null_dist.append(np.mean([x["interaction"] for x in perm_li]))

        null_dist = np.array(null_dist)
        p_perm = np.mean(np.abs(null_dist) >= np.abs(observed))

        results[target] = {
            "observed": float(observed),
            "p_perm": float(p_perm),
            "null_mean": float(null_dist.mean()),
            "null_std": float(null_dist.std()),
        }
        print(f"  {target:10s}: obs={observed:+.4f}, p_perm={p_perm:.4f}")

    with open(tables_dir / "permutation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _apply_decision_rules(int_df: pd.DataFrame, wild: Dict, boot: Dict,
                          tables_dir: Path) -> Dict:
    """Apply Codex-specified decision rules."""
    print("\n--- Decision rules ---")

    block = boot.get("block", {})
    attn = boot.get("self_attn", {})
    mlp = boot.get("mlp", {})

    block_neg = block.get("ci_excludes_zero", False) and block.get("mean", 0) < 0
    attn_neg = attn.get("ci_excludes_zero", False) and attn.get("mean", 0) < 0
    mlp_neg = mlp.get("ci_excludes_zero", False) and mlp.get("mean", 0) < 0
    mlp_rope = mlp.get("in_rope", False)
    attn_rope = attn.get("in_rope", False)
    block_weak = not block_neg
    attn_weak = not attn_neg
    mlp_weak = not mlp_neg

    # Decision rules from Codex
    if attn_neg and mlp_rope:
        verdict = "ATTENTION_PATH_COUPLING"
        explanation = (
            f"Coupling localized in attention path: self_attn interaction="
            f"{attn.get('mean', 0):.4f} (CI excludes zero), "
            f"MLP interaction={mlp.get('mean', 0):.4f} (in ROPE)."
        )
    elif block_neg and attn_weak and mlp_weak:
        verdict = "RESIDUAL_STREAM_INTEGRATION"
        explanation = (
            f"Coupling arises from residual integration: block interaction="
            f"{block.get('mean', 0):.4f} (CI excludes zero), "
            f"but neither self_attn ({attn.get('mean', 0):.4f}) "
            f"nor MLP ({mlp.get('mean', 0):.4f}) individually significant."
        )
    elif attn_neg and mlp_neg:
        verdict = "GENERIC_BLOCK_COUPLING"
        explanation = (
            f"Both branches show coupling: self_attn={attn.get('mean', 0):.4f}, "
            f"MLP={mlp.get('mean', 0):.4f} (both CIs exclude zero). "
            f"Not cleanly separable."
        )
    elif mlp_neg and attn_rope:
        verdict = "MLP_PATH_COUPLING"
        explanation = (
            f"Coupling localized in MLP path: mlp interaction="
            f"{mlp.get('mean', 0):.4f} (CI excludes zero), "
            f"self_attn interaction={attn.get('mean', 0):.4f} (in ROPE)."
        )
    else:
        verdict = "STOP_MECHANISM_CHASING"
        explanation = (
            f"No clean separation. block={block.get('mean', 0):.4f}, "
            f"self_attn={attn.get('mean', 0):.4f}, "
            f"MLP={mlp.get('mean', 0):.4f}. "
            f"Block excl_zero={block_neg}, attn excl_zero={attn_neg}, "
            f"mlp excl_zero={mlp_neg}."
        )

    results = {
        "verdict": verdict,
        "explanation": explanation,
        "block_mean": float(block.get("mean", 0)),
        "block_ci": [float(block.get("ci_lo", 0)), float(block.get("ci_hi", 0))],
        "block_excludes_zero": block_neg,
        "attn_mean": float(attn.get("mean", 0)),
        "attn_ci": [float(attn.get("ci_lo", 0)), float(attn.get("ci_hi", 0))],
        "attn_excludes_zero": attn_neg,
        "attn_in_rope": attn.get("in_rope", False),
        "mlp_mean": float(mlp.get("mean", 0)),
        "mlp_ci": [float(mlp.get("ci_lo", 0)), float(mlp.get("ci_hi", 0))],
        "mlp_excludes_zero": mlp_neg,
        "mlp_in_rope": mlp.get("in_rope", False),
        "wild_p_block": float(wild.get("block", {}).get("p_wild", 1)),
        "wild_p_attn": float(wild.get("self_attn", {}).get("p_wild", 1)),
        "wild_p_mlp": float(wild.get("mlp", {}).get("p_wild", 1)),
    }

    with open(tables_dir / "decision_rules.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")
    return results


def _fit_target_mixed_model(df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Mixed-effects model with target as moderator."""
    print("\n--- Target mixed-effects model ---")
    df = df.copy()
    df["surgery_bin"] = (df["surgery_strength"] > 0).astype(int)
    df["jitter_bin"] = (df["jitter_strength"] > 0).astype(int)
    df["interaction"] = df["surgery_bin"] * df["jitter_bin"]
    df["is_attn"] = (df["target"] == "self_attn").astype(int)
    df["is_mlp"] = (df["target"] == "mlp").astype(int)
    df["attn_x_int"] = df["is_attn"] * df["interaction"]
    df["mlp_x_int"] = df["is_mlp"] * df["interaction"]

    results = {}
    try:
        import statsmodels.api as sm
        exog = df[["surgery_bin", "jitter_bin", "interaction",
                    "is_attn", "is_mlp", "attn_x_int", "mlp_x_int"]].copy()
        exog.insert(0, "intercept", 1)
        logit = sm.Logit(df["correct"].values, exog.values)
        fit = logit.fit(disp=0, cov_type="cluster",
                        cov_kwds={"groups": df["model_name"]})
        names = list(exog.columns)
        for i, n in enumerate(names):
            results[f"coef_{n}"] = float(fit.params[i])
            results[f"se_{n}"] = float(fit.bse[i])
            results[f"p_{n}"] = float(fit.pvalues[i])

        results["interaction_beta"] = float(fit.params[names.index("interaction")])
        results["interaction_p"] = float(fit.pvalues[names.index("interaction")])
        results["attn_x_int_beta"] = float(fit.params[names.index("attn_x_int")])
        results["attn_x_int_p"] = float(fit.pvalues[names.index("attn_x_int")])
        results["mlp_x_int_beta"] = float(fit.params[names.index("mlp_x_int")])
        results["mlp_x_int_p"] = float(fit.pvalues[names.index("mlp_x_int")])

        print(f"  Interaction (block ref): beta={results['interaction_beta']:.4f}, "
              f"p={results['interaction_p']:.4f}")
        print(f"  Attn x Int: beta={results['attn_x_int_beta']:.4f}, "
              f"p={results['attn_x_int_p']:.4f}")
        print(f"  MLP x Int: beta={results['mlp_x_int_beta']:.4f}, "
              f"p={results['mlp_x_int_p']:.4f}")
    except Exception as e:
        print(f"  Mixed model failed: {e}")
        results["method"] = "failed"

    with open(tables_dir / "mixed_effects_target.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _leave_one_out(int_df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Leave-one-out ---")
    results = {}
    for target in HOOK_TARGETS:
        tdf = int_df[int_df["target"] == target]
        model_means = tdf.groupby("model_name")["interaction"].mean()
        full_mean = model_means.mean()
        loo_rows = []
        for mn in model_means.index:
            loo_mean = model_means.drop(mn).mean()
            loo_rows.append({"excluded": mn, "loo_mean": float(loo_mean),
                             "delta": float(loo_mean - full_mean)})
        results[target] = {
            "full_mean": float(full_mean),
            "loo_std": float(pd.DataFrame(loo_rows)["loo_mean"].std()),
            "models": loo_rows,
        }
        print(f"  {target:10s}: mean={full_mean:.4f}, "
              f"LOO std={results[target]['loo_std']:.4f}")

    with open(tables_dir / "loo_by_target.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _split_comparison(df: pd.DataFrame, tables_dir: Path) -> Dict:
    print("\n--- Cal vs holdout ---")
    results = {}
    for target in HOOK_TARGETS:
        tdf = df[df["target"] == target]
        cal_li = _compute_target_interactions(tdf[tdf["split"] == "cal"])
        hold_li = _compute_target_interactions(tdf[tdf["split"] == "hold"])
        cal_mean = np.mean([x["interaction"] for x in cal_li]) if cal_li else 0
        hold_mean = np.mean([x["interaction"] for x in hold_li]) if hold_li else 0
        results[target] = {
            "cal_interaction": float(cal_mean),
            "hold_interaction": float(hold_mean),
            "gap_pp": float(abs(cal_mean - hold_mean) * 100),
        }
        print(f"  {target:10s}: cal={cal_mean:.4f}, hold={hold_mean:.4f} "
              f"(gap: {results[target]['gap_pp']:.1f}pp)")

    with open(tables_dir / "split_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Figures ──────────────────────────────────────────────────────────────
TARGET_COLORS = {"block": "#2196F3", "self_attn": "#FF5722", "mlp": "#4CAF50"}
MODEL_MARKERS = {"Qwen2.5-7B": "o", "Qwen3-8B": "s", "OLMo2-7B": "^",
                 "DSR1-7B": "D"}


def _plot_target_interaction(int_df: pd.DataFrame, boot: Dict,
                             fig_dir: Path):
    """Main figure: interaction by target with bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    targets = HOOK_TARGETS
    x_pos = np.arange(len(targets))
    means = []
    ci_los = []
    ci_his = []
    colors = []

    for t in targets:
        b = boot.get(t, {})
        means.append(b.get("mean", 0))
        ci_los.append(b.get("ci_lo", 0))
        ci_his.append(b.get("ci_hi", 0))
        colors.append(TARGET_COLORS[t])

    means = np.array(means)
    ci_los = np.array(ci_los)
    ci_his = np.array(ci_his)
    errors = np.array([means - ci_los, ci_his - means])

    bars = ax.bar(x_pos, means, yerr=errors, capsize=8, color=colors,
                  alpha=0.7, edgecolor="black", linewidth=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhspan(-SESOI_PROB, SESOI_PROB, alpha=0.15, color="green",
               label="ROPE")

    # Overlay individual model points
    for i, t in enumerate(targets):
        tdf = int_df[int_df["target"] == t]
        model_means = tdf.groupby("model_name")["interaction"].mean()
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15,
                                                     len(model_means))
        for j, (mn, val) in enumerate(model_means.items()):
            marker = MODEL_MARKERS.get(mn, "o")
            ax.scatter(i + jitter[j], val, marker=marker, s=40,
                       c="black", alpha=0.6, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=11)
    ax.set_ylabel("Interaction Effect", fontsize=12)
    ax.set_title("Exp-017: Surgery x Jitter Interaction by Hook Target\n"
                 "7B+ Transformers (4 models, clustered bootstrap CIs)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(fig_dir / "target_interaction.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  -> target_interaction.png")


def _plot_target_heatmap(int_df: pd.DataFrame, fig_dir: Path):
    """Heatmap: model x target interaction."""
    # Pool across layers for the primary view
    pivot = int_df.pivot_table(
        values="interaction", index="model_name",
        columns="target", aggfunc="mean"
    )
    pivot = pivot[HOOK_TARGETS]  # force order

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Hook Target")
    ax.set_ylabel("Model")
    ax.set_title("Exp-017: Interaction Heatmap (Model x Target)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(val) > 0.05 else "black")

    plt.colorbar(im, ax=ax, label="Interaction Effect")
    plt.tight_layout()
    plt.savefig(fig_dir / "target_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> target_heatmap.png")


def _plot_model_target_profiles(int_df: pd.DataFrame, fig_dir: Path):
    """Per-model interaction by target, split by layer."""
    layers = sorted(int_df["layer_target"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 6),
                              sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax_idx, lt in enumerate(layers):
        ax = axes[ax_idx]
        ldf = int_df[int_df["layer_target"] == lt]
        models = sorted(ldf["model_name"].unique())
        x_pos = np.arange(len(HOOK_TARGETS))
        width = 0.18

        for mi, mn in enumerate(models):
            mdf = ldf[ldf["model_name"] == mn]
            vals = [mdf[mdf["target"] == t]["interaction"].values[0]
                    if len(mdf[mdf["target"] == t]) > 0 else 0
                    for t in HOOK_TARGETS]
            offset = (mi - len(models) / 2 + 0.5) * width
            ax.bar(x_pos + offset, vals, width, label=mn, alpha=0.8)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.axhspan(-SESOI_PROB, SESOI_PROB, alpha=0.15, color="green")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([t.replace("_", "\n") for t in HOOK_TARGETS])
        ax.set_ylabel("Interaction" if ax_idx == 0 else "")
        ax.set_title(f"Layer {lt:.2f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Exp-017: Per-Model Interaction by Target and Layer",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_target_profiles.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  -> model_target_profiles.png")


def _plot_branch_diagnostics(diag_df: pd.DataFrame, fig_dir: Path):
    """Branch norm comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Norm comparison
    ax = axes[0]
    models = sorted(diag_df["model_name"].unique())
    x_pos = np.arange(len(models))
    width = 0.25
    for bi, (branch, color) in enumerate(
        [("attn_norm", TARGET_COLORS["self_attn"]),
         ("mlp_norm", TARGET_COLORS["mlp"])]
    ):
        vals = [diag_df[diag_df["model_name"] == m][branch].mean()
                for m in models]
        offset = (bi - 0.5) * width
        ax.bar(x_pos + offset, vals, width, label=branch.replace("_norm", ""),
               color=color, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Branch Output Norms")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Cosine alignment
    ax = axes[1]
    cos_metrics = ["cos_attn_mlp", "cos_attn_block", "cos_mlp_block"]
    cos_labels = ["attn-mlp", "attn-block", "mlp-block"]
    for ci, (metric, label) in enumerate(zip(cos_metrics, cos_labels)):
        vals = [diag_df[diag_df["model_name"] == m][metric].mean()
                for m in models]
        offset = (ci - 1) * width
        ax.bar(x_pos + offset, vals, width, label=label, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Branch Alignment")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Exp-017: Branch Diagnostics (Clean Pass)", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "branch_diagnostics.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  -> branch_diagnostics.png")


def _plot_diagnostics_vs_interaction(int_df: pd.DataFrame,
                                      diag_df: pd.DataFrame,
                                      fig_dir: Path):
    """Scatter: branch diagnostics vs interaction magnitude."""
    # Merge diagnostics with interaction data (for self_attn and mlp targets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (target, diag_col) in enumerate(
        [("self_attn", "attn_norm"), ("mlp", "mlp_norm")]
    ):
        ax = axes[ax_idx]
        t_int = int_df[int_df["target"] == target][
            ["model_name", "layer_target", "interaction"]
        ]
        merged = t_int.merge(
            diag_df[["model_name", "layer_target", diag_col]],
            on=["model_name", "layer_target"], how="inner"
        )
        if len(merged) < 3:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center")
            continue

        for mn in merged["model_name"].unique():
            msub = merged[merged["model_name"] == mn]
            marker = MODEL_MARKERS.get(mn, "o")
            ax.scatter(msub[diag_col], msub["interaction"],
                       marker=marker, s=60, alpha=0.7, label=mn)

        if len(merged) >= 3:
            r, p = sp_stats.pearsonr(merged[diag_col], merged["interaction"])
            z = np.polyfit(merged[diag_col], merged["interaction"], 1)
            x_line = np.linspace(merged[diag_col].min(),
                                 merged[diag_col].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.4,
                    label=f"r={r:.3f}, p={p:.3f}")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel(f"{diag_col.replace('_', ' ').title()}")
        ax.set_ylabel("Interaction Effect")
        ax.set_title(f"{target} Norm vs Interaction")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp-017: Branch Diagnostics vs Interaction", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "diagnostics_vs_interaction.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  -> diagnostics_vs_interaction.png")


# ── Key findings + verdict ───────────────────────────────────────────────
def _write_findings(df, int_df, diag_df, target_summary, wild, boot, perm,
                    decision, mixed, loo, split, out_dir):
    lines = []
    lines.append(f"models_tested: {df['model_name'].nunique()}")
    lines.append(f"total_trials: {len(df)}")
    lines.append(f"diagnostic_measurements: {len(diag_df)}")
    lines.append(f"targets: {', '.join(HOOK_TARGETS)}")
    lines.append(f"domains: {', '.join(sorted(df['domain'].unique()))}")
    lines.append(f"layer_fractions: {LAYER_FRACTIONS}")
    lines.append(f"surgery_strength: {SURGERY_STRENGTH}")
    lines.append(f"jitter_strength: {JITTER_STRENGTH}")
    lines.append(f"seeds: {len(SEEDS)}")
    lines.append(f"prompts: {len(df['prompt_id'].unique())}")

    # Global interaction by target
    for target in HOOK_TARGETS:
        tdf = int_df[int_df["target"] == target]
        mean_int = tdf["interaction"].mean()
        lines.append(f"target_{target}_mean_interaction: {mean_int:.5f}")

    # Per-model mean interactions
    for mn in sorted(int_df["model_name"].unique()):
        for target in HOOK_TARGETS:
            subset = int_df[(int_df["model_name"] == mn) &
                            (int_df["target"] == target)]
            if len(subset) > 0:
                mint = subset["interaction"].mean()
                lines.append(f"model_{mn}_{target}_interaction: {mint:.5f}")

    # Wild-cluster bootstrap
    for target in HOOK_TARGETS:
        w = wild.get(target, {})
        lines.append(f"wild_p_{target}: {w.get('p_wild', 1):.4f}")

    # Clustered bootstrap CIs
    for target in HOOK_TARGETS:
        b = boot.get(target, {})
        lines.append(f"bootstrap_{target}_ci: "
                     f"[{b.get('ci_lo', 0):.4f}, {b.get('ci_hi', 0):.4f}]")
        lines.append(f"bootstrap_{target}_excludes_zero: "
                     f"{b.get('ci_excludes_zero', False)}")

    # Permutation
    for target in HOOK_TARGETS:
        p = perm.get(target, {})
        lines.append(f"permutation_p_{target}: {p.get('p_perm', 1):.4f}")

    # Mixed-effects
    if "interaction_beta" in mixed:
        lines.append(f"mixed_interaction_beta: "
                     f"{mixed['interaction_beta']:.4f}")
        lines.append(f"mixed_interaction_p: {mixed['interaction_p']:.4f}")
        lines.append(f"mixed_attn_x_int_beta: "
                     f"{mixed.get('attn_x_int_beta', 0):.4f}")
        lines.append(f"mixed_attn_x_int_p: "
                     f"{mixed.get('attn_x_int_p', 1):.4f}")
        lines.append(f"mixed_mlp_x_int_beta: "
                     f"{mixed.get('mlp_x_int_beta', 0):.4f}")
        lines.append(f"mixed_mlp_x_int_p: "
                     f"{mixed.get('mlp_x_int_p', 1):.4f}")

    # Cal-holdout
    max_gap = max(s.get("gap_pp", 0) for s in split.values())
    lines.append(f"max_cal_holdout_gap_pp: {max_gap:.1f}")

    # Branch diagnostics summary
    for mn in sorted(diag_df["model_name"].unique()):
        mdf = diag_df[diag_df["model_name"] == mn]
        lines.append(
            f"diag_{mn}_attn_norm: {mdf['attn_norm'].mean():.1f}"
        )
        lines.append(
            f"diag_{mn}_mlp_norm: {mdf['mlp_norm'].mean():.1f}"
        )
        lines.append(
            f"diag_{mn}_attn_mlp_ratio: {mdf['attn_mlp_ratio'].mean():.3f}"
        )

    # Verdict
    lines.append(f"verdict: {decision['verdict']}")
    lines.append(f"verdict_explanation: {decision['explanation']}")

    # Business + scientific implications
    v = decision["verdict"]
    if v == "ATTENTION_PATH_COUPLING":
        biz = (f"{v} -- The 7B+ surgery-jitter coupling is localized in the "
               f"attention path. Deployment testing can focus perturbation "
               f"monitoring on attention outputs rather than full blocks.")
        sci = (f"Submodule decomposition reveals the coupling is carried by "
               f"the attention mechanism, not the MLP or residual integration. "
               f"This suggests attention-specific geometry drives the effect.")
    elif v == "RESIDUAL_STREAM_INTEGRATION":
        biz = (f"{v} -- The coupling arises from how attention and MLP outputs "
               f"integrate in the residual stream. Neither branch alone carries "
               f"the effect. Full-block monitoring needed for deployment.")
        sci = (f"The coupling is an emergent property of residual integration, "
               f"not localized in either computational branch. This points to "
               f"superposition or interference effects in the residual stream.")
    elif v == "GENERIC_BLOCK_COUPLING":
        biz = (f"{v} -- Both attention and MLP paths carry the coupling. "
               f"No shortcut for deployment testing; full block perturbation "
               f"testing required at all submodules.")
        sci = (f"The coupling is distributed across both computational branches "
               f"within the transformer block. This suggests a pervasive "
               f"architectural property rather than a pathway-specific effect.")
    elif v == "MLP_PATH_COUPLING":
        biz = (f"{v} -- The 7B+ coupling is localized in the MLP path. "
               f"Deployment perturbation testing can focus on MLP outputs.")
        sci = (f"Submodule decomposition reveals the MLP branch carries the "
               f"coupling. This suggests MLP hidden-state geometry is the "
               f"relevant bottleneck.")
    else:
        biz = (f"{v} -- No clean separation of coupling across submodules. "
               f"Mechanism remains unresolved after 6 experiments "
               f"(exp-012 through exp-017). Recommend freezing mechanism "
               f"investigation and pivoting to deployment consequence testing.")
        sci = (f"Submodule decomposition at 3 targets x 2 layers does not "
               f"cleanly localize the coupling. Combined with exp-016 (no "
               f"layer localization, no geometry prediction, no training "
               f"regularization), the effect appears to be a deep, distributed "
               f"architectural property that resists simple mechanistic "
               f"explanation.")

    lines.append(f"business_implication: {biz}")
    lines.append(f"scientific_implication: {sci}")

    (out_dir / "key_findings.txt").write_text("\n".join(lines),
                                               encoding="utf-8")
    print(f"\nFindings written. Verdict: {v}")
    print(f"  {decision['explanation']}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Exp-017: Submodule Locus of 7B+ Coupling")
    parser.add_argument("--stage",
                        choices=["collect", "diagnostics", "analyze", "all"],
                        default="all")
    parser.add_argument("--outdir",
                        default="analysis/orthogonality_submodule_017")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("collect", "all"):
        collect_factorial(out_dir)

    if args.stage in ("diagnostics", "all"):
        collect_diagnostics(out_dir)

    if args.stage in ("analyze", "all"):
        run_analysis(out_dir)


if __name__ == "__main__":
    main()
