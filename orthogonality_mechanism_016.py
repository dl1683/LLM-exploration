#!/usr/bin/env python3
"""
orthogonality_mechanism_016.py — Exp-016: Layerwise Coupling Mechanism in
7B+ Base vs Reasoning-Tuned Transformers

Codex-directed follow-up to Exp-015. Investigates WHY base transformers
show surgery×jitter coupling while reasoning-tuned transformers do not.

Design:
  5 models × 5 layers × 2×2 factorial × 5 seeds × 64 prompts = 32,000 eval trials
  + clean activation geometry pass (128 prompts across all layers)

  Groups:
    base_transformer: Qwen2.5-7B, Qwen3-8B, OLMo2-7B
    reasoning_tuned: DSR1-7B, DSR1-Llama-8B

  Layer fractions: [0.20, 0.35, 0.50, 0.65, 0.80]
  Fixed strengths: surgery=0.08, jitter=0.08
  Seeds: [11, 23, 37, 43, 59]
  Prompts: Same 64 from exp-014/015

  Phase 1: Factorial evaluation (32,000 trials)
  Phase 2: Clean geometry pass (PR, anisotropy per layer per model)

Hypotheses (pre-registered):
  H1: Coupling is layer-localized in mid/late layers for base transformers
  H2: Reasoning training compresses/regularizes the vulnerable band
  H3: Local geometry (PR, anisotropy) at each layer predicts interaction strength

Decision rules:
  - If base transformers show interaction concentrated in layers 0.35-0.65 and
    reasoning models show near-zero at those same layers → TRAINING_REGULARIZATION
  - If interaction correlates with local PR or anisotropy (r > 0.5, p < 0.05) →
    GEOMETRIC_BOTTLENECK
  - If reasoning models show interaction relocated to different layers →
    RELOCATED_VULNERABLE_CIRCUIT
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

LAYER_FRACTIONS = [0.20, 0.35, 0.50, 0.65, 0.80]
SURGERY_STRENGTH = 0.08
JITTER_STRENGTH = 0.08
SEEDS = [11, 23, 37, 43, 59]
MAX_NEW_TOKENS = 10

BATCH_SIZES = {"base_transformer": 4, "reasoning_tuned": 1}

# ── Model registry (all cached from exp-014/015) ─────────────────────────
MODELS: List[Tuple[str, str, str, float]] = [
    # Base transformers (3)
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B", "base_transformer", 7.0),
    ("Qwen/Qwen3-8B", "Qwen3-8B", "base_transformer", 8.0),
    ("allenai/OLMo-2-1124-7B", "OLMo2-7B", "base_transformer", 7.0),
    # Reasoning-tuned transformers (2)
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DSR1-7B", "reasoning_tuned", 7.0),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DSR1-Llama-8B", "reasoning_tuned", 8.0),
]

# ── Prompts: 64 total (32 cal + 32 holdout) — identical to exp-014/015 ───
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

# Extended prompts for geometry pass (128 total = 64 original + 64 new)
GEOMETRY_EXTRA_PROMPTS = [
    "The chemical symbol for iron is",
    "The speed of sound is approximately 343 meters per",
    "The boiling point of water at sea level is 100 degrees",
    "The formula for the area of a circle is pi times r",
    "Light travels faster than",
    "The periodic table has how many known elements? Answer with just the number:",
    "What is 12 * 12? Answer with just the number:",
    "The freezing point of water in Fahrenheit is",
    "Mount Everest is the tallest mountain on",
    "The human body has 206",
    "What is 15 * 15? Answer with just the number:",
    "The capital of France is",
    "What is 1000 / 8? Answer with just the number:",
    "The chemical formula for water is",
    "Photosynthesis converts sunlight into",
    "What is 8 * 7? Answer with just the number:",
    "The Earth orbits the",
    "Newton's first law describes objects at rest or in uniform",
    "What is 50 + 75? Answer with just the number:",
    "The smallest prime number is",
    "What is the square root of 64?",
    "DNA replication occurs during the S phase of the cell",
    "The speed of light in a vacuum is approximately 3 times 10 to the power of",
    "What is 19 + 23? Answer with just the number:",
    "The largest mammal on Earth is the blue",
    "What is 200 - 87? Answer with just the number:",
    "The atomic number of hydrogen is",
    "If x + 5 = 12, then x equals",
    "What is 6 * 9? Answer with just the number:",
    "Mitochondria are often called the powerhouse of the",
    "The chemical symbol for potassium is",
    "What is 360 / 4? Answer with just the number:",
    "The number of chromosomes in a human cell is",
    "What is 14 * 14? Answer with just the number:",
    "The third planet from the Sun is",
    "What is 81 / 9? Answer with just the number:",
    "The capital of Germany is",
    "What is 25 * 4? Answer with just the number:",
    "The speed of sound in air is approximately 343 meters per",
    "What is 17 + 28? Answer with just the number:",
    "The chemical symbol for silver is",
    "What is 10^3? Answer with just the number:",
    "The freezing point of water in Kelvin is approximately",
    "What is 33 + 44? Answer with just the number:",
    "The largest continent by area is",
    "What is 7 * 7? Answer with just the number:",
    "The chemical formula for carbon dioxide is",
    "What is 500 - 123? Answer with just the number:",
    "Gravity on the Moon is about one sixth of",
    "What is 18 * 3? Answer with just the number:",
    "The capital of Italy is",
    "What is 1024 / 2? Answer with just the number:",
    "An octagon has how many sides? Answer with just the number:",
    "What is 6^2? Answer with just the number:",
    "The chemical symbol for copper is",
    "What is 99 + 1? Answer with just the number:",
    "The longest river in the world is the",
    "What is 13 * 7? Answer with just the number:",
    "Absolute zero is 0 on the Kelvin scale, which is approximately -273 degrees",
    "What is 45 + 67? Answer with just the number:",
    "The atomic number of oxygen is",
    "What is 20 * 20? Answer with just the number:",
    "The capital of Spain is",
    "What is 1000 - 999? Answer with just the number:",
]


# ── Answer parsing (identical to exp-014/015) ─────────────────────────────
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


# ── Hooks (identical to exp-012-015) ──────────────────────────────────────
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


class ActivationCaptureHook:
    """Hook to capture hidden-state activations for geometry analysis."""
    def __init__(self):
        self.activations = []
        self.handle = None

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Store mean-pooled activation across sequence (one vector per sample)
        self.activations.append(hidden[:, -1, :].detach().float().cpu())

    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    def get_and_clear(self) -> torch.Tensor:
        if self.activations:
            result = torch.cat(self.activations, dim=0)
            self.activations = []
            return result
        return torch.tensor([])


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


# ── Geometry computation ─────────────────────────────────────────────────
def compute_participation_ratio(activations: torch.Tensor) -> float:
    """Compute PR from activation matrix [n_samples, d_model]."""
    if activations.shape[0] < 2:
        return float('nan')
    X = activations.float()
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=0)
    total = eigenvalues.sum()
    if total < 1e-10:
        return float('nan')
    normalized = eigenvalues / total
    pr = 1.0 / (normalized ** 2).sum().item()
    return pr


def compute_anisotropy(activations: torch.Tensor) -> float:
    """Compute anisotropy as mean cosine similarity between activation vectors."""
    if activations.shape[0] < 2:
        return float('nan')
    X = activations.float()
    norms = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_normed = X / norms
    # Sample pairs for efficiency
    n = min(activations.shape[0], 200)
    idx = torch.randperm(activations.shape[0])[:n]
    X_sub = X_normed[idx]
    cos_sim = X_sub @ X_sub.T
    # Mean of upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    anisotropy = cos_sim[mask].mean().item()
    return anisotropy


# ── Phase 1: Factorial evaluation ────────────────────────────────────────
def collect_factorial(out_dir: Path) -> pd.DataFrame:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = tables_dir / "raw_trials.csv"

    all_rows = []
    completed_models = set()

    if raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        all_rows = existing.to_dict("records")
        completed_models = set(existing["model_name"].unique())
        print(f"  [RESUME] Loaded {len(all_rows)} rows for {completed_models}")

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

    n_conditions = len(conditions) * len(SEEDS)
    n_layers = len(LAYER_FRACTIONS)

    total_expected = len(MODELS) * n_layers * n_conditions * len(all_prompts)
    print(f"\n{'=' * 70}")
    print(f"EXP-016 PHASE 1: LAYERWISE FACTORIAL EVALUATION")
    print(f"Models: {len(MODELS)} ({sum(1 for _,_,g,_ in MODELS if g=='base_transformer')} base, "
          f"{sum(1 for _,_,g,_ in MODELS if g=='reasoning_tuned')} reasoning)")
    print(f"Layer fractions: {LAYER_FRACTIONS}")
    print(f"Prompts: {len(all_prompts)}, Seeds: {len(SEEDS)}, Conditions: {len(conditions)}")
    print(f"Expected trials: {total_expected}")
    print(f"Already completed: {completed_models}")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, group, params_b) in enumerate(MODELS):
        if model_name in completed_models:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (completed)")
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
        print(f"  Loaded in {time.time() - t_model:.1f}s")

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM: {mem_gb:.1f}GB allocated")

        device = next(model.parameters()).device
        batch_size = BATCH_SIZES.get(group, 1)
        model_rows = []
        model_failed = False

        try:
            for layer_frac in LAYER_FRACTIONS:
                layer_idx = get_layer_idx(total_layers, layer_frac)
                actual_frac = layer_idx / max(total_layers - 1, 1)
                print(f"\n  Layer {layer_frac:.2f} -> idx {layer_idx}/{total_layers} "
                      f"(actual={actual_frac:.3f})")

                cond_count = 0
                for surg_s, jit_s, cond_label in conditions:
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

                        results = generate_prompts(model, tokenizer, all_prompts, device, batch_size)

                        for p, (text, correct, confidence) in results:
                            model_rows.append({
                                "model_name": model_name, "group": group,
                                "params_b": params_b, "seed": seed,
                                "layer_idx": layer_idx, "layer_frac": round(actual_frac, 3),
                                "layer_target": layer_frac,
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
                        print(f"    [{cond_count}/{n_conditions}] L={layer_frac:.2f} "
                              f"{cond_label} seed={seed} ({elapsed:.0f}s, {rate:.1f} t/s)")

        except Exception as model_err:
            print(f"  [MODEL FAIL] {model_name}: {type(model_err).__name__}: {model_err}")
            model_failed = True

        if not model_failed and model_rows:
            all_rows.extend(model_rows)
            completed_models.add(model_name)
            pd.DataFrame(all_rows).to_csv(raw_csv, index=False)
            print(f"  SAVED {len(model_rows)} trials for {model_name}")

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_model
        print(f"  {model_name}: {len(model_rows)} trials in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    df = pd.DataFrame(all_rows)
    df.to_csv(raw_csv, index=False)
    print(f"\nPhase 1 complete: {len(df)} rows, {df['model_name'].nunique()} models")
    return df


# ── Phase 2: Clean geometry pass ─────────────────────────────────────────
def collect_geometry(out_dir: Path) -> pd.DataFrame:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    geometry_csv = tables_dir / "layer_geometry.csv"

    if geometry_csv.exists():
        existing = pd.read_csv(geometry_csv)
        completed_models = set(existing["model_name"].unique())
        if len(completed_models) >= len(MODELS):
            print(f"  [RESUME] Geometry complete for all models")
            return existing
        print(f"  [RESUME] Geometry done for {completed_models}")
    else:
        completed_models = set()

    # Build geometry prompts: all 64 eval prompts + 64 extra = 128
    geo_prompts = []
    for domain, items in PROMPTS.items():
        for item in items:
            geo_prompts.append(item["prompt"])
    geo_prompts.extend(GEOMETRY_EXTRA_PROMPTS)

    all_geo_rows = []
    if geometry_csv.exists():
        all_geo_rows = pd.read_csv(geometry_csv).to_dict("records")

    print(f"\n{'=' * 70}")
    print(f"EXP-016 PHASE 2: CLEAN GEOMETRY PASS")
    print(f"Models: {len(MODELS)}, Layers: {len(LAYER_FRACTIONS)}, Prompts: {len(geo_prompts)}")
    print(f"Metrics: PR, PR/d_model, anisotropy")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, group, params_b) in enumerate(MODELS):
        if model_name in completed_models:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (geometry done)")
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
        d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") else None
        device = next(model.parameters()).device

        for layer_frac in LAYER_FRACTIONS:
            layer_idx = get_layer_idx(total_layers, layer_frac)
            actual_frac = layer_idx / max(total_layers - 1, 1)

            capture = ActivationCaptureHook()
            capture.attach(layers[layer_idx])

            # Run all geometry prompts (no perturbation)
            for prompt_text in geo_prompts:
                inp = tokenizer(
                    prompt_text, return_tensors="pt",
                    truncation=True, max_length=128
                ).to(device)
                with torch.no_grad():
                    model(**inp)

            capture.detach()
            acts = capture.get_and_clear()

            if acts.shape[0] < 2:
                print(f"    Layer {layer_frac:.2f}: no activations captured")
                continue

            pr = compute_participation_ratio(acts)
            aniso = compute_anisotropy(acts)
            d = d_model if d_model else acts.shape[1]
            pr_norm = pr / d if d else float('nan')

            all_geo_rows.append({
                "model_name": model_name, "group": group, "params_b": params_b,
                "layer_idx": layer_idx, "layer_frac": round(actual_frac, 3),
                "layer_target": layer_frac,
                "d_model": d, "pr": round(pr, 4), "pr_normalized": round(pr_norm, 6),
                "anisotropy": round(aniso, 6), "n_prompts": acts.shape[0],
            })
            print(f"    Layer {layer_frac:.2f}: PR={pr:.2f}, PR/d={pr_norm:.4f}, "
                  f"aniso={aniso:.4f}")

            del acts
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        completed_models.add(model_name)

        # Save after each model
        pd.DataFrame(all_geo_rows).to_csv(geometry_csv, index=False)
        print(f"  {model_name}: geometry in {time.time() - t_model:.0f}s")

    geo_df = pd.DataFrame(all_geo_rows)
    geo_df.to_csv(geometry_csv, index=False)
    print(f"\nPhase 2 complete: {len(geo_df)} layer-model measurements")
    return geo_df


# ── Analysis ─────────────────────────────────────────────────────────────
def run_analysis(out_dir: Path):
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tables_dir / "raw_trials.csv")
    geo_df = pd.read_csv(tables_dir / "layer_geometry.csv")

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {len(df)} trials, {df['model_name'].nunique()} models, "
          f"{len(geo_df)} geometry measurements")
    print(f"Groups: {sorted(df['group'].unique())}")
    print(f"{'=' * 70}")

    # ── 1. Per-model per-layer interaction ──
    layer_interactions = _compute_layer_interactions(df)
    li_df = pd.DataFrame(layer_interactions)
    li_df.to_csv(tables_dir / "interaction_by_model_layer.csv", index=False)

    # ── 2. Group-level layerwise interaction profiles ──
    group_layer = _compute_group_layer_profiles(li_df, tables_dir)

    # ── 3. Test H1: Layer localization ──
    h1_results = _test_h1_layer_localization(li_df, tables_dir)

    # ── 4. Test H2: Reasoning regularization ──
    h2_results = _test_h2_reasoning_regularization(li_df, tables_dir)

    # ── 5. Test H3: Geometry predicts interaction ──
    h3_results = _test_h3_geometry_prediction(li_df, geo_df, tables_dir)

    # ── 6. Mixed-effects with layer interaction ──
    mixed = _fit_layerwise_mixed_model(df, tables_dir)

    # ── 7. Bootstrap per layer position ──
    boot = _bootstrap_by_layer(li_df, tables_dir)

    # ── 8. Permutation test ──
    perm = _permutation_test(df, tables_dir)

    # ── 9. LOO per group ──
    loo = _leave_one_out_by_group(li_df, tables_dir)

    # ── 10. Cal vs holdout ──
    split = _split_comparison(df, tables_dir)

    # ── 11. Main effects by layer ──
    main_effects = _main_effects_by_layer(df, tables_dir)

    # ── 12. Figures ──
    _plot_layerwise_interaction(li_df, figures_dir)
    _plot_group_comparison(li_df, figures_dir)
    _plot_geometry_profiles(geo_df, figures_dir)
    _plot_geometry_vs_interaction(li_df, geo_df, h3_results, figures_dir)
    _plot_heatmap(li_df, figures_dir)

    # ── 13. Key findings + verdict ──
    _write_findings(df, li_df, geo_df, group_layer, h1_results, h2_results,
                    h3_results, mixed, boot, perm, loo, split, main_effects, out_dir)


def _compute_layer_interactions(df: pd.DataFrame) -> List[Dict]:
    interactions = []
    for model_name in df["model_name"].unique():
        mdf = df[df["model_name"] == model_name]
        group = mdf["group"].iloc[0]
        params_b = mdf["params_b"].iloc[0]

        for layer_target in mdf["layer_target"].unique():
            ldf = mdf[mdf["layer_target"] == layer_target]
            acc = {}
            for (ss, js), grp in ldf.groupby(["surgery_strength", "jitter_strength"]):
                key = ("surg" if ss > 0 else "ctrl", "jit" if js > 0 else "off")
                acc[key] = grp["correct"].mean()
            c00 = acc.get(("ctrl", "off"), np.nan)
            c10 = acc.get(("surg", "off"), np.nan)
            c01 = acc.get(("ctrl", "jit"), np.nan)
            c11 = acc.get(("surg", "jit"), np.nan)
            interaction = (c11 - c01) - (c10 - c00)
            interactions.append({
                "model_name": model_name, "group": group, "params_b": params_b,
                "layer_target": float(layer_target),
                "interaction": interaction,
                "acc_control": c00, "acc_surgery": c10,
                "acc_jitter": c01, "acc_both": c11,
                "surgery_effect": c10 - c00 if not np.isnan(c10 - c00) else np.nan,
                "jitter_effect": c01 - c00 if not np.isnan(c01 - c00) else np.nan,
            })
    return interactions


def _compute_group_layer_profiles(li_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    profiles = []
    for group in li_df["group"].unique():
        gdf = li_df[li_df["group"] == group]
        for lt in sorted(gdf["layer_target"].unique()):
            ltdf = gdf[gdf["layer_target"] == lt]
            profiles.append({
                "group": group, "layer_target": lt,
                "mean_interaction": ltdf["interaction"].mean(),
                "std_interaction": ltdf["interaction"].std(),
                "n_models": len(ltdf),
                "mean_surgery_effect": ltdf["surgery_effect"].mean(),
                "mean_jitter_effect": ltdf["jitter_effect"].mean(),
            })
    prof_df = pd.DataFrame(profiles)
    prof_df.to_csv(tables_dir / "group_layer_profiles.csv", index=False)
    print(f"\nGroup-layer profiles:")
    for _, row in prof_df.iterrows():
        print(f"  {row['group']:25s} L={row['layer_target']:.2f}: "
              f"int={row['mean_interaction']:+.4f} +/- {row['std_interaction']:.4f}")
    return prof_df


def _test_h1_layer_localization(li_df: pd.DataFrame, tables_dir: Path) -> Dict:
    """H1: Coupling is layer-localized in mid/late layers for base transformers."""
    print("\n--- H1: Layer localization test ---")
    base = li_df[li_df["group"] == "base_transformer"]
    mid_late = base[base["layer_target"].isin([0.35, 0.50, 0.65])]
    early_late = base[base["layer_target"].isin([0.20, 0.80])]

    mid_late_int = mid_late["interaction"].mean()
    early_late_int = early_late["interaction"].mean()

    # Are mid/late layers more negative than early/extreme?
    t_stat, t_p = sp_stats.ttest_ind(
        mid_late["interaction"].values, early_late["interaction"].values,
        alternative="less"
    )

    # Peak layer for base transformers
    base_by_layer = base.groupby("layer_target")["interaction"].mean()
    peak_layer = base_by_layer.idxmin()
    peak_value = base_by_layer.min()

    results = {
        "mid_late_mean": float(mid_late_int),
        "early_extreme_mean": float(early_late_int),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "peak_layer": float(peak_layer),
        "peak_interaction": float(peak_value),
        "h1_supported": bool(t_p < 0.05 and mid_late_int < -SESOI_PROB),
        "localization_ratio": float(abs(mid_late_int) / (abs(early_late_int) + 1e-10)),
    }

    with open(tables_dir / "h1_layer_localization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Mid/late mean: {mid_late_int:.4f}, Early/extreme: {early_late_int:.4f}")
    print(f"  t={t_stat:.3f}, p={t_p:.4f}, peak at L={peak_layer:.2f} ({peak_value:.4f})")
    print(f"  H1 supported: {results['h1_supported']}")
    return results


def _test_h2_reasoning_regularization(li_df: pd.DataFrame, tables_dir: Path) -> Dict:
    """H2: Reasoning training compresses/regularizes the vulnerable band."""
    print("\n--- H2: Reasoning regularization test ---")
    base = li_df[li_df["group"] == "base_transformer"]
    reasoning = li_df[li_df["group"] == "reasoning_tuned"]

    # Compare interaction profiles across all layers
    base_profile = base.groupby("layer_target")["interaction"].mean()
    reasoning_profile = reasoning.groupby("layer_target")["interaction"].mean()

    # Wilcoxon signed-rank test on paired layer-level interactions
    common_layers = sorted(set(base_profile.index) & set(reasoning_profile.index))
    base_vals = [base_profile[l] for l in common_layers]
    reasoning_vals = [reasoning_profile[l] for l in common_layers]

    if len(common_layers) >= 5:
        w_stat, w_p = sp_stats.wilcoxon(base_vals, reasoning_vals, alternative="less")
    else:
        w_stat, w_p = float('nan'), float('nan')

    # Effect size: mean difference across layers
    diffs = [b - r for b, r in zip(base_vals, reasoning_vals)]
    mean_diff = np.mean(diffs)

    # Max interaction magnitude comparison
    base_max_neg = min(base_vals) if base_vals else 0
    reasoning_max_neg = min(reasoning_vals) if reasoning_vals else 0

    results = {
        "base_mean_interaction": float(np.mean(base_vals)),
        "reasoning_mean_interaction": float(np.mean(reasoning_vals)),
        "mean_difference": float(mean_diff),
        "wilcoxon_w": float(w_stat),
        "wilcoxon_p": float(w_p),
        "base_peak_negative": float(base_max_neg),
        "reasoning_peak_negative": float(reasoning_max_neg),
        "h2_supported": bool(not np.isnan(w_p) and w_p < 0.05 and mean_diff < -SESOI_PROB),
        "layer_diffs": {str(l): float(d) for l, d in zip(common_layers, diffs)},
    }

    with open(tables_dir / "h2_reasoning_regularization.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Base mean: {np.mean(base_vals):.4f}, Reasoning mean: {np.mean(reasoning_vals):.4f}")
    print(f"  Mean diff: {mean_diff:.4f}, Wilcoxon p={w_p:.4f}")
    print(f"  H2 supported: {results['h2_supported']}")
    return results


def _test_h3_geometry_prediction(li_df: pd.DataFrame, geo_df: pd.DataFrame,
                                  tables_dir: Path) -> Dict:
    """H3: Local geometry (PR, anisotropy) predicts interaction strength."""
    print("\n--- H3: Geometry predicts interaction ---")

    # Merge interaction data with geometry data
    merged = li_df.merge(
        geo_df[["model_name", "layer_target", "pr", "pr_normalized", "anisotropy"]],
        on=["model_name", "layer_target"], how="inner"
    )

    if len(merged) < 5:
        print(f"  Insufficient merged data ({len(merged)} rows)")
        results = {"merged_n": len(merged), "h3_supported": False}
        with open(tables_dir / "h3_geometry_prediction.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Correlation: PR vs interaction
    r_pr, p_pr = sp_stats.pearsonr(merged["pr"], merged["interaction"])
    r_pr_norm, p_pr_norm = sp_stats.pearsonr(merged["pr_normalized"], merged["interaction"])
    r_aniso, p_aniso = sp_stats.pearsonr(merged["anisotropy"], merged["interaction"])

    # Spearman for robustness
    rho_pr, rho_p_pr = sp_stats.spearmanr(merged["pr"], merged["interaction"])
    rho_aniso, rho_p_aniso = sp_stats.spearmanr(merged["anisotropy"], merged["interaction"])

    # Group-separated correlations
    group_corrs = {}
    for group in merged["group"].unique():
        gsub = merged[merged["group"] == group]
        if len(gsub) >= 4:
            r, p = sp_stats.pearsonr(gsub["pr"], gsub["interaction"])
            group_corrs[group] = {"pr_r": float(r), "pr_p": float(p), "n": len(gsub)}

    # Multiple regression: interaction ~ PR + anisotropy + group
    try:
        import statsmodels.api as sm
        X = merged[["pr_normalized", "anisotropy"]].copy()
        X["group_binary"] = (merged["group"] == "base_transformer").astype(int)
        X.insert(0, "intercept", 1)
        y = merged["interaction"].values
        ols = sm.OLS(y, X.values).fit()
        regression = {
            "r_squared": float(ols.rsquared),
            "adj_r_squared": float(ols.rsquared_adj),
            "f_stat": float(ols.fvalue),
            "f_p": float(ols.f_pvalue),
            "coefficients": {name: float(c) for name, c in
                           zip(X.columns, ols.params)},
            "p_values": {name: float(p) for name, p in
                        zip(X.columns, ols.pvalues)},
        }
    except Exception as e:
        print(f"  Regression failed: {e}")
        regression = {"error": str(e)}

    best_predictor = "pr" if abs(r_pr) > abs(r_aniso) else "anisotropy"
    best_r = max(abs(r_pr), abs(r_aniso))
    best_p = p_pr if best_predictor == "pr" else p_aniso

    results = {
        "merged_n": len(merged),
        "pr_pearson_r": float(r_pr), "pr_pearson_p": float(p_pr),
        "pr_norm_pearson_r": float(r_pr_norm), "pr_norm_pearson_p": float(p_pr_norm),
        "anisotropy_pearson_r": float(r_aniso), "anisotropy_pearson_p": float(p_aniso),
        "pr_spearman_rho": float(rho_pr), "pr_spearman_p": float(rho_p_pr),
        "anisotropy_spearman_rho": float(rho_aniso), "anisotropy_spearman_p": float(rho_p_aniso),
        "best_predictor": best_predictor,
        "best_r": float(best_r),
        "best_p": float(best_p),
        "group_correlations": group_corrs,
        "regression": regression,
        "h3_supported": bool(best_r > 0.5 and best_p < 0.05),
    }

    with open(tables_dir / "h3_geometry_prediction.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  PR vs interaction: r={r_pr:.3f}, p={p_pr:.4f}")
    print(f"  Anisotropy vs interaction: r={r_aniso:.3f}, p={p_aniso:.4f}")
    print(f"  Best predictor: {best_predictor} (|r|={best_r:.3f}, p={best_p:.4f})")
    print(f"  H3 supported: {results['h3_supported']}")
    return results


def _fit_layerwise_mixed_model(df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Mixed-effects model with layer position and group interactions."""
    print("\n--- Layerwise mixed-effects model ---")
    df = df.copy()
    df["surgery_bin"] = (df["surgery_strength"] > 0).astype(int)
    df["jitter_bin"] = (df["jitter_strength"] > 0).astype(int)
    df["interaction"] = df["surgery_bin"] * df["jitter_bin"]
    df["is_base"] = (df["group"] == "base_transformer").astype(int)
    df["layer_centered"] = df["layer_target"] - 0.50  # center at mid-layer
    df["group_x_int"] = df["is_base"] * df["interaction"]
    df["layer_x_int"] = df["layer_centered"] * df["interaction"]

    results = {}
    try:
        import statsmodels.api as sm
        exog = df[["surgery_bin", "jitter_bin", "interaction", "is_base",
                    "layer_centered", "group_x_int", "layer_x_int"]].copy()
        exog.insert(0, "intercept", 1)
        logit = sm.Logit(df["correct"].values, exog.values)
        fit = logit.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["model_name"]})
        names = list(exog.columns)
        for i, n in enumerate(names):
            results[f"coef_{n}"] = float(fit.params[i])
            results[f"se_{n}"] = float(fit.bse[i])
            results[f"p_{n}"] = float(fit.pvalues[i])

        results["interaction_beta"] = float(fit.params[names.index("interaction")])
        results["interaction_p"] = float(fit.pvalues[names.index("interaction")])
        results["group_x_int_beta"] = float(fit.params[names.index("group_x_int")])
        results["group_x_int_p"] = float(fit.pvalues[names.index("group_x_int")])
        results["layer_x_int_beta"] = float(fit.params[names.index("layer_x_int")])
        results["layer_x_int_p"] = float(fit.pvalues[names.index("layer_x_int")])
        results["method"] = "Logit_ClusterRobust"

        print(f"  Interaction: beta={results['interaction_beta']:.4f}, p={results['interaction_p']:.4f}")
        print(f"  Group x Int: beta={results['group_x_int_beta']:.4f}, p={results['group_x_int_p']:.4f}")
        print(f"  Layer x Int: beta={results['layer_x_int_beta']:.4f}, p={results['layer_x_int_p']:.4f}")
    except Exception as e:
        print(f"  Mixed model failed: {e}")
        results["method"] = "failed"

    with open(tables_dir / "mixed_effects_layerwise.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _bootstrap_by_layer(li_df: pd.DataFrame, tables_dir: Path, n_boot: int = 5000) -> Dict:
    """Bootstrap interaction CIs per layer position and group."""
    print("\n--- Bootstrap by layer (5000) ---")
    rng = np.random.default_rng(42)
    results = {"layers": {}}

    for lt in sorted(li_df["layer_target"].unique()):
        layer_data = li_df[li_df["layer_target"] == lt]
        models = layer_data["model_name"].values
        model_ints = layer_data.set_index("model_name")["interaction"]
        n = len(models)

        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(models, size=n, replace=True)
            boot_means.append(model_ints[sample].mean())

        boot_means = np.array(boot_means)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        results["layers"][str(lt)] = {
            "mean": float(model_ints.mean()),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
        }
        print(f"  L={lt:.2f}: mean={model_ints.mean():.4f}, "
              f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    with open(tables_dir / "bootstrap_by_layer.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _permutation_test(df: pd.DataFrame, tables_dir: Path, n_perm: int = 5000) -> Dict:
    """Permutation test on global interaction (pooled across layers)."""
    print("\n--- Permutation test (5000) ---")
    rng = np.random.default_rng(42)

    # Observed: mean interaction across all model-layer combos
    li = _compute_layer_interactions(df)
    li_df = pd.DataFrame(li)
    observed = li_df["interaction"].mean()

    null_dist = []
    for i in range(n_perm):
        df_perm = df.copy()
        for mn in df_perm["model_name"].unique():
            mask = df_perm["model_name"] == mn
            idx = df_perm.index[mask]
            df_perm.loc[idx, "surgery_strength"] = rng.permutation(
                df_perm.loc[idx, "surgery_strength"].values)
            df_perm.loc[idx, "jitter_strength"] = rng.permutation(
                df_perm.loc[idx, "jitter_strength"].values)
        perm_li = _compute_layer_interactions(df_perm)
        null_dist.append(np.mean([x["interaction"] for x in perm_li]))
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


def _leave_one_out_by_group(li_df: pd.DataFrame, tables_dir: Path) -> Dict:
    """LOO analysis per group."""
    print("\n--- Leave-one-out by group ---")
    results = {}
    for group in li_df["group"].unique():
        gdf = li_df[li_df["group"] == group]
        model_means = gdf.groupby("model_name")["interaction"].mean()
        full_mean = model_means.mean()
        loo_rows = []
        for mn in model_means.index:
            loo_mean = model_means.drop(mn).mean()
            loo_rows.append({"excluded": mn, "loo_mean": float(loo_mean),
                             "delta": float(loo_mean - full_mean)})
        results[group] = {
            "full_mean": float(full_mean),
            "loo_std": float(pd.DataFrame(loo_rows)["loo_mean"].std()) if len(loo_rows) > 1 else 0,
            "models": loo_rows,
        }
        print(f"  {group}: mean={full_mean:.4f}, LOO std={results[group]['loo_std']:.4f}")

    with open(tables_dir / "loo_by_group.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _split_comparison(df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Cal vs holdout split comparison."""
    print("\n--- Cal vs holdout ---")
    cal_li = _compute_layer_interactions(df[df["split"] == "cal"])
    hold_li = _compute_layer_interactions(df[df["split"] == "hold"])

    cal_mean = np.mean([x["interaction"] for x in cal_li]) if cal_li else 0
    hold_mean = np.mean([x["interaction"] for x in hold_li]) if hold_li else 0

    results = {
        "calibration_interaction": float(cal_mean),
        "holdout_interaction": float(hold_mean),
        "gap_pp": float(abs(cal_mean - hold_mean) * 100),
    }

    with open(tables_dir / "split_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Cal: {cal_mean:.4f}, Hold: {hold_mean:.4f} (gap: {results['gap_pp']:.1f}pp)")
    return results


def _main_effects_by_layer(df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Main effects by layer position."""
    print("\n--- Main effects by layer ---")
    results = {}
    for lt in sorted(df["layer_target"].unique()):
        ldf = df[df["layer_target"] == lt]
        acc_c = ldf[(ldf["surgery_strength"] == 0) & (ldf["jitter_strength"] == 0)]["correct"].mean()
        acc_s = ldf[(ldf["surgery_strength"] > 0) & (ldf["jitter_strength"] == 0)]["correct"].mean()
        acc_j = ldf[(ldf["surgery_strength"] == 0) & (ldf["jitter_strength"] > 0)]["correct"].mean()
        results[str(lt)] = {
            "baseline": float(acc_c),
            "surgery_effect": float(acc_s - acc_c),
            "jitter_effect": float(acc_j - acc_c),
        }
        print(f"  L={lt:.2f}: base={acc_c:.3f}, surg={acc_s-acc_c:+.3f}, jit={acc_j-acc_c:+.3f}")

    with open(tables_dir / "main_effects_by_layer.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Figures ──────────────────────────────────────────────────────────────
GROUP_COLORS = {"base_transformer": "#2196F3", "reasoning_tuned": "#FF5722"}
MODEL_MARKERS = {"Qwen2.5-7B": "o", "Qwen3-8B": "s", "OLMo2-7B": "^",
                 "DSR1-7B": "D", "DSR1-Llama-8B": "v"}


def _plot_layerwise_interaction(li_df: pd.DataFrame, fig_dir: Path):
    """Main figure: interaction profile by layer for each group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for group, color in GROUP_COLORS.items():
        gdf = li_df[li_df["group"] == group]
        if len(gdf) == 0:
            continue
        profile = gdf.groupby("layer_target")["interaction"].agg(["mean", "std", "count"])
        profile["se"] = profile["std"] / np.sqrt(profile["count"])

        ax.plot(profile.index, profile["mean"], "o-", color=color,
                label=group.replace("_", " ").title(), linewidth=2, markersize=8)
        ax.fill_between(profile.index,
                        profile["mean"] - 1.96 * profile["se"],
                        profile["mean"] + 1.96 * profile["se"],
                        alpha=0.2, color=color)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhspan(-SESOI_PROB, SESOI_PROB, alpha=0.15, color="green", label="SESOI")
    ax.set_xlabel("Layer Position (fraction)", fontsize=12)
    ax.set_ylabel("Interaction Effect", fontsize=12)
    ax.set_title("Exp-016: Layerwise Surgery x Jitter Interaction\n"
                 "Base Transformer vs Reasoning-Tuned (7B+)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(LAYER_FRACTIONS)
    plt.tight_layout()
    plt.savefig(fig_dir / "layerwise_interaction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> layerwise_interaction.png")


def _plot_group_comparison(li_df: pd.DataFrame, fig_dir: Path):
    """Per-model interaction profiles overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, group in enumerate(["base_transformer", "reasoning_tuned"]):
        ax = axes[ax_idx]
        gdf = li_df[li_df["group"] == group]
        for model_name in gdf["model_name"].unique():
            mdf = gdf[gdf["model_name"] == model_name]
            mdf_sorted = mdf.sort_values("layer_target")
            marker = MODEL_MARKERS.get(model_name, "o")
            ax.plot(mdf_sorted["layer_target"], mdf_sorted["interaction"],
                    f"{marker}-", label=model_name, linewidth=1.5, markersize=7)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.axhspan(-SESOI_PROB, SESOI_PROB, alpha=0.15, color="green")
        ax.set_xlabel("Layer Position")
        ax.set_ylabel("Interaction Effect" if ax_idx == 0 else "")
        ax.set_title(group.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAYER_FRACTIONS)

    plt.suptitle("Exp-016: Per-Model Interaction Profiles", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_interaction_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> model_interaction_profiles.png")


def _plot_geometry_profiles(geo_df: pd.DataFrame, fig_dir: Path):
    """PR and anisotropy profiles by layer for each group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col_idx, metric in enumerate(["pr_normalized", "anisotropy"]):
        ax = axes[col_idx]
        for group, color in GROUP_COLORS.items():
            gdf = geo_df[geo_df["group"] == group]
            if len(gdf) == 0:
                continue
            profile = gdf.groupby("layer_target")[metric].agg(["mean", "std"])
            ax.plot(profile.index, profile["mean"], "o-", color=color,
                    label=group.replace("_", " ").title(), linewidth=2, markersize=8)
            ax.fill_between(profile.index,
                            profile["mean"] - profile["std"],
                            profile["mean"] + profile["std"],
                            alpha=0.2, color=color)

        ax.set_xlabel("Layer Position")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Layerwise {metric.replace('_', ' ').title()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAYER_FRACTIONS)

    plt.suptitle("Exp-016: Representation Geometry Profiles (Clean Pass)", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "geometry_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> geometry_profiles.png")


def _plot_geometry_vs_interaction(li_df: pd.DataFrame, geo_df: pd.DataFrame,
                                   h3_results: Dict, fig_dir: Path):
    """Scatter: PR and anisotropy vs interaction at each layer-model."""
    merged = li_df.merge(
        geo_df[["model_name", "layer_target", "pr", "pr_normalized", "anisotropy"]],
        on=["model_name", "layer_target"], how="inner"
    )
    if len(merged) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (metric, label) in enumerate([("pr_normalized", "PR / d_model"),
                                               ("anisotropy", "Anisotropy")]):
        ax = axes[ax_idx]
        for group, color in GROUP_COLORS.items():
            gsub = merged[merged["group"] == group]
            for mn in gsub["model_name"].unique():
                msub = gsub[gsub["model_name"] == mn]
                marker = MODEL_MARKERS.get(mn, "o")
                ax.scatter(msub[metric], msub["interaction"],
                           c=color, marker=marker, s=60, alpha=0.7, label=mn)

        # Regression line
        r_key = f"{metric.split('_')[0]}_pearson_r"
        p_key = f"{metric.split('_')[0]}_pearson_p"
        if r_key in h3_results:
            z = np.polyfit(merged[metric], merged["interaction"], 1)
            x_line = np.linspace(merged[metric].min(), merged[metric].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5,
                    label=f"r={h3_results.get(r_key, 0):.3f}")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Interaction Effect")
        ax.set_title(f"{label} vs Interaction")
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate legend
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp-016: Geometry vs Interaction", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "geometry_vs_interaction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> geometry_vs_interaction.png")


def _plot_heatmap(li_df: pd.DataFrame, fig_dir: Path):
    """Heatmap: model x layer interaction values."""
    pivot = li_df.pivot_table(values="interaction", index="model_name",
                               columns="layer_target", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Layer Position")
    ax.set_ylabel("Model")
    ax.set_title("Exp-016: Interaction Heatmap (Model x Layer)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(val) > 0.05 else "black")

    plt.colorbar(im, ax=ax, label="Interaction Effect")
    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_heatmap.png")


# ── Key findings + verdict ───────────────────────────────────────────────
def _write_findings(df, li_df, geo_df, group_layer, h1, h2, h3,
                    mixed, boot, perm, loo, split, main_effects, out_dir):
    lines = []
    lines.append(f"models_tested: {df['model_name'].nunique()}")
    lines.append(f"total_trials: {len(df)}")
    lines.append(f"geometry_measurements: {len(geo_df)}")
    lines.append(f"groups: {', '.join(sorted(df['group'].unique()))}")
    lines.append(f"domains: {', '.join(sorted(df['domain'].unique()))}")
    lines.append(f"layer_fractions: {LAYER_FRACTIONS}")
    lines.append(f"surgery_strength: {SURGERY_STRENGTH}")
    lines.append(f"jitter_strength: {JITTER_STRENGTH}")
    lines.append(f"seeds: {len(SEEDS)}")
    lines.append(f"prompts: {len(df['prompt_id'].unique())}")

    # Global interaction
    global_int = li_df["interaction"].mean()
    lines.append(f"global_mean_interaction: {global_int:.5f}")

    # Group interactions
    for group in sorted(li_df["group"].unique()):
        gint = li_df[li_df["group"] == group]["interaction"].mean()
        lines.append(f"group_{group}_interaction: {gint:.5f}")

    # Per-model mean interactions (across layers)
    for mn in sorted(li_df["model_name"].unique()):
        mint = li_df[li_df["model_name"] == mn]["interaction"].mean()
        lines.append(f"model_{mn}_mean_interaction: {mint:.5f}")

    # Mixed-effects
    if "interaction_beta" in mixed:
        lines.append(f"mixed_interaction_beta: {mixed['interaction_beta']:.4f}")
        lines.append(f"mixed_interaction_p: {mixed['interaction_p']:.4f}")
        lines.append(f"mixed_group_x_int_beta: {mixed['group_x_int_beta']:.4f}")
        lines.append(f"mixed_group_x_int_p: {mixed['group_x_int_p']:.4f}")
        lines.append(f"mixed_layer_x_int_beta: {mixed['layer_x_int_beta']:.4f}")
        lines.append(f"mixed_layer_x_int_p: {mixed['layer_x_int_p']:.4f}")

    # Bootstrap
    for lt, bdata in boot.get("layers", {}).items():
        lines.append(f"bootstrap_L{lt}_ci: [{bdata['ci_lo']:.4f}, {bdata['ci_hi']:.4f}]")

    # Permutation
    lines.append(f"permutation_p: {perm['p_perm']:.4f}")

    # Split
    lines.append(f"calibration_interaction: {split['calibration_interaction']:.4f}")
    lines.append(f"holdout_interaction: {split['holdout_interaction']:.4f}")
    lines.append(f"cal_holdout_gap_pp: {split['gap_pp']:.1f}")

    # Hypotheses
    lines.append(f"h1_layer_localization: {h1['h1_supported']}")
    lines.append(f"h1_peak_layer: {h1['peak_layer']:.2f}")
    lines.append(f"h1_peak_interaction: {h1['peak_interaction']:.4f}")
    lines.append(f"h1_localization_ratio: {h1['localization_ratio']:.2f}")

    lines.append(f"h2_reasoning_regularization: {h2['h2_supported']}")
    lines.append(f"h2_base_mean: {h2['base_mean_interaction']:.4f}")
    lines.append(f"h2_reasoning_mean: {h2['reasoning_mean_interaction']:.4f}")
    lines.append(f"h2_wilcoxon_p: {h2['wilcoxon_p']:.4f}")

    lines.append(f"h3_geometry_prediction: {h3['h3_supported']}")
    if "best_predictor" in h3:
        lines.append(f"h3_best_predictor: {h3['best_predictor']}")
        lines.append(f"h3_best_r: {h3['best_r']:.3f}")
        lines.append(f"h3_best_p: {h3['best_p']:.4f}")
    if "regression" in h3 and "r_squared" in h3["regression"]:
        lines.append(f"h3_regression_r2: {h3['regression']['r_squared']:.4f}")

    # Geometry summaries
    for group in sorted(geo_df["group"].unique()):
        gdf = geo_df[geo_df["group"] == group]
        lines.append(f"geometry_{group}_mean_pr: {gdf['pr'].mean():.2f}")
        lines.append(f"geometry_{group}_mean_anisotropy: {gdf['anisotropy'].mean():.4f}")

    # Verdict
    # Decision rules from Codex spec
    base_coupling = h2.get("base_mean_interaction", 0)
    reasoning_coupling = h2.get("reasoning_mean_interaction", 0)
    base_stronger = abs(base_coupling) > abs(reasoning_coupling)

    if h1["h1_supported"] and h2["h2_supported"]:
        verdict = "TRAINING_REGULARIZATION"
        explanation = ("Base transformers show layer-localized coupling concentrated at "
                      f"L={h1['peak_layer']:.2f}. Reasoning training compresses/regularizes "
                      "this vulnerable band, eliminating the interaction.")
    elif h3["h3_supported"]:
        verdict = "GEOMETRIC_BOTTLENECK"
        explanation = (f"Interaction strength predicted by local {h3.get('best_predictor', 'geometry')} "
                      f"(r={h3.get('best_r', 0):.3f}). The coupling arises from geometric "
                      "properties of hidden states, not training method per se.")
    elif (h2["h2_supported"] and not h1["h1_supported"]):
        verdict = "TRAINING_REGULARIZATION (uniform across layers)"
        explanation = ("Reasoning training uniformly reduces coupling across all layers, "
                      "not just at specific positions.")
    elif (not h2["h2_supported"] and base_stronger and
          abs(reasoning_coupling) < SESOI_PROB):
        verdict = "TRAINING_SPECIFIC_COUPLING (weak evidence)"
        explanation = ("Base transformers couple more than reasoning-tuned, but formal "
                      "significance not reached. Directional evidence only.")
    else:
        verdict = "NO_CLEAR_MECHANISM"
        explanation = ("Neither layer localization, training regularization, nor geometric "
                      "prediction clearly explains the coupling difference.")

    lines.append(f"verdict: {verdict}")
    lines.append(f"verdict_explanation: {explanation}")

    # Business + scientific implications
    lines.append(
        f"business_implication: {verdict} — "
        f"Base transformers at 7B+ show interaction={base_coupling:.4f} "
        f"while reasoning-tuned show {reasoning_coupling:.4f}. "
        f"{'Reasoning fine-tuning appears to regularize the coupling, suggesting deployment testing can be simpler for reasoning-tuned models.' if h2['h2_supported'] else 'Mechanism unclear; both groups should undergo full testing.'}"
    )
    lines.append(
        f"scientific_implication: Layerwise decomposition at 5 positions reveals "
        f"{'layer-localized coupling at L=' + str(h1['peak_layer']) + ' for base transformers' if h1['h1_supported'] else 'no significant layer localization'}. "
        f"{'Geometry (PR/anisotropy) predicts interaction (r=' + str(h3.get('best_r', 0)) + ')' if h3['h3_supported'] else 'Geometry does not significantly predict interaction'}. "
        f"The base vs reasoning difference is {'statistically confirmed' if h2['h2_supported'] else 'directional but not formally significant'}."
    )

    (out_dir / "key_findings.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFindings written. Verdict: {verdict}")
    print(f"  {explanation}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Exp-016: Layerwise Coupling Mechanism in 7B+ Transformers")
    parser.add_argument("--stage", choices=["collect", "geometry", "analyze", "all"],
                        default="all")
    parser.add_argument("--outdir", default="analysis/orthogonality_mechanism_016")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("collect", "all"):
        collect_factorial(out_dir)

    if args.stage in ("geometry", "all"):
        collect_geometry(out_dir)

    if args.stage in ("analyze", "all"):
        run_analysis(out_dir)


if __name__ == "__main__":
    main()
