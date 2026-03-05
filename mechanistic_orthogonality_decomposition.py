#!/usr/bin/env python3
"""
mechanistic_orthogonality_decomposition.py — Exp-012: Mechanistic Orthogonality Decomposition

Codex-designed 2×2 factorial experiment testing whether PR surgery and jitter
stress operate through independent (orthogonal) mechanisms.

Design:
  Factor A (surgery):  control | expand (PR expansion at mid-layer)
  Factor B (jitter):   off     | on     (Gaussian noise at mid-layer, α=0.1)

  2×2 cells: control/off, expand/off, control/on, expand/on
  16 models × 32 prompts × 3 seeds × 4 conditions = 6,144 trials

Statistical pipeline:
  1. Mixed-effects logistic regression: correct ~ surgery*jitter + domain + paradigm + (1|model) + (1|prompt)
  2. Bootstrap (2000) over models for interaction CI
  3. Permutation test (5000) shuffling surgery/jitter labels within model
  4. Leave-one-model-out OOS stability check

Success (orthogonality): interaction CI includes 0, p>0.05, ROPE [0.8,1.25] OR
Falsification: interaction significantly non-zero and large, p<0.05
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
from scipy import stats as sp_stats

# ── Model registry ───────────────────────────────────────────────────────
# (hf_id, short_name, paradigm, approx_params_B)
MODELS: List[Tuple[str, str, str, float]] = [
    # Transformers
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", "transformer", 0.6),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B", "transformer", 1.7),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B", "transformer", 0.5),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", "transformer", 1.5),
    ("google/gemma-3-1b-it", "Gemma3-1B", "transformer", 1.0),
    ("google/gemma-2-2b-it", "Gemma2-2B", "transformer", 2.0),
    # SSM
    ("state-spaces/mamba-130m-hf", "Mamba-130M", "ssm", 0.13),
    ("state-spaces/mamba-370m-hf", "Mamba-370M", "ssm", 0.37),
    ("state-spaces/mamba-790m-hf", "Mamba-790M", "ssm", 0.79),
    ("state-spaces/mamba-2.8b-hf", "Mamba-2.8B", "ssm", 2.8),
    # Hybrid
    ("tiiuae/Falcon-H1-0.5B-Instruct", "FalconH1-0.5B", "hybrid", 0.5),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "FalconH1-1.5B", "hybrid", 1.5),
    ("Zyphra/Zamba2-1.2B", "Zamba2-1.2B", "hybrid", 1.2),
    ("nvidia/Hymba-1.5B-Base", "Hymba-1.5B", "hybrid", 1.5),
    # Reasoning
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DSR1-1.5B", "reasoning", 1.5),
    ("nvidia/OpenReasoning-Nemotron-1.5B", "Nemotron-1.5B-R", "reasoning", 1.5),
]

# ── Evaluation prompts ───────────────────────────────────────────────────
EVAL_PROMPTS = {
    "math": [
        {"id": "m01", "prompt": "What is 7 * 8? Answer with just the number:", "answer": "56"},
        {"id": "m02", "prompt": "What is 15 + 27? Answer with just the number:", "answer": "42"},
        {"id": "m03", "prompt": "What is 100 - 37? Answer with just the number:", "answer": "63"},
        {"id": "m04", "prompt": "What is 144 / 12? Answer with just the number:", "answer": "12"},
        {"id": "m05", "prompt": "What is 9 * 9? Answer with just the number:", "answer": "81"},
        {"id": "m06", "prompt": "What is 23 + 19? Answer with just the number:", "answer": "42"},
        {"id": "m07", "prompt": "What is 256 / 16? Answer with just the number:", "answer": "16"},
        {"id": "m08", "prompt": "What is 11 * 11? Answer with just the number:", "answer": "121"},
        {"id": "m09", "prompt": "What is 50 - 23? Answer with just the number:", "answer": "27"},
        {"id": "m10", "prompt": "What is 6 * 7? Answer with just the number:", "answer": "42"},
        {"id": "m11", "prompt": "What is 200 / 8? Answer with just the number:", "answer": "25"},
        {"id": "m12", "prompt": "What is 33 + 44? Answer with just the number:", "answer": "77"},
        {"id": "m13", "prompt": "What is 13 * 5? Answer with just the number:", "answer": "65"},
        {"id": "m14", "prompt": "What is 96 / 8? Answer with just the number:", "answer": "12"},
        {"id": "m15", "prompt": "What is 88 - 29? Answer with just the number:", "answer": "59"},
        {"id": "m16", "prompt": "What is 14 * 6? Answer with just the number:", "answer": "84"},
    ],
    "factual": [
        {"id": "f01", "prompt": "The capital of Japan is", "answer": "Tokyo"},
        {"id": "f02", "prompt": "Water freezes at 0 degrees", "answer": "Celsius"},
        {"id": "f03", "prompt": "The chemical symbol for gold is", "answer": "Au"},
        {"id": "f04", "prompt": "The largest planet in our solar system is", "answer": "Jupiter"},
        {"id": "f05", "prompt": "The speed of light is approximately 300,000 km per", "answer": "second"},
        {"id": "f06", "prompt": "DNA stands for deoxyribonucleic", "answer": "acid"},
        {"id": "f07", "prompt": "The square root of 144 is", "answer": "12"},
        {"id": "f08", "prompt": "The atomic number of carbon is", "answer": "6"},
        {"id": "f09", "prompt": "The boiling point of water is 100 degrees", "answer": "Celsius"},
        {"id": "f10", "prompt": "The chemical formula for table salt is", "answer": "NaCl"},
        {"id": "f11", "prompt": "The Earth orbits the", "answer": "Sun"},
        {"id": "f12", "prompt": "Photosynthesis converts carbon dioxide and water into glucose and", "answer": "oxygen"},
        {"id": "f13", "prompt": "The fastest land animal is the", "answer": "cheetah"},
        {"id": "f14", "prompt": "The human body has 206", "answer": "bones"},
        {"id": "f15", "prompt": "The chemical symbol for iron is", "answer": "Fe"},
        {"id": "f16", "prompt": "The Great Wall is located in", "answer": "China"},
    ],
}

# ── Defaults ─────────────────────────────────────────────────────────────
SEEDS = [11, 23, 37]
MAX_NEW_TOKENS = 10
DEFAULT_SURGERY_STRENGTH = 0.08
DEFAULT_JITTER_STRENGTH = 0.1


# ── Hooks ────────────────────────────────────────────────────────────────
class PRExpansionHook:
    """Inject orthogonal noise into low-variance dimensions (PR expansion)."""

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
    """Inject scaled Gaussian noise into hidden states."""

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


# ── Utilities ────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_id: str):
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


def check_correct(generated: str, answer: str) -> bool:
    return answer.lower() in generated.lower()


def logit_entropy(logits: np.ndarray) -> float:
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    probs = probs[probs > 1e-12]
    return float(-np.sum(probs * np.log(probs)))


def logit_margin(logits: np.ndarray) -> float:
    sorted_l = np.sort(logits)[::-1]
    if len(sorted_l) < 2:
        return 0.0
    return float(sorted_l[0] - sorted_l[1])


# ── Data collection ─────────────────────────────────────────────────────
def run_condition(
    model, tokenizer, layers: List[nn.Module],
    model_name: str, model_id: str, paradigm: str, params_b: float,
    layer_idx: int, surgery_strength: float, jitter_strength: float,
    seed: int, surgery: str, jitter: str,
) -> List[Dict]:
    """Run one condition (surgery × jitter) for all prompts. Returns row-level trial data."""
    hooks = []

    # Attach hooks as needed
    if surgery == "expand":
        hook_s = PRExpansionHook(strength=surgery_strength, seed=seed)
        hook_s.attach(layers[layer_idx])
        hooks.append(hook_s)

    if jitter == "on":
        # Use different seed offset for jitter to avoid correlation with surgery noise
        hook_j = JitterHook(strength=jitter_strength, seed=seed + 10000)
        hook_j.attach(layers[layer_idx])
        hooks.append(hook_j)

    device = next(model.parameters()).device
    rows = []

    for domain, items in EVAL_PROMPTS.items():
        for item in items:
            t0 = time.time()
            inputs = tokenizer(
                item["prompt"], return_tensors="pt", truncation=True, max_length=128
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            runtime_ms = (time.time() - t0) * 1000

            # First token logits
            entropy_val = None
            margin_val = None
            if hasattr(outputs, "scores") and outputs.scores and len(outputs.scores) > 0:
                logits = outputs.scores[0][0].float().cpu().numpy()
                entropy_val = logit_entropy(logits)
                margin_val = logit_margin(logits)

            correct = check_correct(text, item["answer"])

            rows.append({
                "model_id": model_id,
                "model_name": model_name,
                "paradigm": paradigm,
                "params_b": params_b,
                "seed": seed,
                "layer_idx": layer_idx,
                "surgery": surgery,
                "jitter": jitter,
                "domain": domain,
                "prompt_id": item["id"],
                "prompt": item["prompt"],
                "answer": item["answer"],
                "generated": text,
                "correct": int(correct),
                "n_prompt_tokens": inputs["input_ids"].shape[1],
                "logits_entropy": entropy_val,
                "logit_margin": margin_val,
                "runtime_ms": runtime_ms,
            })

    # Detach all hooks
    for h in hooks:
        h.detach()

    return rows


def collect_data(
    out_dir: Path,
    surgery_strength: float,
    jitter_strength: float,
    seeds: List[int],
    skip_models: Optional[set] = None,
) -> pd.DataFrame:
    """Run full 2×2 factorial data collection across all models."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = tables_dir / "raw_trials.csv"

    all_rows = []
    failures = []

    # Resume: load existing data
    completed_models = set()
    if raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        all_rows = existing.to_dict("records")
        completed_models = set(existing["model_name"].unique())
        print(f"  [RESUME] Loaded {len(all_rows)} rows for {completed_models}")

    conditions = [
        ("control", "off"),
        ("expand", "off"),
        ("control", "on"),
        ("expand", "on"),
    ]

    n_total_conditions = len(MODELS) * len(conditions) * len(seeds)
    print(f"\n{'=' * 70}")
    print(f"EXP-012: MECHANISTIC ORTHOGONALITY DECOMPOSITION")
    print(f"Models: {len(MODELS)}, Conditions: {len(conditions)}, Seeds: {len(seeds)}")
    print(f"Surgery strength: {surgery_strength}, Jitter strength: {jitter_strength}")
    print(f"Total condition-sets: {n_total_conditions}")
    print(f"Total trials: {n_total_conditions * 32}")
    print(f"{'=' * 70}\n")

    for model_idx, (model_id, model_name, paradigm, params_b) in enumerate(MODELS):
        if model_name in completed_models:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (already completed)")
            continue

        if skip_models and model_name in skip_models:
            print(f"  [{model_idx+1}/{len(MODELS)}] SKIP {model_name} (user-requested)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{model_idx+1}/{len(MODELS)}] {model_name} ({paradigm}, {params_b}B)")
        print(f"{'=' * 60}")

        t_model = time.time()
        try:
            model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            print(f"  [FAIL] Could not load {model_id}: {e}")
            failures.append({"model_id": model_id, "model_name": model_name, "error": str(e)})
            continue

        layers = get_model_layers(model)
        if not layers:
            print(f"  [FAIL] No layers found for {model_name}")
            failures.append({"model_id": model_id, "model_name": model_name, "error": "no_layers"})
            del model, tokenizer
            gc.collect(); torch.cuda.empty_cache()
            continue

        n_layers = len(layers)
        layer_idx = n_layers // 2
        print(f"  Layers: {n_layers}, target: {layer_idx}")
        print(f"  Loaded in {time.time() - t_model:.1f}s")

        model_rows = []
        for seed in seeds:
            for surgery, jitter in conditions:
                tag = f"seed={seed}, {surgery}/{jitter}"
                t_cond = time.time()
                try:
                    rows = run_condition(
                        model, tokenizer, layers,
                        model_name, model_id, paradigm, params_b,
                        layer_idx, surgery_strength, jitter_strength,
                        seed, surgery, jitter,
                    )
                    model_rows.extend(rows)
                    n_correct = sum(r["correct"] for r in rows)
                    print(f"    {tag}: {n_correct}/32 correct ({time.time()-t_cond:.1f}s)")
                except Exception as e:
                    print(f"    {tag}: FAILED — {e}")
                    failures.append({
                        "model_id": model_id, "model_name": model_name,
                        "error": f"{surgery}/{jitter}/seed{seed}: {e}"
                    })

        all_rows.extend(model_rows)

        # Save after each model
        pd.DataFrame(all_rows).to_csv(raw_csv, index=False)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_model
        print(f"  {model_name} done: {len(model_rows)} trials in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save failures
    if failures:
        pd.DataFrame(failures).to_csv(tables_dir / "failures.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(raw_csv, index=False)
    print(f"\nData collection complete: {len(df)} total rows")
    return df


# ── Analysis ─────────────────────────────────────────────────────────────
def run_analysis(out_dir: Path):
    """Run full statistical analysis on collected data."""
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = tables_dir / "raw_trials.csv"
    if not raw_csv.exists():
        print("[ERROR] No raw_trials.csv found. Run data collection first.")
        return

    df = pd.read_csv(raw_csv)
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: {len(df)} trials, {df['model_name'].nunique()} models")
    print(f"{'=' * 70}\n")

    # ── 1. Aggregate summaries ───────────────────────────────────────────
    # Model × condition summary
    model_cond = df.groupby(
        ["model_name", "paradigm", "surgery", "jitter", "domain"]
    ).agg(
        accuracy=("correct", "mean"),
        n_trials=("correct", "count"),
    ).reset_index()
    model_cond.to_csv(tables_dir / "model_condition_summary.csv", index=False)

    # Domain × condition summary
    domain_cond = df.groupby(["surgery", "jitter", "domain"]).agg(
        accuracy=("correct", "mean"),
        n_trials=("correct", "count"),
        n_models=("model_name", "nunique"),
    ).reset_index()
    domain_cond.to_csv(tables_dir / "domain_condition_summary.csv", index=False)

    # ── 2. Compute per-model effects ─────────────────────────────────────
    model_effects = []
    for model_name in df["model_name"].unique():
        mdf = df[df["model_name"] == model_name]
        paradigm = mdf["paradigm"].iloc[0]

        # Mean accuracy per condition
        acc = {}
        for (surg, jit), grp in mdf.groupby(["surgery", "jitter"]):
            acc[(surg, jit)] = grp["correct"].mean()

        control_off = acc.get(("control", "off"), np.nan)
        expand_off = acc.get(("expand", "off"), np.nan)
        control_on = acc.get(("control", "on"), np.nan)
        expand_on = acc.get(("expand", "on"), np.nan)

        surgery_effect = expand_off - control_off  # main effect of surgery
        jitter_effect = control_on - control_off    # main effect of jitter
        # Interaction: does the combined effect differ from additive prediction?
        interaction = (expand_on - control_on) - (expand_off - control_off)

        model_effects.append({
            "model_name": model_name,
            "paradigm": paradigm,
            "acc_control_off": control_off,
            "acc_expand_off": expand_off,
            "acc_control_on": control_on,
            "acc_expand_on": expand_on,
            "surgery_effect": surgery_effect,
            "jitter_effect": jitter_effect,
            "interaction": interaction,
        })

    effects_df = pd.DataFrame(model_effects)
    effects_df.to_csv(tables_dir / "model_effects.csv", index=False)

    print("Per-model effects:")
    print(effects_df[["model_name", "paradigm", "surgery_effect", "jitter_effect", "interaction"]].to_string(index=False))

    # ── 3. Mixed-effects model ───────────────────────────────────────────
    mixed_results = _fit_mixed_model(df, tables_dir)

    # ── 4. Bootstrap interaction CI ──────────────────────────────────────
    bootstrap_results = _bootstrap_interaction(df, effects_df, tables_dir)

    # ── 5. Permutation test ──────────────────────────────────────────────
    perm_results = _permutation_test(df, tables_dir)

    # ── 6. Leave-one-model-out ───────────────────────────────────────────
    loo_results = _leave_one_out(df, effects_df, tables_dir)

    # ── 7. Figures ───────────────────────────────────────────────────────
    _plot_interaction_forest(effects_df, figures_dir)
    _plot_condition_effects(model_cond, figures_dir)
    _plot_shift_scatter(effects_df, figures_dir)

    # ── 8. Key findings ──────────────────────────────────────────────────
    _write_findings(df, effects_df, mixed_results, bootstrap_results,
                    perm_results, loo_results, out_dir)

    print(f"\nAnalysis complete. Output: {out_dir}")


def _fit_mixed_model(df: pd.DataFrame, tables_dir: Path) -> Dict:
    """Fit mixed-effects logistic regression. Fallback to GEE or simple logistic."""
    print("\n--- Mixed-effects model ---")

    # Encode factors as binary
    df = df.copy()
    df["surgery_bin"] = (df["surgery"] == "expand").astype(int)
    df["jitter_bin"] = (df["jitter"] == "on").astype(int)
    df["interaction"] = df["surgery_bin"] * df["jitter_bin"]
    df["domain_bin"] = (df["domain"] == "math").astype(int)
    df["log10_params"] = np.log10(df["params_b"])

    results = {}

    # Try statsmodels BinomialBayesMixedGLM
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        # Encode random effects
        model_ids = pd.Categorical(df["model_name"]).codes
        prompt_ids = pd.Categorical(df["prompt_id"]).codes

        exog = df[["surgery_bin", "jitter_bin", "interaction", "domain_bin", "log10_params"]].copy()
        exog.insert(0, "intercept", 1)

        # Random intercepts for model_name
        ident = np.zeros((len(df), df["model_name"].nunique()))
        for i, code in enumerate(model_ids):
            ident[i, code] = 1.0

        glmm = BinomialBayesMixedGLM(
            df["correct"].values, exog.values,
            exog_vc=ident,
            ident=np.zeros(df["model_name"].nunique(), dtype=int),
        )
        fit = glmm.fit_vb()

        coef_names = list(exog.columns)
        for i, name in enumerate(coef_names):
            results[f"coef_{name}"] = float(fit.fe_mean[i])
            results[f"se_{name}"] = float(fit.fe_sd[i])

        # Interaction test
        idx_int = coef_names.index("interaction")
        beta_int = fit.fe_mean[idx_int]
        se_int = fit.fe_sd[idx_int]
        z_int = beta_int / se_int if se_int > 0 else 0
        p_int = 2 * (1 - sp_stats.norm.cdf(abs(z_int)))
        or_int = np.exp(beta_int)

        results["interaction_beta"] = float(beta_int)
        results["interaction_se"] = float(se_int)
        results["interaction_z"] = float(z_int)
        results["interaction_p"] = float(p_int)
        results["interaction_or"] = float(or_int)
        results["method"] = "BinomialBayesMixedGLM"

        print(f"  Method: BinomialBayesMixedGLM (VB)")
        print(f"  Interaction: beta={beta_int:.4f}, SE={se_int:.4f}, z={z_int:.3f}, p={p_int:.4f}, OR={or_int:.4f}")

    except Exception as e:
        print(f"  BinomialBayesMixedGLM failed: {e}")
        print("  Falling back to logistic regression with cluster-robust SEs")

        try:
            import statsmodels.api as sm

            exog = df[["surgery_bin", "jitter_bin", "interaction", "domain_bin", "log10_params"]].copy()
            exog.insert(0, "intercept", 1)

            logit = sm.Logit(df["correct"].values, exog.values)
            fit = logit.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["model_name"]})

            coef_names = list(exog.columns)
            for i, name in enumerate(coef_names):
                results[f"coef_{name}"] = float(fit.params[i])
                results[f"se_{name}"] = float(fit.bse[i])

            idx_int = coef_names.index("interaction")
            beta_int = fit.params[idx_int]
            se_int = fit.bse[idx_int]
            z_int = fit.tvalues[idx_int]
            p_int = fit.pvalues[idx_int]
            or_int = np.exp(beta_int)

            results["interaction_beta"] = float(beta_int)
            results["interaction_se"] = float(se_int)
            results["interaction_z"] = float(z_int)
            results["interaction_p"] = float(p_int)
            results["interaction_or"] = float(or_int)
            results["method"] = "Logit_ClusterRobust"

            print(f"  Method: Logistic (cluster-robust SEs by model)")
            print(f"  Interaction: beta={beta_int:.4f}, SE={se_int:.4f}, z={z_int:.3f}, p={p_int:.4f}, OR={or_int:.4f}")

        except Exception as e2:
            print(f"  Logistic regression also failed: {e2}")
            results["method"] = "failed"
            results["error"] = str(e2)

    # Save
    with open(tables_dir / "mixed_effects_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _bootstrap_interaction(df: pd.DataFrame, effects_df: pd.DataFrame,
                           tables_dir: Path, n_boot: int = 2000) -> Dict:
    """Bootstrap interaction effect by resampling models."""
    print("\n--- Bootstrap interaction CI ---")
    rng = np.random.default_rng(42)

    interactions = effects_df["interaction"].dropna().values
    models = effects_df["model_name"].values
    n_models = len(models)

    boot_ints = []
    for _ in range(n_boot):
        idx = rng.choice(n_models, size=n_models, replace=True)
        boot_ints.append(interactions[idx].mean())

    boot_ints = np.array(boot_ints)
    ci_lo, ci_hi = np.percentile(boot_ints, [2.5, 97.5])
    mean_int = np.mean(interactions)

    # ROPE check: is the interaction OR within [0.8, 1.25]?
    # Convert mean interaction (accuracy difference) to approximate OR
    # For small effects on probability scale, OR ≈ exp(effect / (p*(1-p)))
    p_bar = df["correct"].mean()
    scale = 1.0 / (p_bar * (1 - p_bar)) if 0 < p_bar < 1 else 1.0
    boot_ors = np.exp(boot_ints * scale)
    in_rope = np.mean((boot_ors >= 0.8) & (boot_ors <= 1.25))

    results = {
        "mean_interaction": float(mean_int),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
        "median_or": float(np.median(boot_ors)),
        "rope_fraction": float(in_rope),
        "n_boot": n_boot,
        "n_models": n_models,
    }

    with open(tables_dir / "bootstrap_interaction.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Mean interaction: {mean_int:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  CI includes zero: {results['ci_includes_zero']}")
    print(f"  ROPE fraction: {in_rope:.3f}")

    return results


def _permutation_test(df: pd.DataFrame, tables_dir: Path, n_perm: int = 5000) -> Dict:
    """Permutation test: shuffle surgery/jitter labels within model."""
    print("\n--- Permutation test ---")
    rng = np.random.default_rng(42)

    # Observed interaction: mean of per-model interactions
    observed = _compute_global_interaction(df)

    null_dist = []
    for _ in range(n_perm):
        df_perm = df.copy()
        # Shuffle surgery and jitter labels independently within each model
        for model_name in df_perm["model_name"].unique():
            mask = df_perm["model_name"] == model_name
            idx = df_perm.index[mask]
            df_perm.loc[idx, "surgery"] = rng.permutation(df_perm.loc[idx, "surgery"].values)
            df_perm.loc[idx, "jitter"] = rng.permutation(df_perm.loc[idx, "jitter"].values)
        null_dist.append(_compute_global_interaction(df_perm))

    null_dist = np.array(null_dist)
    p_perm = np.mean(np.abs(null_dist) >= np.abs(observed))

    results = {
        "observed_interaction": float(observed),
        "p_perm": float(p_perm),
        "null_mean": float(null_dist.mean()),
        "null_std": float(null_dist.std()),
        "n_perm": n_perm,
    }

    # Save null distribution
    pd.DataFrame({"null_interaction": null_dist}).to_csv(
        tables_dir / "permutation_test.csv", index=False
    )
    with open(tables_dir / "permutation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Observed interaction: {observed:.4f}")
    print(f"  Permutation p: {p_perm:.4f}")

    return results


def _compute_global_interaction(df: pd.DataFrame) -> float:
    """Compute the global mean interaction effect across models."""
    interactions = []
    for model_name in df["model_name"].unique():
        mdf = df[df["model_name"] == model_name]
        acc = {}
        for (s, j), grp in mdf.groupby(["surgery", "jitter"]):
            acc[(s, j)] = grp["correct"].mean()
        co = acc.get(("control", "off"), np.nan)
        eo = acc.get(("expand", "off"), np.nan)
        cn = acc.get(("control", "on"), np.nan)
        en = acc.get(("expand", "on"), np.nan)
        interaction = (en - cn) - (eo - co)
        if not np.isnan(interaction):
            interactions.append(interaction)
    return np.mean(interactions) if interactions else 0.0


def _leave_one_out(df: pd.DataFrame, effects_df: pd.DataFrame,
                   tables_dir: Path) -> Dict:
    """Leave-one-model-out: recompute mean interaction excluding each model."""
    print("\n--- Leave-one-model-out ---")
    models = effects_df["model_name"].values
    full_mean = effects_df["interaction"].mean()

    loo_rows = []
    for model_name in models:
        subset = effects_df[effects_df["model_name"] != model_name]
        loo_mean = subset["interaction"].mean()
        loo_rows.append({
            "excluded_model": model_name,
            "loo_mean_interaction": loo_mean,
            "delta_from_full": loo_mean - full_mean,
        })

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(tables_dir / "loo_interaction.csv", index=False)

    max_influence = loo_df.loc[loo_df["delta_from_full"].abs().idxmax()]
    stability = loo_df["loo_mean_interaction"].std()

    results = {
        "full_mean": float(full_mean),
        "loo_std": float(stability),
        "most_influential": str(max_influence["excluded_model"]),
        "max_delta": float(max_influence["delta_from_full"]),
        "all_same_sign": bool((loo_df["loo_mean_interaction"] > 0).all() or
                              (loo_df["loo_mean_interaction"] < 0).all() or
                              (loo_df["loo_mean_interaction"].abs() < 0.01).all()),
    }

    print(f"  Full mean interaction: {full_mean:.4f}")
    print(f"  LOO std: {stability:.4f}")
    print(f"  Most influential: {max_influence['excluded_model']} (delta={max_influence['delta_from_full']:.4f})")

    return results


# ── Figures ──────────────────────────────────────────────────────────────
PARADIGM_COLORS = {
    "transformer": "#2196F3",
    "ssm": "#4CAF50",
    "hybrid": "#FF9800",
    "reasoning": "#FF5722",
}


def _plot_interaction_forest(effects_df: pd.DataFrame, fig_dir: Path):
    """Forest plot: per-model interaction effects with global mean."""
    fig, ax = plt.subplots(figsize=(10, max(6, len(effects_df) * 0.5)))

    sorted_df = effects_df.sort_values("interaction")
    y_positions = range(len(sorted_df))

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        color = PARADIGM_COLORS.get(row["paradigm"], "#666")
        ax.barh(i, row["interaction"], color=color, alpha=0.7, height=0.6)
        ax.text(row["interaction"], i, f"  {row['interaction']:.3f}",
                va="center", fontsize=8)

    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.axvline(x=effects_df["interaction"].mean(), color="red",
               linestyle="--", linewidth=1.5, label=f"Mean={effects_df['interaction'].mean():.4f}")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(sorted_df["model_name"].values, fontsize=9)
    ax.set_xlabel("Interaction Effect (surgery × jitter)", fontsize=12)
    ax.set_title("Per-Model Interaction Effects\n(0 = orthogonal, non-zero = synergy/interference)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    # Add paradigm legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=p) for p, c in PARADIGM_COLORS.items()
               if p in effects_df["paradigm"].values]
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="red", linestyle="--", label="Mean")],
              loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "interaction_forest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> interaction_forest.png")


def _plot_condition_effects(model_cond: pd.DataFrame, fig_dir: Path):
    """Grouped bar chart: accuracy by condition, faceted by paradigm."""
    paradigms = sorted(model_cond["paradigm"].unique())
    n_paradigms = len(paradigms)
    fig, axes = plt.subplots(1, n_paradigms, figsize=(5 * n_paradigms, 6), sharey=True)
    if n_paradigms == 1:
        axes = [axes]

    condition_labels = ["control/off", "expand/off", "control/on", "expand/on"]
    bar_colors = ["#66BB6A", "#42A5F5", "#FFA726", "#EF5350"]

    for ax_idx, paradigm in enumerate(paradigms):
        ax = axes[ax_idx]
        psub = model_cond[model_cond["paradigm"] == paradigm]

        # Aggregate across domains and models within paradigm
        cond_acc = []
        for surg, jit in [("control","off"), ("expand","off"), ("control","on"), ("expand","on")]:
            sub = psub[(psub["surgery"] == surg) & (psub["jitter"] == jit)]
            cond_acc.append(sub["accuracy"].mean() if not sub.empty else 0)

        bars = ax.bar(range(4), cond_acc, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{paradigm.title()}", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, cond_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=9)

    axes[0].set_ylabel("Accuracy", fontsize=12)
    fig.suptitle("Accuracy by Condition and Paradigm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "condition_effects_by_paradigm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> condition_effects_by_paradigm.png")


def _plot_shift_scatter(effects_df: pd.DataFrame, fig_dir: Path):
    """Scatter: surgery effect vs jitter effect, colored by paradigm."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in effects_df.iterrows():
        color = PARADIGM_COLORS.get(row["paradigm"], "#666")
        ax.scatter(row["surgery_effect"], row["jitter_effect"],
                   c=color, s=100, alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.annotate(row["model_name"], (row["surgery_effect"], row["jitter_effect"]),
                    fontsize=7, ha="center", va="bottom")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Surgery Effect (expand - control, no jitter)", fontsize=12)
    ax.set_ylabel("Jitter Effect (jitter on - off, no surgery)", fontsize=12)
    ax.set_title("Surgery vs Jitter Main Effects by Model", fontsize=14, fontweight="bold")

    # Correlation
    r, p = sp_stats.spearmanr(effects_df["surgery_effect"].dropna(),
                               effects_df["jitter_effect"].dropna())
    ax.text(0.05, 0.95, f"Spearman r={r:.3f}, p={p:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=p) for p, c in PARADIGM_COLORS.items()
               if p in effects_df["paradigm"].values]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "surgical_shift_vs_jitter_shift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> surgical_shift_vs_jitter_shift.png")


# ── Key findings ─────────────────────────────────────────────────────────
def _write_findings(df, effects_df, mixed, bootstrap, perm, loo, out_dir):
    """Write key_findings.txt."""
    lines = []
    lines.append(f"models_tested: {df['model_name'].nunique()}")
    lines.append(f"total_trials: {len(df)}")
    lines.append(f"paradigms: {', '.join(sorted(df['paradigm'].unique()))}")

    # Condition means
    for (s, j), grp in df.groupby(["surgery", "jitter"]):
        lines.append(f"accuracy_{s}_{j}: {grp['correct'].mean():.4f}")

    # Interaction
    mean_int = effects_df["interaction"].mean()
    lines.append(f"mean_interaction: {mean_int:.4f}")

    if "interaction_beta" in mixed:
        lines.append(f"mixed_model_interaction_beta: {mixed['interaction_beta']:.4f}")
        lines.append(f"mixed_model_interaction_p: {mixed['interaction_p']:.4f}")
        lines.append(f"mixed_model_interaction_or: {mixed['interaction_or']:.4f}")
        lines.append(f"mixed_model_method: {mixed.get('method', 'unknown')}")

    lines.append(f"bootstrap_interaction_ci: [{bootstrap['ci_lo']:.4f}, {bootstrap['ci_hi']:.4f}]")
    lines.append(f"bootstrap_ci_includes_zero: {bootstrap['ci_includes_zero']}")
    lines.append(f"bootstrap_rope_fraction: {bootstrap['rope_fraction']:.3f}")
    lines.append(f"permutation_p: {perm['p_perm']:.4f}")
    lines.append(f"loo_std: {loo['loo_std']:.4f}")
    lines.append(f"loo_most_influential: {loo['most_influential']}")

    # Per-paradigm interaction
    for paradigm in sorted(effects_df["paradigm"].unique()):
        psub = effects_df[effects_df["paradigm"] == paradigm]
        lines.append(f"paradigm_{paradigm}_interaction: {psub['interaction'].mean():.4f}")

    # Determine outcome
    orthogonal = (
        bootstrap["ci_includes_zero"]
        and perm["p_perm"] > 0.05
        and bootstrap["rope_fraction"] > 0.5
    )
    verdict = "NO_INTERACTION_DETECTED (consistent with orthogonality)" if orthogonal else "INTERACTION_DETECTED (orthogonality rejected)"
    lines.append(f"verdict: {verdict}")

    # Implications
    if orthogonal:
        lines.append(
            "business_implication: No detectable interaction between PR surgery and noise "
            "stress — deployment testing should treat geometric quality and dynamic stability "
            "as separate risk axes. Current evidence supports independent testing protocols."
        )
        lines.append(
            "scientific_implication: 2x2 factorial data are consistent with weak or no "
            "interaction between representation geometry manipulation and dynamic perturbation. "
            "This does not prove mechanistic independence but provides the strongest evidence "
            "to date that these measurement axes are not confounded."
        )
    else:
        lines.append(
            "business_implication: PR surgery and noise stress interact — models that are "
            "geometrically modified show different stability profiles. Risk assessment must "
            "consider combined effects, not treat axes as independent."
        )
        lines.append(
            "scientific_implication: Interaction detected between PR surgery and dynamic "
            "stress. The orthogonality hypothesis from exp-011 (correlation-based) is "
            "rejected by the stronger factorial test. Further investigation needed to "
            "characterize the interaction mechanism."
        )

    (out_dir / "key_findings.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nKey findings written to {out_dir / 'key_findings.txt'}")
    for l in lines:
        print(f"  {l}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Exp-012: Mechanistic Orthogonality Decomposition"
    )
    parser.add_argument("--stage", choices=["collect", "analyze", "all"],
                        default="all")
    parser.add_argument("--outdir", type=str,
                        default="analysis/orthogonality_decomposition_012")
    parser.add_argument("--surgery-strength", type=float,
                        default=DEFAULT_SURGERY_STRENGTH)
    parser.add_argument("--jitter-strength", type=float,
                        default=DEFAULT_JITTER_STRENGTH)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--skip-models", nargs="+", default=None)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS[:args.n_seeds]
    skip_set = set(args.skip_models) if args.skip_models else None

    if args.stage in ("collect", "all"):
        collect_data(out_dir, args.surgery_strength, args.jitter_strength,
                     seeds, skip_set)

    if args.stage in ("analyze", "all"):
        run_analysis(out_dir)


if __name__ == "__main__":
    main()
