#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecosystem_analysis import add_derived_metrics, flatten_model, load_payload, short_name
from ecosystem_deep_dynamics import add_adoption_residual, add_research_features
from measured_compatibility_tax import ARCH_SUPPORT, FRAMEWORK_WEIGHTS, MODEL_TO_ARCH, compute_measured_support

EPS = 1e-12
BASELINE_BIN = "lead_6_0"
EVENT_BIN_ORDER = [
    "lead_18_plus",
    "lead_18_12",
    "lead_12_6",
    "lead_6_0",
    "lag_0_6",
    "lag_6_12",
    "lag_12_18",
    "lag_18_plus",
]
BIN_TO_MIDPOINT = {
    "lead_18_plus": -21.0,
    "lead_18_12": -15.0,
    "lead_12_6": -9.0,
    "lead_6_0": -3.0,
    "lag_0_6": 3.0,
    "lag_6_12": 9.0,
    "lag_12_18": 15.0,
    "lag_18_plus": 21.0,
}

FRAMEWORKS = list(FRAMEWORK_WEIGHTS.keys())
ARCHITECTURE_KEYS = sorted(ARCH_SUPPORT.keys())

# Mapping: FRAMEWORK_SUPPORT_TIMELINE[framework][architecture] = first date support was added.
# Sources: GitHub PRs, release notes, changelogs for each framework.
# Dates are YYYY-MM-DD (1st of month if only month known).
# Entries left as None or omitted → NaT (no event).  Only pairs with ARCH_SUPPORT > 0 matter.
FRAMEWORK_SUPPORT_TIMELINE: Dict[str, Dict[str, str]] = {
    "vllm": {
        # Pure transformers — supported shortly after vLLM launch (Jun 2023) or arch release
        "llama":        "2023-06-20",  # vLLM v0.1.0, built for llama
        "qwen2":        "2024-06-07",  # Qwen2 release, vLLM same-week
        "qwen3":        "2025-04-29",  # Qwen3 release Apr 2025
        "gemma":        "2024-02-21",  # Gemma 1 release
        "gemma2":       "2024-06-27",  # Gemma 2 release
        "gemma3":       "2025-03-12",  # Gemma 3 release
        "phi3":         "2024-04-23",  # Phi-3 release
        "phi4":         "2024-12-12",  # Phi-4 release
        "granite":      "2024-10-21",  # Granite 3.0 release
        "olmo2":        "2024-11-01",  # OLMo-2 release
        "smollm3":      "2025-06-01",  # SmolLM3 release ~mid-2025
        "deepseek_r1":  "2025-01-20",  # DeepSeek-R1 release (qwen/llama arch)
        "falcon3":      "2024-12-12",  # Falcon3 release Dec 2024
        "mistral":      "2023-09-27",  # Mistral 7B release
        "cohere":       "2024-04-04",  # Command-R release
        "internlm":     "2023-07-06",  # InternLM release
        "baichuan":     "2023-06-15",  # Baichuan release
        "tinyllama":    "2023-09-01",  # TinyLlama (llama arch)
        "nemotron":     "2024-10-01",  # Nemotron (llama arch)
        # MoE
        "mixtral":      "2024-01-08",  # Mixtral release, vLLM MoE support early Jan
        "deepseek_moe": "2024-06-01",  # DeepSeek-V2 May 2024
        "qwen_moe":     "2024-04-01",  # Qwen1.5-MoE Mar 2024
        "olmoe":        "2024-10-01",  # OLMoE Sep 2024
        "arctic":       "2024-05-01",  # Arctic Apr 2024 (0.5 support)
        "llama4_moe":   "2025-04-05",  # Llama 4 Apr 2025
        # Pure SSM
        "mamba":        "2024-07-01",  # vLLM ~Jun-Jul 2024 (via Jamba work)
        "mamba2":       "2024-10-01",  # vLLM late 2024 (0.5 support)
        "falcon_mamba": "2024-10-01",  # vLLM PR #9325
        # Hybrid
        "falcon_h1":    "2025-05-20",  # Falcon-H1 announcement with vLLM
        "granite_hybrid":"2025-10-01", # Granite 4.0-H release (0.5 support)
        "nemotron_h":   "2025-03-18",  # Nemotron-H release Mar 2025
        "zamba2":       "2025-01-15",  # vLLM PR #13185
        "jamba":        "2024-07-02",  # vLLM PR #4115
        "hymba":        "2025-01-01",  # Partial support (0.5)
        # Liquid
        "lfm2":         "2025-02-01",  # LFM2 release
        # Reasoning (qwen2/llama arch)
        "qwq":          "2024-12-01",  # QwQ release (qwen2 arch)
        "marco_o1":     "2024-12-01",  # Marco-o1 (qwen2/llama arch)
    },
    "llamacpp": {
        # Pure transformers — llama.cpp launch Mar 2023, added archs progressively
        "llama":        "2023-03-10",  # llama.cpp created for llama
        "qwen2":        "2024-06-07",  # Qwen2 GGUF support
        "qwen3":        "2025-04-29",  # Qwen3 GGUF
        "gemma":        "2024-02-21",  # Gemma GGUF
        "gemma2":       "2024-06-27",  # Gemma 2 GGUF
        "gemma3":       "2025-03-12",  # Gemma 3 GGUF
        "phi3":         "2024-04-23",  # Phi-3 GGUF
        "phi4":         "2024-12-12",  # Phi-4 GGUF
        "granite":      "2024-10-21",  # Granite GGUF
        "olmo2":        "2024-11-01",  # OLMo-2 GGUF
        "smollm3":      "2025-06-01",  # SmolLM3 GGUF
        "deepseek_r1":  "2025-01-20",  # DeepSeek-R1 GGUF
        "falcon3":      "2024-12-12",  # Falcon3 GGUF
        "mistral":      "2023-10-10",  # Mistral 7B GGUF
        "cohere":       "2024-04-04",  # Command-R GGUF
        "internlm":     "2023-09-01",  # InternLM GGUF
        "baichuan":     "2023-08-01",  # Baichuan GGUF
        "tinyllama":    "2023-09-01",  # TinyLlama GGUF (llama arch)
        "nemotron":     "2024-10-01",  # Nemotron GGUF (llama arch)
        # MoE
        "mixtral":      "2024-01-08",  # Mixtral GGUF
        "deepseek_moe": "2024-07-01",  # DeepSeek-V2 GGUF
        "qwen_moe":     "2024-04-01",  # Qwen MoE GGUF
        "olmoe":        "2024-10-01",  # OLMoE GGUF
        "arctic":       "2024-06-01",  # Arctic GGUF (0.5)
        "llama4_moe":   "2025-04-05",  # Llama 4 GGUF
        # Pure SSM
        "mamba":        "2024-03-08",  # llama.cpp PR #5328
        "mamba2":       "2024-09-01",  # llama.cpp PR #9126 (0.5)
        "falcon_mamba": "2024-10-01",  # After issue #9009 Aug 2024
        # Hybrid
        "falcon_h1":    "2025-06-01",  # Falcon-H1 GGUF
        "granite_hybrid":"2025-10-01", # Granite 4.0-H GGUF
        "nemotron_h":   "2025-03-18",  # Nemotron-H GGUF
        "zamba2":       "2025-02-01",  # Zamba2 GGUF
        "jamba":        "2024-06-01",  # llama.cpp PR #7531
        "hymba":        "2025-01-01",  # Partial GGUF (0.5)
        # Liquid
        "lfm2":         "2025-02-01",  # LFM2 GGUF
        # RWKV
        "rwkv7":        "2025-02-01",  # RWKV7 GGUF
        "rwkv6":        "2024-09-01",  # llama.cpp PR #8980
        "rwkv5":        "2024-09-01",  # Same PR as RWKV6
        "arwkv7":       "2025-03-01",  # ARWKV GGUF
        # Diffusion
        "llada":        "2025-03-01",  # LLaDA GGUF
        "llada2":       "2025-07-01",  # LLaDA2 GGUF
        # Reasoning
        "qwq":          "2024-12-01",  # QwQ GGUF (qwen2 arch)
        "marco_o1":     "2024-12-01",  # Marco-o1 GGUF
    },
    "tgi": {
        # TGI supports models via HF Transformers integration
        # Pure transformers
        "llama":        "2023-03-01",  # TGI early support
        "qwen2":        "2024-06-07",
        "gemma":        "2024-02-21",
        "gemma2":       "2024-06-27",
        "gemma3":       "2025-03-12",
        "phi3":         "2024-04-23",
        "phi4":         "2024-12-12",
        "granite":      "2024-10-21",
        "deepseek_r1":  "2025-01-20",
        "falcon3":      "2024-12-12",
        "mistral":      "2023-10-10",
        "cohere":       "2024-04-04",
        "internlm":     "2023-09-01",  # 0.5 support
        "tinyllama":    "2023-09-01",
        "nemotron":     "2024-10-01",
        # MoE
        "mixtral":      "2024-01-08",
        "deepseek_moe": "2024-06-01",  # 0.5 support
        "llama4_moe":   "2025-04-05",  # 0.5 support
        # SSM — TGI added Mamba via HF Transformers
        "mamba":        "2024-03-05",  # HF Transformers Mamba support
        # Reasoning
        "qwq":          "2024-12-01",
        "marco_o1":     "2024-12-01",
    },
    "gptq": {
        # AutoGPTQ/GPTQModel — primarily transformer architectures
        "llama":        "2023-04-01",  # AutoGPTQ early days
        "qwen2":        "2024-06-07",
        "qwen3":        "2025-04-29",
        "gemma":        "2024-03-01",
        "gemma2":       "2024-06-27",
        "gemma3":       "2025-03-12",
        "phi3":         "2024-04-23",
        "phi4":         "2024-12-12",
        "granite":      "2024-10-21",
        "olmo2":        "2024-11-01",
        "smollm3":      "2025-06-01",
        "deepseek_r1":  "2025-01-20",
        "falcon3":      "2024-12-12",
        "mistral":      "2023-10-10",
        "cohere":       "2024-04-04",
        "internlm":     "2023-09-01",
        "baichuan":     "2023-08-01",
        "tinyllama":    "2023-09-01",
        "nemotron":     "2024-10-01",
        # MoE
        "mixtral":      "2024-02-01",
        "deepseek_moe": "2024-07-01",  # 0.5 support
        "qwen_moe":     "2024-05-01",  # 0.5 support
        "olmoe":        "2024-10-01",  # 0.5 support
        "llama4_moe":   "2025-04-05",  # 0.5 support
        # Hybrid (limited)
        "falcon_h1":    "2025-06-01",
        "nemotron_h":   "2025-03-18",
        # Reasoning
        "qwq":          "2024-12-01",
        "marco_o1":     "2024-12-01",
    },
    "awq": {
        # AutoAWQ — transformer-focused
        "llama":        "2023-06-01",  # AWQ launch
        "qwen2":        "2024-06-07",
        "qwen3":        "2025-04-29",
        "gemma":        "2024-03-01",
        "gemma2":       "2024-06-27",
        "phi3":         "2024-04-23",
        "phi4":         "2024-12-12",
        "deepseek_r1":  "2025-01-20",
        "falcon3":      "2024-12-12",
        "mistral":      "2023-10-10",
        "cohere":       "2024-05-01",  # 0.5 support
        "internlm":     "2023-09-01",
        "baichuan":     "2023-08-01",
        "tinyllama":    "2023-09-01",
        "nemotron":     "2024-10-01",
        # MoE
        "mixtral":      "2024-02-01",
        # Reasoning
        "qwq":          "2024-12-01",
        "marco_o1":     "2024-12-01",
    },
    "bnb": {
        # BitsAndBytes — works via HF Transformers, broad support
        # Pure transformers
        "llama":        "2023-03-01",  # Early HF/bnb support
        "qwen2":        "2024-06-07",
        "qwen3":        "2025-04-29",
        "gemma":        "2024-02-21",
        "gemma2":       "2024-06-27",
        "gemma3":       "2025-03-12",
        "phi3":         "2024-04-23",
        "phi4":         "2024-12-12",
        "granite":      "2024-10-21",
        "olmo2":        "2024-11-01",
        "smollm3":      "2025-06-01",
        "deepseek_r1":  "2025-01-20",
        "falcon3":      "2024-12-12",
        "mistral":      "2023-10-10",
        "cohere":       "2024-04-04",
        "internlm":     "2023-09-01",
        "baichuan":     "2023-08-01",
        "tinyllama":    "2023-09-01",
        "nemotron":     "2024-10-01",
        # MoE
        "mixtral":      "2024-01-08",
        "deepseek_moe": "2024-06-01",  # 0.5
        "qwen_moe":     "2024-04-01",  # 0.5
        "olmoe":        "2024-10-01",  # 0.5
        "arctic":       "2024-05-01",  # 0.5
        "llama4_moe":   "2025-04-05",  # 0.5
        # Pure SSM
        "mamba":        "2024-03-05",  # HF Transformers Mamba date
        "mamba2":       "2024-08-06",  # HF Transformers Mamba2 date
        "falcon_mamba": "2024-08-12",  # HF Transformers FalconMamba date
        "rene":         "2024-06-01",  # Basic HF loading (0.5)
        # Hybrid
        "falcon_h1":    "2025-05-20",
        "granite_hybrid":"2025-10-01",
        "nemotron_h":   "2025-03-18",
        "zamba2":       "2025-01-27",  # HF Transformers v4.49
        "jamba":        "2024-04-18",  # HF Transformers Jamba
        "hymba":        "2024-11-01",  # Via trust_remote_code (0.5)
        "stripedhyena": "2024-01-01",  # Via custom code (0.5)
        # Liquid
        "lfm2":         "2025-02-01",
        # RWKV
        "rwkv7":        "2025-02-01",
        "rwkv6":        "2024-11-24",  # HF RWKV6 Nov 2024
        "rwkv5":        "2023-05-09",  # HF original RWKV support
        "arwkv7":       "2025-03-01",
        # xLSTM
        "xlstm":        "2025-07-25",  # HF xLSTM support
        # RetNet
        "retnet":       "2023-12-01",  # Via custom code (0.5)
        # Diffusion
        "llada":        "2025-03-01",
        "llada2":       "2025-07-01",
        # Reasoning
        "qwq":          "2024-12-01",
        "marco_o1":     "2024-12-01",
    },
}


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text)).strip("_").lower()
    return token or "x"


def parse_timeline_date(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    s = str(value).strip()
    if not s:
        return pd.NaT
    upper = s.upper()
    if "YYYY" in upper or "TODO" in upper or upper in {"TBD", "NA", "N/A", "?"}:
        return pd.NaT
    return pd.to_datetime(s, errors="coerce")


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def load_dataframe(input_path: Path) -> Tuple[dict, pd.DataFrame]:
    meta, models = load_payload(input_path)
    df = pd.DataFrame([flatten_model(m) for m in models])

    extras = []
    for m in models:
        ps = m.get("paradigm_specific") or {}
        extras.append(
            {
                "model_id": m.get("model_id"),
                "num_ssm_layers": ps.get("num_ssm_layers"),
                "num_attention_layers": ps.get("num_attention_layers"),
                "ssm_layer_ratio": ps.get("ssm_layer_ratio"),
                "attention_layer_ratio": ps.get("attention_layer_ratio"),
            }
        )
    extra_df = pd.DataFrame(extras)
    df = df.merge(extra_df, on="model_id", how="left")

    df = add_derived_metrics(df)
    df = add_research_features(df)
    df["is_long_context_128k_f"] = df["is_long_context_128k"].fillna(False).astype(float)
    df["has_quant_f"] = df["has_quant"].fillna(False).astype(float)
    df["has_ssm_signals_f"] = df["has_ssm_signals"].fillna(False).astype(float)
    df, _ = add_adoption_residual(df)

    return meta, df


def attach_support_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    support_records = [compute_measured_support(mid) for mid in out["model_id"]]
    support_df = pd.DataFrame(support_records, index=out.index)
    for col in support_df.columns:
        out[col] = support_df[col]

    mapped_arch = out["model_id"].map(MODEL_TO_ARCH)
    if "arch_key" in out.columns:
        out["arch_key"] = mapped_arch.fillna(out["arch_key"])
    else:
        out["arch_key"] = mapped_arch
    out["arch_key"] = out["arch_key"].fillna("unknown")

    out["release_month_str"] = out["release_month"].dt.to_period("M").astype(str)
    out.loc[out["release_month"].isna(), "release_month_str"] = "missing"

    paradigm_n = out["inferred_paradigm"].fillna("unknown").value_counts()
    arch_n = out["arch_key"].fillna("unknown").value_counts()

    out["paradigm_n"] = out["inferred_paradigm"].fillna("unknown").map(paradigm_n).astype(float)
    out["arch_n"] = out["arch_key"].fillna("unknown").map(arch_n).astype(float)
    out["imbalance_weight"] = 1.0 / np.sqrt(out["paradigm_n"] * out["arch_n"])
    out["imbalance_weight"] = out["imbalance_weight"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["imbalance_weight"] = out["imbalance_weight"] / out["imbalance_weight"].mean()

    return out


def build_timeline_events(df: pd.DataFrame, timeline: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sample_min = df["release_date"].min()
    sample_max = df["release_date"].max()

    for framework, arch_dates in timeline.items():
        for arch_key, raw_date in arch_dates.items():
            date_added = parse_timeline_date(raw_date)
            level = ARCH_SUPPORT.get(str(arch_key), {}).get(str(framework), np.nan)
            level = float(level) if pd.notna(level) else np.nan

            has_date = bool(pd.notna(date_added))
            valid_event = bool(has_date and pd.notna(level) and level > 0)
            mid_sample_shock = bool(
                valid_event
                and pd.notna(sample_min)
                and pd.notna(sample_max)
                and (date_added > sample_min)
                and (date_added < sample_max)
            )

            rows.append(
                {
                    "framework": framework,
                    "arch_key": arch_key,
                    "date_raw": raw_date,
                    "date_added": date_added,
                    "current_support_level": level,
                    "has_date": has_date,
                    "valid_event": valid_event,
                    "mid_sample_shock": mid_sample_shock,
                }
            )

    out_cols = [
        "framework",
        "arch_key",
        "event_id",
        "date_raw",
        "date_added",
        "current_support_level",
        "has_date",
        "valid_event",
        "mid_sample_shock",
    ]
    if not rows:
        return pd.DataFrame(columns=out_cols)

    events = pd.DataFrame(rows)
    events["event_id"] = events["framework"].astype(str) + "::" + events["arch_key"].astype(str)
    events = events[out_cols]
    events = events.sort_values(["date_added", "framework", "arch_key"], na_position="last").reset_index(drop=True)
    return events


def add_release_timing_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if events.empty:
        out["support_score_at_release"] = np.nan
        out["frameworks_supported_at_release"] = np.nan
        out["first_shock_date"] = pd.NaT
        out["post_first_shock_release"] = np.nan
        out["months_from_first_shock"] = np.nan
        out["support_gain_after_release"] = np.nan
        return out

    valid_events = events[events["valid_event"] & events["date_added"].notna()].copy()
    date_lookup = {(r.framework, r.arch_key): r.date_added for r in valid_events.itertuples()}
    first_shock_lookup = valid_events.groupby("arch_key")["date_added"].min().to_dict()

    rows: List[Dict[str, Any]] = []
    for row in out.itertuples():
        arch_key = getattr(row, "arch_key", "unknown")
        release_date = getattr(row, "release_date", pd.NaT)

        cur_support = ARCH_SUPPORT.get(str(arch_key), {})
        known_weight = 0.0
        release_weighted_support = 0.0
        frameworks_supported_at_release = 0

        for fw, w in FRAMEWORK_WEIGHTS.items():
            level = cur_support.get(fw, np.nan)
            if pd.isna(level):
                continue
            event_date = date_lookup.get((fw, arch_key), pd.NaT)
            if pd.isna(event_date):
                continue

            known_weight += float(w)
            if pd.notna(release_date) and (release_date >= event_date):
                release_weighted_support += float(w) * float(level)
                if float(level) > 0:
                    frameworks_supported_at_release += 1

        support_score_at_release = (
            release_weighted_support / known_weight if known_weight > 0 else np.nan
        )

        first_shock_date = first_shock_lookup.get(arch_key, pd.NaT)
        post_first_shock_release = float(
            pd.notna(release_date) and pd.notna(first_shock_date) and (release_date >= first_shock_date)
        )
        months_from_first_shock = (
            (release_date - first_shock_date).days / 30.44
            if pd.notna(release_date) and pd.notna(first_shock_date)
            else np.nan
        )

        rows.append(
            {
                "support_score_at_release": support_score_at_release,
                "frameworks_supported_at_release": frameworks_supported_at_release,
                "first_shock_date": first_shock_date,
                "post_first_shock_release": post_first_shock_release,
                "months_from_first_shock": months_from_first_shock,
            }
        )

    timing_df = pd.DataFrame(rows, index=out.index)
    for col in timing_df.columns:
        out[col] = timing_df[col]

    out["support_gain_after_release"] = out["support_score"] - out["support_score_at_release"]
    return out


def dominant_paradigm_by_arch(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for arch_key, sub in df.groupby("arch_key"):
        mode = sub["inferred_paradigm"].dropna().mode()
        out[str(arch_key)] = str(mode.iloc[0]) if not mode.empty else "unknown"
    return out


def assign_event_bin(rel_months: float) -> str:
    if pd.isna(rel_months):
        return "missing"
    if rel_months < -18:
        return "lead_18_plus"
    if rel_months < -12:
        return "lead_18_12"
    if rel_months < -6:
        return "lead_12_6"
    if rel_months < 0:
        return "lead_6_0"
    if rel_months < 6:
        return "lag_0_6"
    if rel_months < 12:
        return "lag_6_12"
    if rel_months < 18:
        return "lag_12_18"
    return "lag_18_plus"


def build_event_panel(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window_months: int = 18,
    min_cell: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    panel_cols = [
        "event_id",
        "event_framework",
        "event_arch_key",
        "event_paradigm",
        "event_date",
        "model_id",
        "org",
        "inferred_paradigm",
        "arch_key",
        "release_date",
        "release_month_str",
        "downloads",
        "log10_downloads",
        "adoption_residual",
        "log10_params",
        "param_b",
        "recency_months",
        "support_score",
        "support_score_at_release",
        "measured_tax",
        "imbalance_weight",
        "treated",
        "post",
        "treat_post",
        "rel_months",
        "event_time_bin",
        "treated_pre_n",
        "treated_post_n",
        "control_pre_n",
        "control_post_n",
    ]

    valid_events = events[events["valid_event"] & events["date_added"].notna()].copy()
    if valid_events.empty:
        empty = pd.DataFrame(columns=panel_cols + ["event_n", "analysis_weight"])
        return empty, valid_events

    rows: List[pd.DataFrame] = []
    window_days = int(window_months * 30.44)
    arch_paradigm = dominant_paradigm_by_arch(df)

    for ev in valid_events.itertuples():
        event_date = ev.date_added
        event_arch = ev.arch_key
        event_paradigm = arch_paradigm.get(str(event_arch), "unknown")

        treated_mask = df["arch_key"].eq(event_arch)
        in_window = df["release_date"].notna() & ((df["release_date"] - event_date).dt.days.abs() <= window_days)

        control_mask = (~treated_mask) & df["inferred_paradigm"].eq(event_paradigm)
        sub = df[in_window & (treated_mask | control_mask)].copy()

        control_n = int((sub["arch_key"] != event_arch).sum())
        if control_n < 2 * min_cell:
            control_mask = ~treated_mask
            sub = df[in_window & (treated_mask | control_mask)].copy()

        if sub.empty:
            continue

        sub["treated"] = (sub["arch_key"] == event_arch).astype(float)
        sub["post"] = (sub["release_date"] >= event_date).astype(float)
        sub["treat_post"] = sub["treated"] * sub["post"]
        sub["rel_months"] = (sub["release_date"] - event_date).dt.days / 30.44
        sub["event_time_bin"] = sub["rel_months"].apply(assign_event_bin)

        t_pre = int(((sub["treated"] == 1) & (sub["post"] == 0)).sum())
        t_post = int(((sub["treated"] == 1) & (sub["post"] == 1)).sum())
        c_pre = int(((sub["treated"] == 0) & (sub["post"] == 0)).sum())
        c_post = int(((sub["treated"] == 0) & (sub["post"] == 1)).sum())

        if min(t_pre, t_post, c_pre, c_post) < min_cell:
            continue

        sub["event_id"] = ev.event_id
        sub["event_framework"] = ev.framework
        sub["event_arch_key"] = event_arch
        sub["event_paradigm"] = event_paradigm
        sub["event_date"] = event_date
        sub["treated_pre_n"] = t_pre
        sub["treated_post_n"] = t_post
        sub["control_pre_n"] = c_pre
        sub["control_post_n"] = c_post

        rows.append(sub[panel_cols])

    if not rows:
        empty = pd.DataFrame(columns=panel_cols + ["event_n", "analysis_weight"])
        return empty, valid_events

    panel = pd.concat(rows, ignore_index=True)
    panel["event_n"] = panel.groupby("event_id")["model_id"].transform("count").astype(float)
    panel["analysis_weight"] = panel["imbalance_weight"].fillna(1.0) / panel["event_n"].clip(lower=1.0)
    panel["analysis_weight"] = panel["analysis_weight"] / panel["analysis_weight"].mean()

    return panel, valid_events


def build_design_matrix(
    work: pd.DataFrame,
    x_cols: Sequence[str],
    fe_cols: Sequence[str],
) -> pd.DataFrame:
    blocks: List[pd.DataFrame] = [pd.DataFrame({"intercept": np.ones(len(work), dtype=float)}, index=work.index)]

    for col in x_cols:
        if col not in work.columns:
            continue
        block = pd.to_numeric(work[col], errors="coerce").astype(float).rename(col).to_frame()
        blocks.append(block)

    for fe in fe_cols:
        if fe not in work.columns:
            continue
        dummies = pd.get_dummies(
            work[fe].fillna("missing").astype(str),
            prefix=f"fe_{sanitize_token(fe)}",
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            blocks.append(dummies)

    x = pd.concat(blocks, axis=1)
    x = x.loc[:, ~x.columns.duplicated()]

    keep_cols: List[str] = []
    for col in x.columns:
        if col == "intercept":
            keep_cols.append(col)
            continue
        vals = pd.to_numeric(x[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        if float(vals.var(ddof=0)) <= EPS:
            continue
        keep_cols.append(col)

    return x[keep_cols]


def ols_from_matrix(y: np.ndarray, x: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n, k = x.shape
    if weights is None:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ beta
        resid = y - y_hat
        dof = max(n - k, 1)
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        r2 = 1.0 - (ss_res / (ss_tot + EPS))
    else:
        w = np.asarray(weights, dtype=float)
        sw = np.sqrt(w)
        xw = x * sw[:, None]
        yw = y * sw

        beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        y_hat = x @ beta
        resid = y - y_hat
        dof = max(n - k, 1)
        sigma2 = float((w * (resid ** 2)).sum() / dof)
        cov = sigma2 * np.linalg.pinv(xw.T @ xw)
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

        y_bar = float(np.average(y, weights=w))
        ss_tot = float((w * ((y - y_bar) ** 2)).sum())
        ss_res = float((w * (resid ** 2)).sum())
        r2 = 1.0 - (ss_res / (ss_tot + EPS))

    t_vals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return beta, se, t_vals, r2


def fit_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    fe_cols: Sequence[str] = (),
    weight_col: Optional[str] = None,
    min_n: int = 12,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    numeric_cols = [y_col] + [c for c in x_cols if c in df.columns]
    if weight_col is not None and weight_col in df.columns:
        numeric_cols.append(weight_col)

    keep_cols = numeric_cols + [c for c in fe_cols if c in df.columns]
    work = df[keep_cols].replace([np.inf, -np.inf], np.nan).copy()
    work = work.dropna(subset=numeric_cols)

    if len(work) < max(min_n, len(x_cols) + 3):
        return pd.DataFrame(), {"n": float(len(work)), "r2": np.nan, "k": 0.0}, pd.Series(dtype=float)

    x_df = build_design_matrix(work, x_cols=x_cols, fe_cols=fe_cols)
    if x_df.empty:
        return pd.DataFrame(), {"n": float(len(work)), "r2": np.nan, "k": 0.0}, pd.Series(dtype=float)
    if len(work) <= x_df.shape[1]:
        return pd.DataFrame(), {"n": float(len(work)), "r2": np.nan, "k": float(x_df.shape[1])}, pd.Series(dtype=float)

    y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=float)
    x_mat = x_df.to_numpy(dtype=float)

    if weight_col is not None and weight_col in work.columns:
        w = pd.to_numeric(work[weight_col], errors="coerce").fillna(1.0).clip(lower=EPS).to_numpy(dtype=float)
    else:
        w = None

    beta, se, t_vals, r2 = ols_from_matrix(y, x_mat, weights=w)

    coef_df = pd.DataFrame(
        {
            "term": x_df.columns,
            "coef": beta,
            "std_err": se,
            "t_value": t_vals,
            "n": len(work),
            "r2": r2,
        }
    )
    fitted = pd.Series(x_mat @ beta, index=work.index, name="fitted")
    info = {"n": float(len(work)), "r2": float(r2), "k": float(x_df.shape[1])}
    return coef_df, info, fitted


def term_value(coef_df: pd.DataFrame, term: str, col: str) -> float:
    if coef_df.empty or term not in set(coef_df["term"]):
        return np.nan
    return float(coef_df.loc[coef_df["term"] == term, col].iloc[0])


def did_from_means(sub: pd.DataFrame, y_col: str = "adoption_residual") -> float:
    def mean_of(mask: pd.Series) -> float:
        vals = pd.to_numeric(sub.loc[mask, y_col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else np.nan

    t_post = mean_of((sub["treated"] == 1) & (sub["post"] == 1))
    t_pre = mean_of((sub["treated"] == 1) & (sub["post"] == 0))
    c_post = mean_of((sub["treated"] == 0) & (sub["post"] == 1))
    c_pre = mean_of((sub["treated"] == 0) & (sub["post"] == 0))

    if any(pd.isna(v) for v in [t_post, t_pre, c_post, c_pre]):
        return np.nan
    return (t_post - t_pre) - (c_post - c_pre)


def run_pooled_did_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "spec",
        "n",
        "r2",
        "did_coef",
        "did_se",
        "did_t",
        "did_ci_low",
        "did_ci_high",
        "treated_coef",
        "post_coef",
        "controls",
        "fixed_effects",
        "weighting",
    ]
    if panel.empty:
        return pd.DataFrame(columns=cols)

    specs = [
        ("s1_did_only", ["treated", "post", "treat_post"], []),
        ("s2_plus_size_recency", ["treated", "post", "treat_post", "log10_params", "recency_months"], []),
        ("s3_plus_org_fe", ["treated", "post", "treat_post", "log10_params", "recency_months"], ["org"]),
        ("s4_plus_release_month_fe", ["treated", "post", "treat_post", "log10_params", "recency_months"], ["org", "release_month_str"]),
        ("s5_plus_event_fe", ["treated", "post", "treat_post", "log10_params", "recency_months"], ["org", "release_month_str", "event_id"]),
    ]

    rows: List[Dict[str, Any]] = []
    for spec_name, x_cols, fe_cols in specs:
        coef_df, info, _ = fit_ols(
            panel,
            y_col="adoption_residual",
            x_cols=x_cols,
            fe_cols=fe_cols,
            weight_col="analysis_weight",
            min_n=16,
        )
        did_coef = term_value(coef_df, "treat_post", "coef")
        did_se = term_value(coef_df, "treat_post", "std_err")
        did_t = term_value(coef_df, "treat_post", "t_value")

        rows.append(
            {
                "spec": spec_name,
                "n": int(info["n"]),
                "r2": info["r2"],
                "did_coef": did_coef,
                "did_se": did_se,
                "did_t": did_t,
                "did_ci_low": did_coef - 1.96 * did_se if pd.notna(did_coef) and pd.notna(did_se) else np.nan,
                "did_ci_high": did_coef + 1.96 * did_se if pd.notna(did_coef) and pd.notna(did_se) else np.nan,
                "treated_coef": term_value(coef_df, "treated", "coef"),
                "post_coef": term_value(coef_df, "post", "coef"),
                "controls": "|".join([c for c in x_cols if c not in {"treated", "post", "treat_post"}]),
                "fixed_effects": "|".join(fe_cols) if fe_cols else "none",
                "weighting": "analysis_weight",
            }
        )

    return pd.DataFrame(rows, columns=cols)


def run_event_level_did(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_id",
        "event_framework",
        "event_arch_key",
        "event_paradigm",
        "event_date",
        "event_n",
        "treated_pre_n",
        "treated_post_n",
        "control_pre_n",
        "control_post_n",
        "did_raw",
        "did_coef",
        "did_se",
        "did_t",
        "pretrend_slope_diff",
        "pretrend_t",
        "pretrend_pass",
        "did_coef_weighted",
        "did_raw_weighted",
    ]
    if panel.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []

    for event_id, sub in panel.groupby("event_id"):
        sub = sub.copy()
        did_raw = did_from_means(sub, y_col="adoption_residual")

        coef_df, _, _ = fit_ols(
            sub,
            y_col="adoption_residual",
            x_cols=["treated", "post", "treat_post", "log10_params", "recency_months"],
            fe_cols=["org", "release_month_str"],
            weight_col="analysis_weight",
            min_n=8,
        )
        did_coef = term_value(coef_df, "treat_post", "coef")
        did_se = term_value(coef_df, "treat_post", "std_err")
        did_t = term_value(coef_df, "treat_post", "t_value")

        pre_sub = sub[sub["post"] == 0].copy()
        pretrend_slope_diff = np.nan
        pretrend_t = np.nan
        if len(pre_sub) >= 8:
            pre_sub["treated_x_rel"] = pre_sub["treated"] * pre_sub["rel_months"]
            pre_coef, _, _ = fit_ols(
                pre_sub,
                y_col="adoption_residual",
                x_cols=["treated", "rel_months", "treated_x_rel", "log10_params", "recency_months"],
                fe_cols=["org"],
                weight_col="analysis_weight",
                min_n=8,
            )
            pretrend_slope_diff = term_value(pre_coef, "treated_x_rel", "coef")
            pretrend_t = term_value(pre_coef, "treated_x_rel", "t_value")

        row0 = sub.iloc[0]
        rows.append(
            {
                "event_id": event_id,
                "event_framework": row0["event_framework"],
                "event_arch_key": row0["event_arch_key"],
                "event_paradigm": row0["event_paradigm"],
                "event_date": row0["event_date"],
                "event_n": int(len(sub)),
                "treated_pre_n": int(row0["treated_pre_n"]),
                "treated_post_n": int(row0["treated_post_n"]),
                "control_pre_n": int(row0["control_pre_n"]),
                "control_post_n": int(row0["control_post_n"]),
                "did_raw": did_raw,
                "did_coef": did_coef,
                "did_se": did_se,
                "did_t": did_t,
                "pretrend_slope_diff": pretrend_slope_diff,
                "pretrend_t": pretrend_t,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)

    out["pretrend_pass"] = out["pretrend_t"].abs().le(1.96) | out["pretrend_t"].isna()
    weights = out["event_n"].clip(lower=1)
    weights = weights / weights.sum()
    out["did_coef_weighted"] = out["did_coef"] * weights
    out["did_raw_weighted"] = out["did_raw"] * weights

    out = out.sort_values("event_date").reset_index(drop=True)
    return out[cols]


def run_event_study(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coef_cols = [
        "bin",
        "midpoint_months",
        "coef",
        "std_err",
        "t_value",
        "ci_low",
        "ci_high",
        "is_pre",
        "is_reference",
    ]
    diag_cols = [
        "n",
        "lead_bins",
        "lead_mean_abs_coef",
        "lead_max_abs_t",
        "lead_share_significant",
    ]
    if panel.empty:
        return pd.DataFrame(columns=coef_cols), pd.DataFrame(columns=diag_cols)

    work = panel[panel["event_time_bin"] != "missing"].copy()
    if work.empty:
        return pd.DataFrame(columns=coef_cols), pd.DataFrame(columns=diag_cols)

    for b in EVENT_BIN_ORDER:
        if b == BASELINE_BIN:
            continue
        work[f"bin_{b}"] = (work["event_time_bin"] == b).astype(float)
        work[f"did_{b}"] = work["treated"] * work[f"bin_{b}"]

    x_cols = ["treated"]
    x_cols += [f"bin_{b}" for b in EVENT_BIN_ORDER if b != BASELINE_BIN]
    x_cols += [f"did_{b}" for b in EVENT_BIN_ORDER if b != BASELINE_BIN]
    x_cols += ["log10_params", "recency_months"]

    coef_df, info, _ = fit_ols(
        work,
        y_col="adoption_residual",
        x_cols=x_cols,
        fe_cols=["event_id", "org", "release_month_str"],
        weight_col="analysis_weight",
        min_n=20,
    )

    rows: List[Dict[str, Any]] = []
    for b in EVENT_BIN_ORDER:
        if b == BASELINE_BIN:
            rows.append(
                {
                    "bin": b,
                    "midpoint_months": BIN_TO_MIDPOINT[b],
                    "coef": 0.0,
                    "std_err": np.nan,
                    "t_value": np.nan,
                    "is_pre": True,
                    "is_reference": True,
                }
            )
            continue

        term = f"did_{b}"
        coef = term_value(coef_df, term, "coef")
        se = term_value(coef_df, term, "std_err")
        t = term_value(coef_df, term, "t_value")
        rows.append(
            {
                "bin": b,
                "midpoint_months": BIN_TO_MIDPOINT[b],
                "coef": coef,
                "std_err": se,
                "t_value": t,
                "is_pre": b.startswith("lead"),
                "is_reference": False,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=coef_cols), pd.DataFrame(columns=diag_cols)

    out["ci_low"] = out["coef"] - 1.96 * out["std_err"].fillna(0.0)
    out["ci_high"] = out["coef"] + 1.96 * out["std_err"].fillna(0.0)
    out = out[coef_cols].sort_values("midpoint_months").reset_index(drop=True)

    lead = out[(out["is_pre"]) & (~out["is_reference"])].copy()
    lead_sig = lead["t_value"].abs() > 1.96
    diagnostics = pd.DataFrame(
        [
            {
                "n": int(info["n"]),
                "lead_bins": int(len(lead)),
                "lead_mean_abs_coef": float(lead["coef"].abs().mean()) if not lead.empty else np.nan,
                "lead_max_abs_t": float(lead["t_value"].abs().max()) if not lead.empty else np.nan,
                "lead_share_significant": float(lead_sig.mean()) if not lead.empty else np.nan,
            }
        ],
        columns=diag_cols,
    )

    return out, diagnostics


def raw_did_by_event(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["event_id", "event_n", "did_raw"]
    if panel.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for event_id, sub in panel.groupby("event_id"):
        rows.append(
            {
                "event_id": event_id,
                "event_n": int(len(sub)),
                "did_raw": did_from_means(sub, y_col="adoption_residual"),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def run_placebo_tests(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window_months: int,
    min_cell: int,
    n_random: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_cols = ["scenario", "mean_raw_did", "std_raw_did", "events_used", "p_vs_real_abs"]
    dist_cols = ["iteration", "random_placebo_mean_raw_did"]

    valid = events[events["valid_event"] & events["date_added"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=dist_cols)

    real_panel, _ = build_event_panel(df, valid, window_months=window_months, min_cell=min_cell)
    real_raw = raw_did_by_event(real_panel)
    real_effect = weighted_mean(real_raw["did_raw"], real_raw["event_n"]) if not real_raw.empty else np.nan

    rows: List[Dict[str, Any]] = [
        {
            "scenario": "real",
            "mean_raw_did": real_effect,
            "std_raw_did": float(real_raw["did_raw"].std(ddof=0)) if len(real_raw) > 1 else np.nan,
            "events_used": int(len(real_raw)),
            "p_vs_real_abs": np.nan,
        }
    ]

    for label, day_shift in [("placebo_minus_12m", -365), ("placebo_plus_12m", 365)]:
        shifted = valid.copy()
        shifted["date_added"] = shifted["date_added"] + pd.to_timedelta(day_shift, unit="D")
        ppanel, _ = build_event_panel(df, shifted, window_months=window_months, min_cell=min_cell)
        praw = raw_did_by_event(ppanel)
        peff = weighted_mean(praw["did_raw"], praw["event_n"]) if not praw.empty else np.nan

        rows.append(
            {
                "scenario": label,
                "mean_raw_did": peff,
                "std_raw_did": float(praw["did_raw"].std(ddof=0)) if len(praw) > 1 else np.nan,
                "events_used": int(len(praw)),
                "p_vs_real_abs": np.nan,
            }
        )

    rng = np.random.default_rng(seed)
    random_vals: List[Dict[str, Any]] = []
    min_release = df["release_date"].min()
    max_release = df["release_date"].max()

    if pd.notna(min_release) and pd.notna(max_release) and (max_release > min_release):
        buffer = pd.Timedelta(days=90)
        low = min_release + buffer
        high = max_release - buffer
        if high <= low:
            low, high = min_release, max_release

        span_days = int((high - low).days)
        if span_days > 0:
            for i in range(n_random):
                random_events = valid.copy()
                draws = rng.integers(0, span_days + 1, size=len(random_events))
                random_events["date_added"] = low + pd.to_timedelta(draws, unit="D")

                rpanel, _ = build_event_panel(df, random_events, window_months=window_months, min_cell=min_cell)
                rraw = raw_did_by_event(rpanel)
                if rraw.empty:
                    continue

                rmean = weighted_mean(rraw["did_raw"], rraw["event_n"])
                if pd.notna(rmean):
                    random_vals.append({"iteration": i + 1, "random_placebo_mean_raw_did": rmean})

    dist = pd.DataFrame(random_vals, columns=dist_cols)

    p_vs_real = np.nan
    if pd.notna(real_effect) and not dist.empty:
        p_vs_real = (np.sum(np.abs(dist["random_placebo_mean_raw_did"]) >= abs(real_effect)) + 1) / (len(dist) + 1)

    rows.append(
        {
            "scenario": "placebo_random_distribution",
            "mean_raw_did": float(dist["random_placebo_mean_raw_did"].mean()) if not dist.empty else np.nan,
            "std_raw_did": float(dist["random_placebo_mean_raw_did"].std(ddof=0)) if not dist.empty else np.nan,
            "events_used": int(valid["event_id"].nunique()),
            "p_vs_real_abs": p_vs_real,
        }
    )

    summary = pd.DataFrame(rows, columns=summary_cols)
    return summary, dist


def run_natural_experiments(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window_months: int = 24,
    min_cell: int = 1,
) -> pd.DataFrame:
    cols = [
        "arch_key",
        "shock_date",
        "paradigm",
        "framework_events",
        "n_treated",
        "treated_pre_n",
        "treated_post_n",
        "control_pre_n",
        "control_post_n",
        "did_raw",
        "did_coef",
        "did_se",
        "did_t",
        "shrink_factor",
        "did_shrunk",
    ]
    valid = events[events["valid_event"] & events["date_added"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=cols)

    sample_min = df["release_date"].min()
    sample_max = df["release_date"].max()
    if pd.isna(sample_min) or pd.isna(sample_max):
        return pd.DataFrame(columns=cols)

    first_shocks = valid.groupby("arch_key")["date_added"].min()
    arch_paradigm = dominant_paradigm_by_arch(df)

    rows: List[Dict[str, Any]] = []
    window_days = int(window_months * 30.44)

    for arch_key, shock_date in first_shocks.items():
        if pd.isna(shock_date):
            continue
        if not (sample_min < shock_date < sample_max):
            continue

        treated_mask = df["arch_key"].eq(arch_key)
        paradigm = arch_paradigm.get(str(arch_key), "unknown")

        in_window = df["release_date"].notna() & ((df["release_date"] - shock_date).dt.days.abs() <= window_days)

        control_mask = (~treated_mask) & df["inferred_paradigm"].eq(paradigm)
        sub = df[in_window & (treated_mask | control_mask)].copy()

        control_n = int((sub["arch_key"] != arch_key).sum())
        if control_n < 2 * max(min_cell, 1):
            control_mask = ~treated_mask
            sub = df[in_window & (treated_mask | control_mask)].copy()

        if sub.empty:
            continue

        sub["treated"] = (sub["arch_key"] == arch_key).astype(float)
        sub["post"] = (sub["release_date"] >= shock_date).astype(float)
        sub["treat_post"] = sub["treated"] * sub["post"]

        t_pre = int(((sub["treated"] == 1) & (sub["post"] == 0)).sum())
        t_post = int(((sub["treated"] == 1) & (sub["post"] == 1)).sum())
        c_pre = int(((sub["treated"] == 0) & (sub["post"] == 0)).sum())
        c_post = int(((sub["treated"] == 0) & (sub["post"] == 1)).sum())

        if min(t_pre, t_post, c_pre, c_post) < min_cell:
            continue

        did_raw = did_from_means(sub, y_col="adoption_residual")
        coef_df, _, _ = fit_ols(
            sub,
            y_col="adoption_residual",
            x_cols=["treated", "post", "treat_post", "log10_params", "recency_months"],
            fe_cols=["org", "release_month_str"],
            weight_col="imbalance_weight",
            min_n=8,
        )
        did_coef = term_value(coef_df, "treat_post", "coef")
        did_se = term_value(coef_df, "treat_post", "std_err")
        did_t = term_value(coef_df, "treat_post", "t_value")

        n_treated = t_pre + t_post
        shrink_factor = n_treated / (n_treated + 4.0)
        did_shrunk = did_raw * shrink_factor if pd.notna(did_raw) else np.nan

        rows.append(
            {
                "arch_key": arch_key,
                "shock_date": shock_date,
                "paradigm": paradigm,
                "framework_events": int((valid["arch_key"] == arch_key).sum()),
                "n_treated": n_treated,
                "treated_pre_n": t_pre,
                "treated_post_n": t_post,
                "control_pre_n": c_pre,
                "control_post_n": c_post,
                "did_raw": did_raw,
                "did_coef": did_coef,
                "did_se": did_se,
                "did_t": did_t,
                "shrink_factor": shrink_factor,
                "did_shrunk": did_shrunk,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    return out.sort_values("did_shrunk", ascending=False).reset_index(drop=True)


def build_prepost_cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "release_cohort",
        "inferred_paradigm",
        "model_count",
        "mean_adoption_residual",
        "median_downloads",
        "mean_log10_params",
        "mean_support_at_release",
        "mean_current_support",
        "mean_support_gain_after_release",
    ]
    work = df.dropna(subset=["post_first_shock_release", "adoption_residual"]).copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    work["release_cohort"] = np.where(
        work["post_first_shock_release"] >= 0.5,
        "post_shock_release",
        "pre_shock_release",
    )

    out = (
        work.groupby(["release_cohort", "inferred_paradigm"], dropna=False)
        .agg(
            model_count=("model_id", "count"),
            mean_adoption_residual=("adoption_residual", "mean"),
            median_downloads=("downloads", "median"),
            mean_log10_params=("log10_params", "mean"),
            mean_support_at_release=("support_score_at_release", "mean"),
            mean_current_support=("support_score", "mean"),
            mean_support_gain_after_release=("support_gain_after_release", "mean"),
        )
        .reset_index()
        .sort_values(["inferred_paradigm", "release_cohort"])
    )
    return out[cols]


def run_iv_analysis(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "term",
        "coef",
        "std_err",
        "t_value",
        "n",
        "r2",
        "first_stage_f",
        "weak_instrument_flag",
    ]

    needed = [
        "adoption_residual",
        "support_score",
        "support_score_at_release",
        "log10_params",
        "recency_months",
        "org",
        "release_month_str",
        "imbalance_weight",
    ]
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 20:
        return pd.DataFrame(columns=cols)

    fs_coef, fs_info, fs_fitted = fit_ols(
        work,
        y_col="support_score",
        x_cols=["support_score_at_release", "log10_params", "recency_months"],
        fe_cols=["org", "release_month_str"],
        weight_col="imbalance_weight",
        min_n=20,
    )
    if fs_coef.empty or fs_fitted.empty:
        return pd.DataFrame(columns=cols)

    work = work.loc[fs_fitted.index].copy()
    work["support_hat"] = fs_fitted

    iv_coef, iv_info, _ = fit_ols(
        work,
        y_col="adoption_residual",
        x_cols=["support_hat", "log10_params", "recency_months"],
        fe_cols=["org", "release_month_str"],
        weight_col="imbalance_weight",
        min_n=20,
    )

    ols_coef, ols_info, _ = fit_ols(
        work,
        y_col="adoption_residual",
        x_cols=["support_score", "log10_params", "recency_months"],
        fe_cols=["org", "release_month_str"],
        weight_col="imbalance_weight",
        min_n=20,
    )

    fs_t = term_value(fs_coef, "support_score_at_release", "t_value")
    first_stage_f = fs_t ** 2 if pd.notna(fs_t) else np.nan
    weak_flag = bool(pd.notna(first_stage_f) and first_stage_f < 10.0)

    out_rows = [
        {
            "model": "first_stage",
            "term": "support_score_at_release",
            "coef": term_value(fs_coef, "support_score_at_release", "coef"),
            "std_err": term_value(fs_coef, "support_score_at_release", "std_err"),
            "t_value": fs_t,
            "n": int(fs_info["n"]),
            "r2": fs_info["r2"],
            "first_stage_f": first_stage_f,
            "weak_instrument_flag": weak_flag,
        },
        {
            "model": "ols_naive",
            "term": "support_score",
            "coef": term_value(ols_coef, "support_score", "coef"),
            "std_err": term_value(ols_coef, "support_score", "std_err"),
            "t_value": term_value(ols_coef, "support_score", "t_value"),
            "n": int(ols_info["n"]),
            "r2": ols_info["r2"],
            "first_stage_f": first_stage_f,
            "weak_instrument_flag": weak_flag,
        },
        {
            "model": "iv_2sls",
            "term": "support_hat",
            "coef": term_value(iv_coef, "support_hat", "coef"),
            "std_err": term_value(iv_coef, "support_hat", "std_err"),
            "t_value": term_value(iv_coef, "support_hat", "t_value"),
            "n": int(iv_info["n"]),
            "r2": iv_info["r2"],
            "first_stage_f": first_stage_f,
            "weak_instrument_flag": weak_flag,
        },
    ]
    return pd.DataFrame(out_rows, columns=cols)


def save_figures(
    event_study: pd.DataFrame,
    placebo_dist: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    natural_exp: pd.DataFrame,
    out_figs: Path,
) -> None:
    out_figs.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    if not event_study.empty:
        plot_df = event_study.sort_values("midpoint_months").copy()
        nr = plot_df[~plot_df["is_reference"]].copy()

        plt.figure(figsize=(11, 6))
        if not nr.empty:
            yerr = 1.96 * nr["std_err"].fillna(0.0)
            plt.errorbar(
                nr["midpoint_months"],
                nr["coef"],
                yerr=yerr,
                fmt="o",
                capsize=4,
                color="#1f77b4",
                label="Estimate +/- 95% CI",
            )
            plt.plot(nr["midpoint_months"], nr["coef"], color="#1f77b4", linewidth=1.5)

        ref = plot_df[plot_df["is_reference"]]
        if not ref.empty:
            plt.scatter(
                ref["midpoint_months"],
                ref["coef"],
                color="black",
                marker="D",
                s=70,
                label="Reference bin ([-6,0) months)",
            )

        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.xticks(plot_df["midpoint_months"], plot_df["bin"], rotation=30, ha="right")
        plt.ylabel("Relative DiD effect on adoption residual")
        plt.xlabel("Event-time bin")
        plt.title("Event Study: Adoption Shift Around Tooling Support Shocks")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_figs / "01_event_study_coefficients.png", dpi=240)
        plt.close()

    if not placebo_dist.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(placebo_dist["random_placebo_mean_raw_did"], bins=30, color="#4c78a8", alpha=0.75)

        def scenario_mean(name: str) -> float:
            row = placebo_summary[placebo_summary["scenario"] == name]
            if row.empty:
                return np.nan
            return float(row["mean_raw_did"].iloc[0])

        real = scenario_mean("real")
        minus12 = scenario_mean("placebo_minus_12m")
        plus12 = scenario_mean("placebo_plus_12m")

        if pd.notna(real):
            plt.axvline(real, color="#d62728", linewidth=2.5, label=f"Real ({real:+.3f})")
        if pd.notna(minus12):
            plt.axvline(minus12, color="#2ca02c", linestyle="--", linewidth=2, label=f"-12m placebo ({minus12:+.3f})")
        if pd.notna(plus12):
            plt.axvline(plus12, color="#ff7f0e", linestyle="--", linewidth=2, label=f"+12m placebo ({plus12:+.3f})")

        plt.xlabel("Mean raw DiD effect")
        plt.ylabel("Count")
        plt.title("Placebo Test: Randomized Shock Dates vs Real Effect")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_figs / "02_placebo_distribution.png", dpi=240)
        plt.close()

    if not natural_exp.empty:
        plot_df = natural_exp.sort_values("did_shrunk").copy()
        fig_h = max(4.0, 0.35 * len(plot_df))
        plt.figure(figsize=(10, fig_h))
        sns.barplot(data=plot_df, x="did_shrunk", y="arch_key", color="#2ca02c")
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Shrunk DiD effect on adoption residual")
        plt.ylabel("Architecture")
        plt.title("Natural Experiments by Architecture (Empirical-Bayes Shrinkage)")
        plt.tight_layout()
        plt.savefig(out_figs / "03_natural_experiment_effects.png", dpi=240)
        plt.close()


def write_key_findings(
    df: pd.DataFrame,
    events: pd.DataFrame,
    panel: pd.DataFrame,
    pooled_did: pd.DataFrame,
    event_did: pd.DataFrame,
    pretrend_diag: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    iv_summary: pd.DataFrame,
    natural_exp: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []

    total = len(df)
    mapped = int((df["arch_key"] != "unknown").sum()) if "arch_key" in df.columns else 0
    lines.append(f"models_total: {total}")
    lines.append(f"models_with_arch_mapping: {mapped} ({(mapped / total) if total else 0:.1%})")

    if not events.empty:
        lines.append(f"timeline_entries_total: {len(events)}")
        lines.append(f"timeline_dates_filled: {int(events['has_date'].sum())}")
        lines.append(f"valid_support_events: {int(events['valid_event'].sum())}")
        lines.append(f"mid_sample_support_events: {int(events['mid_sample_shock'].sum())}")
    else:
        lines.append("timeline_entries_total: 0")

    if not panel.empty:
        lines.append(f"stacked_event_rows: {len(panel)}")
        lines.append(f"stacked_events_used: {panel['event_id'].nunique()}")

    if not pooled_did.empty:
        full = pooled_did.iloc[-1]
        lines.append(
            f"pooled_did_full_spec: coef={full['did_coef']:+.4f}, t={full['did_t']:+.2f}, n={int(full['n'])}, r2={full['r2']:.3f}"
        )

    if not event_did.empty:
        w_did_coef = weighted_mean(event_did["did_coef"], event_did["event_n"])
        w_did_raw = weighted_mean(event_did["did_raw"], event_did["event_n"])
        pre_pass_rate = float(event_did["pretrend_pass"].mean())
        lines.append(f"event_weighted_did_coef: {w_did_coef:+.4f}")
        lines.append(f"event_weighted_did_raw: {w_did_raw:+.4f}")
        lines.append(f"event_pretrend_pass_rate: {pre_pass_rate:.1%}")

    if not pretrend_diag.empty:
        p = pretrend_diag.iloc[0]
        lines.append(f"event_study_lead_bins: {int(p['lead_bins'])}")
        lines.append(f"event_study_lead_mean_abs_coef: {p['lead_mean_abs_coef']:.4f}")
        lines.append(f"event_study_lead_share_significant: {p['lead_share_significant']:.1%}")

    if not placebo_summary.empty:
        rand = placebo_summary[placebo_summary["scenario"] == "placebo_random_distribution"]
        if not rand.empty:
            r = rand.iloc[0]
            lines.append(f"placebo_random_mean: {r['mean_raw_did']:+.4f}")
            lines.append(f"placebo_random_std: {r['std_raw_did']:.4f}")
            lines.append(f"placebo_p_vs_real_abs: {r['p_vs_real_abs']:.4f}")

    if not iv_summary.empty:
        iv_row = iv_summary[iv_summary["model"] == "iv_2sls"]
        if not iv_row.empty:
            r = iv_row.iloc[0]
            lines.append(f"iv_2sls_support_effect: {r['coef']:+.4f} (t={r['t_value']:+.2f})")
            lines.append(f"iv_first_stage_f: {r['first_stage_f']:.2f}")
            lines.append(f"iv_weak_instrument_flag: {'YES' if bool(r['weak_instrument_flag']) else 'NO'}")

    if not natural_exp.empty:
        top = natural_exp.iloc[0]
        lines.append(
            f"top_natural_experiment: {top['arch_key']} (did_shrunk={top['did_shrunk']:+.4f}, n_treated={int(top['n_treated'])})"
        )

    if not events.empty and int(events["has_date"].sum()) == 0:
        lines.append("action_required: FRAMEWORK_SUPPORT_TIMELINE still has placeholder dates; fill YYYY-MM-DD values and rerun.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal causal analysis for tooling-support shocks vs model adoption."
    )
    parser.add_argument("--input", default="metadata_enriched.json", help="Path to metadata_enriched.json")
    parser.add_argument("--outdir", default="analysis/temporal_causal", help="Output directory")
    parser.add_argument("--window-months", type=int, default=18, help="Event window half-width in months")
    parser.add_argument("--min-cell", type=int, default=2, help="Minimum treated/control pre/post cell size per event")
    parser.add_argument("--placebo-random", type=int, default=500, help="Number of random placebo iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for placebo draws")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    out_tables = outdir / "tables"
    out_figs = outdir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)

    meta, df = load_dataframe(input_path)
    df = attach_support_metrics(df)

    events = build_timeline_events(df, FRAMEWORK_SUPPORT_TIMELINE)
    df = add_release_timing_features(df, events)

    panel, _ = build_event_panel(
        df,
        events,
        window_months=args.window_months,
        min_cell=args.min_cell,
    )

    pooled_did = run_pooled_did_robustness(panel)
    event_did = run_event_level_did(panel)
    event_study, pretrend_diag = run_event_study(panel)
    placebo_summary, placebo_dist = run_placebo_tests(
        df,
        events,
        window_months=args.window_months,
        min_cell=args.min_cell,
        n_random=args.placebo_random,
        seed=args.seed,
    )
    natural_exp = run_natural_experiments(
        df,
        events,
        window_months=max(args.window_months, 24),
        min_cell=max(1, args.min_cell - 1),
    )
    cohort_summary = build_prepost_cohort_summary(df)
    iv_summary = run_iv_analysis(df)

    model_cols = [
        "model_id",
        "org",
        "inferred_paradigm",
        "arch_key",
        "release_date",
        "release_month",
        "downloads",
        "log10_downloads",
        "adoption_residual",
        "log10_params",
        "param_b",
        "recency_months",
        "support_score",
        "measured_tax",
        "support_score_at_release",
        "support_gain_after_release",
        "frameworks_supported_at_release",
        "first_shock_date",
        "post_first_shock_release",
        "months_from_first_shock",
        "imbalance_weight",
    ]
    model_cols = [c for c in model_cols if c in df.columns]

    df[model_cols].to_csv(out_tables / "model_temporal_features.csv", index=False)
    events.to_csv(out_tables / "framework_support_events.csv", index=False)
    panel.to_csv(out_tables / "event_panel_stacked.csv", index=False)
    pooled_did.to_csv(out_tables / "did_pooled_robustness.csv", index=False)
    event_did.to_csv(out_tables / "did_event_estimates.csv", index=False)
    event_study.to_csv(out_tables / "event_study_coefficients.csv", index=False)
    pretrend_diag.to_csv(out_tables / "event_study_pretrend_diagnostics.csv", index=False)
    placebo_summary.to_csv(out_tables / "placebo_test_summary.csv", index=False)
    placebo_dist.to_csv(out_tables / "placebo_test_distribution.csv", index=False)
    natural_exp.to_csv(out_tables / "natural_experiment_architecture.csv", index=False)
    cohort_summary.to_csv(out_tables / "cohort_prepost_summary.csv", index=False)
    iv_summary.to_csv(out_tables / "iv_2sls_summary.csv", index=False)

    save_figures(
        event_study=event_study,
        placebo_dist=placebo_dist,
        placebo_summary=placebo_summary,
        natural_exp=natural_exp,
        out_figs=out_figs,
    )

    write_key_findings(
        df=df,
        events=events,
        panel=panel,
        pooled_did=pooled_did,
        event_did=event_did,
        pretrend_diag=pretrend_diag,
        placebo_summary=placebo_summary,
        iv_summary=iv_summary,
        natural_exp=natural_exp,
        out_path=outdir / "key_findings.txt",
    )

    print(f"Analyzed {len(df)} models")
    print(f"Output tables: {out_tables}")
    print(f"Output figures: {out_figs}")
    print(f"Key findings: {outdir / 'key_findings.txt'}")
    if isinstance(meta, dict):
        print(f"source_file={meta.get('source_file')}, generated_at_utc={meta.get('generated_at_utc')}")
    if not events.empty and int(events["has_date"].sum()) == 0:
        print("WARNING: FRAMEWORK_SUPPORT_TIMELINE has no valid dates. Fill placeholders and rerun.")
    if panel.empty:
        print("WARNING: No valid stacked events passed min-cell checks. Check timeline dates and window settings.")


if __name__ == "__main__":
    main()
