#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ecosystem_analysis import add_derived_metrics, flatten_model, load_payload, short_name
from ecosystem_deep_dynamics import add_adoption_residual, add_research_features, safe_zscore

EPS = 1e-12
DTYPE_BYTES = 2.0

LAYER_LIST_KEY_HINTS = (
    "layers_block_type",
    "layer_types",
    "block_types",
    "layer_type_pattern",
    "layer_pattern",
    "layer_layout",
    "layer_sequence",
    "mix_pattern",
)
ATTN_INDEX_KEY_HINTS = (
    "attention_layers",
    "attn_layer_indices",
    "attn_layers",
    "attn_layer_ids",
    "attention_layer_ids",
    "hybrid_layer_ids",
    "attn_indices",
    "attention_indices",
    "attn_idx",
    "attention_idx",
)
TOTAL_LAYER_KEYS = (
    "num_hidden_layers",
    "num_layers",
    "n_layer",
    "n_layers",
    "decoder_layers",
    "num_blocks",
    "layers",
)
PERIOD_KEY_HINTS = ("period", "every", "stride", "interval")
OFFSET_KEY_HINTS = ("offset", "start")

RECURSIVE_PARADIGMS = {"hybrid", "ssm", "liquid", "rwkv", "xlstm"}

FRAMEWORK_BASE_SUPPORT = {
    "transformer": 0.95,
    "reasoning": 0.90,
    "diffusion": 0.82,
    "hybrid": 0.58,
    "ssm": 0.48,
    "liquid": 0.42,
    "rwkv": 0.50,
    "xlstm": 0.50,
    "unassigned": 0.75,
}
LOWER_IS_BETTER_HINTS = (
    "latency",
    "time",
    "ms",
    "sec",
    "seconds",
    "error",
    "loss",
    "perplex",
    "ppl",
    "bpb",
    "wer",
    "cer",
)


def coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return None if np.isnan(out) else out
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return None
        try:
            out = float(s)
            return None if np.isnan(out) else out
        except ValueError:
            match = re.search(r"-?\d+(?:\.\d+)?", s)
            if not match:
                return None
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def coerce_int(value: Any) -> Optional[int]:
    out = coerce_float(value)
    if out is None:
        return None
    return int(out)


def maybe_parse_json_string(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        parsed = json.loads(s)
    except Exception:
        return None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None


def get_nested_value(node: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = node
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def iter_key_values(node: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(node, Mapping):
        for raw_key, value in node.items():
            key = str(raw_key).lower()
            path = f"{prefix}.{key}" if prefix else key
            yield path, value
            yield from iter_key_values(value, path)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            path = f"{prefix}[{idx}]"
            yield from iter_key_values(value, path)


def leaf_key(path: str) -> str:
    parts = [p for p in re.split(r"[.\[\]]+", path) if p]
    return parts[-1] if parts else path


def normalize_layer_token(token: str) -> str:
    t = str(token).strip().lower()
    if not t:
        return "unknown"
    if any(k in t for k in ("attn", "attention", "self_attention", "self-attention", "mha", "gqa", "transformer")):
        return "attn"
    if any(k in t for k in ("ssm", "mamba", "state", "rwkv", "xlstm", "lstm", "liquid", "rnn", "s4", "hyena")):
        return "ssm"
    return "other"


def parse_layer_list_candidate(value: Any) -> List[str]:
    if not isinstance(value, list) or len(value) < 2:
        return []
    seq: List[str] = []
    for item in value:
        if isinstance(item, Mapping):
            label = (
                item.get("type")
                or item.get("layer_type")
                or item.get("block_type")
                or item.get("name")
                or item.get("module")
                or item.get("op")
                or ""
            )
            seq.append(normalize_layer_token(str(label)))
        elif isinstance(item, str):
            seq.append(normalize_layer_token(item))
        elif isinstance(item, (int, np.integer)):
            if int(item) in (0, 1):
                seq.append("attn" if int(item) == 1 else "ssm")
            else:
                seq.append("other")
        elif isinstance(item, (float, np.floating)) and not np.isnan(item):
            ival = int(item)
            if ival in (0, 1):
                seq.append("attn" if ival == 1 else "ssm")
            else:
                seq.append("other")
        else:
            seq.append("unknown")

    useful = sum(x in {"attn", "ssm"} for x in seq)
    if useful >= max(2, int(0.2 * len(seq))):
        return seq
    return []


def parse_indices_from_value(value: Any) -> List[int]:
    out: List[int] = []

    if isinstance(value, list):
        if value and all(isinstance(v, bool) for v in value):
            out.extend([i for i, flag in enumerate(value) if flag])
        else:
            for v in value:
                iv = coerce_int(v)
                if iv is not None and 0 <= iv <= 4096:
                    out.append(iv)

    elif isinstance(value, Mapping):
        for raw_key, raw_val in value.items():
            if isinstance(raw_val, bool):
                if raw_val:
                    iv_key = coerce_int(raw_key)
                    if iv_key is not None and 0 <= iv_key <= 4096:
                        out.append(iv_key)
            else:
                iv_val = coerce_int(raw_val)
                if iv_val is not None and 0 <= iv_val <= 4096:
                    out.append(iv_val)

    elif isinstance(value, str):
        out.extend([int(x) for x in re.findall(r"\d+", value)])

    return sorted(set(i for i in out if 0 <= i <= 4096))


def minmax_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    lo = float(valid.min())
    hi = float(valid.max())
    if hi - lo <= EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def has_ssm_evidence(model: Mapping[str, Any]) -> bool:
    ps = model.get("paradigm_specific") or {}
    strong_fields = ("ssm_state_size", "ssm_conv_kernel", "ssm_dt_rank", "num_ssm_layers")
    if any((coerce_int(ps.get(k)) or 0) > 0 for k in strong_fields):
        return True

    detected = ps.get("detected_numeric_fields") or {}
    for raw_key, raw_val in detected.items():
        key = str(raw_key).lower()
        if any(tok in key for tok in ("ssm", "mamba", "state", "rwkv", "xlstm", "lstm", "liquid", "hybrid")):
            iv = coerce_int(raw_val)
            if iv is None or iv != 0:
                return True
    return False


def get_raw_config(model: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    candidates = [
        model.get("raw_config_json"),
        model.get("raw_config"),
        model.get("config_json"),
        get_nested_value(model, ("raw", "config_json")),
        get_nested_value(model, ("raw", "config")),
        get_nested_value(model, ("metadata", "config_json")),
        get_nested_value(model, ("metadata", "raw_config_json")),
        get_nested_value(model, ("artifacts", "config_json")),
        get_nested_value(model, ("sources", "config_json")),
    ]

    for cand in candidates:
        if isinstance(cand, Mapping):
            return dict(cand), "embedded_mapping"
        parsed = maybe_parse_json_string(cand)
        if isinstance(parsed, Mapping):
            return parsed, "embedded_json_string"

    return {}, "missing"


def find_total_layers_from_raw(raw_items: Sequence[Tuple[str, Any]]) -> Optional[int]:
    for path, value in raw_items:
        key = leaf_key(path)
        if key in TOTAL_LAYER_KEYS:
            iv = coerce_int(value)
            if iv is not None and 1 <= iv <= 4096:
                return iv
    return None


def find_layer_count_from_raw(raw_items: Sequence[Tuple[str, Any]], mode: str) -> Optional[int]:
    if mode not in {"attn", "ssm"}:
        return None
    for path, value in raw_items:
        key = leaf_key(path)
        if "layer" not in key:
            continue
        if mode == "attn" and not any(tok in key for tok in ("attn", "attention")):
            continue
        if mode == "ssm" and not any(tok in key for tok in ("ssm", "mamba", "rwkv", "xlstm", "liquid", "state")):
            continue
        iv = coerce_int(value)
        if iv is not None and 0 <= iv <= 4096:
            return iv
    return None


def compute_transition_complexity(
    attn_ratio: Optional[float],
    transition_rate: Optional[float],
    periodicity: Optional[float],
) -> float:
    if attn_ratio is None or pd.isna(attn_ratio):
        return np.nan
    if attn_ratio <= EPS or attn_ratio >= 1 - EPS:
        return 0.0

    mix_balance = float(np.clip(1.0 - abs(float(attn_ratio) - 0.5) / 0.5, 0.0, 1.0))
    if periodicity is None or pd.isna(periodicity):
        irregularity = 0.5
    else:
        irregularity = 1.0 - float(np.clip(periodicity, 0.0, 1.0))

    if transition_rate is None or pd.isna(transition_rate):
        score = 0.70 * mix_balance + 0.30 * irregularity
    else:
        score = 0.60 * float(np.clip(transition_rate, 0.0, 1.0)) + 0.25 * mix_balance + 0.15 * irregularity

    return float(np.clip(score, 0.0, 1.0))


def extract_layer_dna(model: Mapping[str, Any]) -> Dict[str, Any]:
    arch = model.get("architecture") or {}
    ps = model.get("paradigm_specific") or {}
    registry = model.get("registry") or {}

    raw_cfg, cfg_source = get_raw_config(model)
    raw_items = list(iter_key_values(raw_cfg))

    total_layers = coerce_int(arch.get("num_layers"))
    if total_layers is None:
        total_layers = find_total_layers_from_raw(raw_items)

    layer_sequence: List[str] = []
    layer_source = "none"

    for path, value in raw_items:
        key = leaf_key(path)
        if any(hint in key for hint in LAYER_LIST_KEY_HINTS):
            seq = parse_layer_list_candidate(value)
            if len(seq) >= 2:
                layer_sequence = seq
                layer_source = f"layer_list:{key}"
                break

    attn_indices: List[int] = []
    for path, value in raw_items:
        key = leaf_key(path)
        if any(hint in key for hint in ATTN_INDEX_KEY_HINTS):
            idx = parse_indices_from_value(value)
            if idx:
                attn_indices = idx
                if layer_source == "none":
                    layer_source = f"attn_indices:{key}"
                break

    attn_period_raw = np.nan
    attn_offset_raw = 0
    for path, value in raw_items:
        key = leaf_key(path)
        if "attn" not in key and "attention" not in key:
            continue
        if any(tok in key for tok in PERIOD_KEY_HINTS):
            iv = coerce_int(value)
            if iv is not None and 1 <= iv <= 512:
                attn_period_raw = float(iv)
        if any(tok in key for tok in OFFSET_KEY_HINTS):
            iv = coerce_int(value)
            if iv is not None and 0 <= iv <= 512:
                attn_offset_raw = int(iv)

    if not attn_indices and pd.notna(attn_period_raw) and total_layers is not None and total_layers > 0:
        period = int(attn_period_raw)
        start = attn_offset_raw if attn_offset_raw < total_layers else 0
        attn_indices = list(range(start, total_layers, period))
        if not attn_indices and total_layers > 0:
            attn_indices = [0]
        if layer_source == "none":
            layer_source = "periodic_schedule"

    if not layer_sequence and attn_indices and total_layers is not None and total_layers > 0:
        ssm_evidence = has_ssm_evidence(model)
        default_other = "ssm" if ssm_evidence else "other"
        idx_set = {i for i in attn_indices if 0 <= i < total_layers}
        layer_sequence = ["attn" if i in idx_set else default_other for i in range(total_layers)]
        if layer_source == "none":
            layer_source = "attn_indices_constructed"

    if total_layers is None and layer_sequence:
        total_layers = len(layer_sequence)

    attn_count: Optional[int] = None
    ssm_count: Optional[int] = None
    if layer_sequence:
        attn_count = int(sum(x == "attn" for x in layer_sequence))
        ssm_count = int(sum(x == "ssm" for x in layer_sequence))

    if attn_count is None:
        attn_count = find_layer_count_from_raw(raw_items, mode="attn")
    if ssm_count is None:
        ssm_count = find_layer_count_from_raw(raw_items, mode="ssm")

    if attn_count is None:
        attn_count = coerce_int(ps.get("num_attention_layers"))
    if ssm_count is None:
        ssm_count = coerce_int(ps.get("num_ssm_layers"))

    ps_attn_ratio = coerce_float(ps.get("attention_layer_ratio"))
    ps_ssm_ratio = coerce_float(ps.get("ssm_layer_ratio"))

    if total_layers is not None and total_layers > 0:
        if attn_count is None and ps_attn_ratio is not None and 0 <= ps_attn_ratio <= 1:
            attn_count = int(round(total_layers * ps_attn_ratio))
        if ssm_count is None and ps_ssm_ratio is not None and 0 <= ps_ssm_ratio <= 1:
            ssm_count = int(round(total_layers * ps_ssm_ratio))

    if total_layers is not None and total_layers > 0:
        if attn_count is not None and ssm_count is None and 0 <= attn_count <= total_layers:
            ssm_count = total_layers - attn_count
        if ssm_count is not None and attn_count is None and 0 <= ssm_count <= total_layers:
            attn_count = total_layers - ssm_count

    paradigm = str(registry.get("paradigm") or "").lower()
    has_attention = coerce_int(arch.get("num_heads")) is not None
    ssm_evidence = has_ssm_evidence(model)

    if total_layers is not None and total_layers > 0 and attn_count is None and ssm_count is None:
        if paradigm in {"transformer", "reasoning", "diffusion"} or (has_attention and not ssm_evidence):
            attn_count, ssm_count = total_layers, 0
            if layer_source == "none":
                layer_source = "count_fallback:attention_only"
        elif paradigm in {"ssm", "liquid", "rwkv", "xlstm"} or (ssm_evidence and not has_attention):
            attn_count, ssm_count = 0, total_layers
            if layer_source == "none":
                layer_source = "count_fallback:ssm_only"
        elif has_attention and ssm_evidence:
            attn_count = max(1, int(round(total_layers * 0.25)))
            attn_count = min(attn_count, total_layers)
            ssm_count = total_layers - attn_count
            if layer_source == "none":
                layer_source = "count_fallback:hybrid_split"

    if total_layers is None and attn_count is not None and ssm_count is not None:
        total_layers = attn_count + ssm_count

    if total_layers is not None and attn_count is not None:
        attn_count = int(np.clip(attn_count, 0, total_layers))
    if total_layers is not None and ssm_count is not None:
        ssm_count = int(np.clip(ssm_count, 0, total_layers))

    if not attn_indices and layer_sequence:
        attn_indices = [i for i, token in enumerate(layer_sequence) if token == "attn"]

    attn_periodicity = np.nan
    attn_period_mode = np.nan
    if len(attn_indices) >= 2:
        diffs = np.diff(sorted(set(attn_indices)))
        if len(diffs) > 0:
            vals, counts = np.unique(diffs, return_counts=True)
            best = int(np.argmax(counts))
            attn_period_mode = float(vals[best])
            attn_periodicity = float(counts[best] / len(diffs))
    if pd.notna(attn_period_raw):
        attn_period_mode = float(attn_period_raw)
        if pd.isna(attn_periodicity):
            attn_periodicity = 1.0

    binary_seq = [x for x in layer_sequence if x in {"attn", "ssm"}]
    transition_count = np.nan
    transition_rate = np.nan
    if len(binary_seq) >= 2:
        transition_count = int(sum(binary_seq[i] != binary_seq[i - 1] for i in range(1, len(binary_seq))))
        transition_rate = float(transition_count / (len(binary_seq) - 1))

    attn_ratio = np.nan
    ssm_ratio = np.nan
    if total_layers is not None and total_layers > 0:
        if attn_count is not None:
            attn_ratio = float(attn_count / total_layers)
        if ssm_count is not None:
            ssm_ratio = float(ssm_count / total_layers)
    elif attn_count is not None and ssm_count is not None and (attn_count + ssm_count) > 0:
        denom = attn_count + ssm_count
        attn_ratio = float(attn_count / denom)
        ssm_ratio = float(ssm_count / denom)

    if pd.isna(ssm_ratio) and pd.notna(attn_ratio):
        ssm_ratio = 1.0 - float(attn_ratio)
    if pd.isna(attn_ratio) and pd.notna(ssm_ratio):
        attn_ratio = 1.0 - float(ssm_ratio)

    layer_entropy = np.nan
    if layer_sequence:
        seq_ser = pd.Series(layer_sequence)
        seq_ser = seq_ser[seq_ser.isin(["attn", "ssm", "other"])]
        if not seq_ser.empty:
            probs = seq_ser.value_counts(normalize=True).to_numpy(dtype=float)
            layer_entropy = float(-(probs * np.log(probs + EPS)).sum())

    transition_complexity = compute_transition_complexity(attn_ratio, transition_rate, attn_periodicity)

    return {
        "model_id": model.get("model_id"),
        "raw_config_present": cfg_source != "missing",
        "raw_config_source": cfg_source,
        "raw_config_key_count": len(raw_items),
        "layer_dna_source": layer_source,
        "layer_total_estimate": total_layers,
        "layer_sequence_length": len(layer_sequence) if layer_sequence else np.nan,
        "attn_layer_count": attn_count,
        "ssm_layer_count": ssm_count,
        "attn_layer_ratio": attn_ratio,
        "ssm_layer_ratio_dna": ssm_ratio,
        "attn_index_count": len(attn_indices),
        "attn_period": attn_period_mode,
        "attn_periodicity": attn_periodicity,
        "layer_transition_count": transition_count,
        "layer_transition_rate": transition_rate,
        "layer_type_entropy": layer_entropy,
        "transition_complexity": transition_complexity,
    }


def load_mechanism_dataframe(input_path: Path) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    meta, models = load_payload(input_path)
    df = pd.DataFrame([flatten_model(m) for m in models])

    extras: List[Dict[str, Any]] = []
    for m in models:
        ps = m.get("paradigm_specific") or {}
        row: Dict[str, Any] = {
            "model_id": m.get("model_id"),
            "num_ssm_layers": ps.get("num_ssm_layers"),
            "num_attention_layers": ps.get("num_attention_layers"),
            "ssm_layer_ratio": ps.get("ssm_layer_ratio"),
            "attention_layer_ratio": ps.get("attention_layer_ratio"),
        }
        row.update(extract_layer_dna(m))
        extras.append(row)

    extra_df = pd.DataFrame(extras)
    df = df.merge(extra_df, on="model_id", how="left")

    df = add_derived_metrics(df)
    df = add_research_features(df)

    df["is_long_context_128k_f"] = df["is_long_context_128k"].fillna(False).astype(float)
    df["has_quant_f"] = df["has_quant"].fillna(False).astype(float)
    df["has_ssm_signals_f"] = df["has_ssm_signals"].fillna(False).astype(float)

    df, adoption_baseline = add_adoption_residual(df)
    return meta, df, adoption_baseline


def add_efficiency_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "hidden_size",
        "ssm_state_size",
        "ssm_conv_kernel",
        "attn_layer_count",
        "ssm_layer_count",
        "num_attention_layers",
        "num_ssm_layers",
        "ssm_layer_ratio",
        "ssm_layer_ratio_dna",
        "attn_layer_ratio",
        "attn_periodicity",
        "layer_transition_rate",
        "transition_complexity",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    ssm_ratio_guess = out["ssm_layer_ratio_dna"].copy()
    ssm_ratio_guess = ssm_ratio_guess.where(ssm_ratio_guess.notna(), out["ssm_layer_ratio"])
    ssm_ratio_guess = ssm_ratio_guess.where(
        ssm_ratio_guess.notna(),
        np.where(out["has_ssm_signals"].fillna(False), 0.75, 0.0),
    )

    attn_layers = out["attn_layer_count"].copy()
    attn_layers = attn_layers.where(attn_layers.notna(), out["num_attention_layers"])
    attn_layers = attn_layers.where(attn_layers.notna(), out["num_layers"] * (1.0 - ssm_ratio_guess))
    attn_layers = pd.to_numeric(attn_layers, errors="coerce").clip(lower=0)

    ssm_layers = out["ssm_layer_count"].copy()
    ssm_layers = ssm_layers.where(ssm_layers.notna(), out["num_ssm_layers"])
    ssm_layers = ssm_layers.where(ssm_layers.notna(), (out["num_layers"] - attn_layers).clip(lower=0))
    ssm_layers = pd.to_numeric(ssm_layers, errors="coerce").clip(lower=0)

    kv_heads = out["num_kv_heads"].where(out["num_kv_heads"].notna(), out["num_heads"])
    head_dim = out["head_dim"].copy()
    need_head_dim = head_dim.isna() & out["hidden_size"].notna() & out["num_heads"].notna() & (out["num_heads"] > 0)
    head_dim.loc[need_head_dim] = out.loc[need_head_dim, "hidden_size"] / out.loc[need_head_dim, "num_heads"]

    bytes_per_token = 2.0 * kv_heads * head_dim * DTYPE_BYTES
    kv_cache_mb = (bytes_per_token * attn_layers * 1000.0) / (1024.0 ** 2)
    out["kv_cache_mb_per_1k_tokens"] = kv_cache_mb.replace([np.inf, -np.inf], np.nan)

    hidden_state = out["hidden_size"].copy()
    need_hidden = hidden_state.isna() & out["num_heads"].notna() & head_dim.notna()
    hidden_state.loc[need_hidden] = out.loc[need_hidden, "num_heads"] * head_dim.loc[need_hidden]

    state_size = out["ssm_state_size"].copy()
    state_size = state_size.where(state_size.notna(), np.where(ssm_layers > 0, 16.0, np.nan))
    conv_kernel = out["ssm_conv_kernel"].fillna(0.0)
    state_slots = state_size + conv_kernel

    state_bytes = ssm_layers * hidden_state * state_slots * DTYPE_BYTES
    out["state_memory_proxy"] = (state_bytes / (1024.0 ** 2)).replace([np.inf, -np.inf], np.nan)

    out["log1p_kv_cache_mb_per_1k_tokens"] = np.log1p(out["kv_cache_mb_per_1k_tokens"].clip(lower=0))
    out["log1p_state_memory_proxy"] = np.log1p(out["state_memory_proxy"].clip(lower=0))

    total_layers = pd.to_numeric(out["layer_total_estimate"], errors="coerce")
    total_layers = total_layers.where(total_layers.notna(), out["num_layers"])

    attn_ratio = out["attn_layer_ratio"].copy()
    need_ratio = attn_ratio.isna() & total_layers.notna() & (total_layers > 0)
    attn_ratio.loc[need_ratio] = attn_layers.loc[need_ratio] / total_layers.loc[need_ratio]

    periodicity = pd.to_numeric(out["attn_periodicity"], errors="coerce")
    mix_balance = (1.0 - (attn_ratio - 0.5).abs() / 0.5).clip(lower=0, upper=1)
    irregularity = (1.0 - periodicity).where(periodicity.notna(), 0.5)
    complexity_fallback = (0.70 * mix_balance + 0.30 * irregularity).clip(lower=0, upper=1)

    complexity = pd.to_numeric(out["transition_complexity"], errors="coerce")
    complexity = complexity.where(complexity.notna(), complexity_fallback)
    pure_mask = (attn_ratio <= EPS) | (attn_ratio >= 1 - EPS)
    complexity = complexity.where(~pure_mask, 0.0)

    out["attn_layer_count"] = attn_layers
    out["ssm_layer_count"] = ssm_layers
    out["attn_layer_ratio"] = attn_ratio
    out["ssm_layer_ratio_dna"] = out["ssm_layer_ratio_dna"].where(
        out["ssm_layer_ratio_dna"].notna(),
        np.where(total_layers > 0, ssm_layers / total_layers, np.nan),
    )
    out["transition_complexity"] = complexity.clip(lower=0, upper=1)

    return out


def add_compatibility_tax_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    base_support = out["inferred_paradigm"].fillna("unassigned").map(FRAMEWORK_BASE_SUPPORT).fillna(0.70)
    transition = pd.to_numeric(out["transition_complexity"], errors="coerce").fillna(0.0)
    kv_norm = minmax_scale(out["log1p_kv_cache_mb_per_1k_tokens"]).fillna(0.0)
    state_norm = minmax_scale(out["log1p_state_memory_proxy"]).fillna(0.0)
    hybrid = pd.to_numeric(out["hybridization_score"], errors="coerce").fillna(0.0)

    quant_relief = out["has_quant"].fillna(False).astype(float) * 0.04
    quant_relief += out["has_gguf"].fillna(False).astype(float) * 0.03

    support = base_support - 0.20 * transition - 0.11 * state_norm - 0.07 * kv_norm - 0.08 * hybrid + quant_relief
    support = support.clip(lower=0.05, upper=0.99)

    out["framework_support_index"] = support
    out["compatibility_tax"] = (1.0 - support).clip(lower=0.0, upper=1.0)
    out["tooling_tax_index"] = out["compatibility_tax"]
    out["kv_cache_norm"] = kv_norm
    out["state_memory_norm"] = state_norm
    return out


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: Sequence[str], standardize: bool = False) -> Tuple[pd.DataFrame, float, int]:
    cols = [y_col] + list(x_cols)
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(work)
    if n < max(10, len(x_cols) + 3):
        return pd.DataFrame(), np.nan, n

    if standardize:
        for col in cols:
            work[col] = safe_zscore(work[col])

    x = work[list(x_cols)].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)

    x_mat = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)

    y_hat = x_mat @ beta
    resid = y - y_hat
    dof = max(n - x_mat.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)

    cov = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(cov))
    t_vals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - (ss_res / (ss_tot + EPS))

    coef = pd.DataFrame(
        {
            "term": ["intercept"] + list(x_cols),
            "coef": beta,
            "std_err": se,
            "t_value": t_vals,
            "n": n,
            "r2": r2,
            "standardized": standardize,
        }
    )
    return coef, r2, n


def get_term_value(coef_df: pd.DataFrame, term: str, column: str) -> float:
    if coef_df.empty or term not in set(coef_df["term"]):
        return np.nan
    row = coef_df[coef_df["term"] == term].iloc[0]
    return float(row[column])


def ols_coef_vector(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    x_mat = np.column_stack([np.ones(len(yv)), xv])
    beta, *_ = np.linalg.lstsq(x_mat, yv, rcond=None)
    return beta


def mediation_analysis(
    df: pd.DataFrame,
    x_col: str = "hybridization_score",
    m_col: str = "compatibility_tax",
    y_col: str = "adoption_residual",
    controls: Optional[Sequence[str]] = None,
    bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    controls = list(controls or [])
    cols = [x_col, m_col, y_col] + controls
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(work)

    summary_base = {
        "x": x_col,
        "m": m_col,
        "y": y_col,
        "controls": "|".join(controls),
        "n": n,
        "a_path": np.nan,
        "b_path": np.nan,
        "c_total": np.nan,
        "c_direct": np.nan,
        "indirect_effect": np.nan,
        "indirect_ci_low": np.nan,
        "indirect_ci_high": np.nan,
        "direct_effect": np.nan,
        "total_effect": np.nan,
        "proportion_mediated": np.nan,
        "bootstrap_p_two_sided": np.nan,
        "bootstrap_samples": 0,
    }

    if n < max(14, len(controls) + 6):
        return pd.DataFrame([summary_base]), pd.DataFrame()

    x = work[x_col].to_numpy(dtype=float)
    m = work[m_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    c = work[controls].to_numpy(dtype=float) if controls else np.empty((n, 0), dtype=float)

    x_c = np.column_stack([x, c]) if c.size else x.reshape(-1, 1)
    y_xm_c = np.column_stack([x, m, c]) if c.size else np.column_stack([x, m])

    beta_a = ols_coef_vector(m, x_c)
    a_path = float(beta_a[1])

    beta_b = ols_coef_vector(y, y_xm_c)
    c_direct = float(beta_b[1])
    b_path = float(beta_b[2])

    beta_c = ols_coef_vector(y, x_c)
    c_total = float(beta_c[1])

    indirect = a_path * b_path
    direct = c_direct
    total = c_total
    proportion = indirect / total if abs(total) > EPS else np.nan

    rng = np.random.default_rng(seed)
    indirect_samples: List[float] = []
    direct_samples: List[float] = []
    total_samples: List[float] = []

    for _ in range(max(0, bootstrap)):
        idx = rng.integers(0, n, n)

        xb = x[idx]
        mb = m[idx]
        yb = y[idx]
        cb = c[idx, :] if c.size else np.empty((n, 0), dtype=float)

        xb_c = np.column_stack([xb, cb]) if cb.size else xb.reshape(-1, 1)
        yb_xm_c = np.column_stack([xb, mb, cb]) if cb.size else np.column_stack([xb, mb])

        ba = ols_coef_vector(mb, xb_c)
        bb = ols_coef_vector(yb, yb_xm_c)
        bc = ols_coef_vector(yb, xb_c)

        ai = float(ba[1])
        bi = float(bb[2])
        ci_direct = float(bb[1])
        ci_total = float(bc[1])

        indirect_samples.append(ai * bi)
        direct_samples.append(ci_direct)
        total_samples.append(ci_total)

    boot_df = pd.DataFrame(
        {
            "indirect_effect": indirect_samples,
            "direct_effect": direct_samples,
            "total_effect": total_samples,
        }
    )

    if not boot_df.empty:
        ci_low, ci_high = np.percentile(boot_df["indirect_effect"], [2.5, 97.5])
        p_two = 2.0 * min(
            float((boot_df["indirect_effect"] <= 0).mean()),
            float((boot_df["indirect_effect"] >= 0).mean()),
        )
        p_two = float(np.clip(p_two, 0.0, 1.0))
    else:
        ci_low, ci_high, p_two = np.nan, np.nan, np.nan

    summary = summary_base.copy()
    summary.update(
        {
            "a_path": a_path,
            "b_path": b_path,
            "c_total": c_total,
            "c_direct": c_direct,
            "indirect_effect": indirect,
            "indirect_ci_low": float(ci_low),
            "indirect_ci_high": float(ci_high),
            "direct_effect": direct,
            "total_effect": total,
            "proportion_mediated": proportion,
            "bootstrap_p_two_sided": p_two,
            "bootstrap_samples": len(boot_df),
        }
    )

    return pd.DataFrame([summary]), boot_df


def build_stage_definitions(df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    stages: List[Tuple[str, List[str]]] = [
        ("stage1_hybrid_only", ["hybridization_score"]),
        ("stage2_plus_compatibility_tax", ["hybridization_score", "compatibility_tax"]),
        (
            "stage3_plus_efficiency_proxies",
            [
                "hybridization_score",
                "compatibility_tax",
                "log1p_kv_cache_mb_per_1k_tokens",
                "log1p_state_memory_proxy",
                "transition_complexity",
            ],
        ),
        (
            "stage4_plus_quality_controls",
            [
                "hybridization_score",
                "compatibility_tax",
                "log1p_kv_cache_mb_per_1k_tokens",
                "log1p_state_memory_proxy",
                "transition_complexity",
                "architecture_quality_index",
                "convergence_score",
                "has_quant_f",
                "is_long_context_128k_f",
            ],
        ),
    ]

    if "benchmark_score" in df.columns and df["benchmark_score"].notna().sum() >= 8:
        stages.append(("stage5_plus_benchmark", stages[-1][1] + ["benchmark_score"]))

    return stages


def staged_ols_analysis(df: pd.DataFrame, y_col: str = "adoption_residual") -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, List[str]]]]:
    stage_defs = build_stage_definitions(df)

    summary_rows: List[Dict[str, Any]] = []
    coef_frames: List[pd.DataFrame] = []

    for stage_name, x_cols in stage_defs:
        coef_unstd, r2_unstd, n_unstd = fit_ols(df, y_col=y_col, x_cols=x_cols, standardize=False)
        coef_std, _, _ = fit_ols(df, y_col=y_col, x_cols=x_cols, standardize=True)

        if not coef_unstd.empty:
            tmp = coef_unstd.copy()
            tmp["stage"] = stage_name
            tmp["scale"] = "unstandardized"
            coef_frames.append(tmp)

        if not coef_std.empty:
            tmp = coef_std.copy()
            tmp["stage"] = stage_name
            tmp["scale"] = "standardized"
            coef_frames.append(tmp)

        summary_rows.append(
            {
                "stage": stage_name,
                "predictors": "|".join(x_cols),
                "n": n_unstd,
                "r2": r2_unstd,
                "hybrid_coef": get_term_value(coef_unstd, "hybridization_score", "coef"),
                "hybrid_se": get_term_value(coef_unstd, "hybridization_score", "std_err"),
                "hybrid_t": get_term_value(coef_unstd, "hybridization_score", "t_value"),
                "hybrid_beta_std": get_term_value(coef_std, "hybridization_score", "coef"),
                "hybrid_t_std": get_term_value(coef_std, "hybridization_score", "t_value"),
                "compatibility_tax_coef": get_term_value(coef_unstd, "compatibility_tax", "coef"),
                "transition_complexity_coef": get_term_value(coef_unstd, "transition_complexity", "coef"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        base_hybrid = summary_df["hybrid_coef"].iloc[0]
        base_hybrid_std = summary_df["hybrid_beta_std"].iloc[0]
        summary_df["hybrid_coef_delta_vs_stage1"] = summary_df["hybrid_coef"] - base_hybrid
        summary_df["hybrid_beta_std_delta_vs_stage1"] = summary_df["hybrid_beta_std"] - base_hybrid_std

    coef_df = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    return summary_df, coef_df, stage_defs


def ols_beta_for_x(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    if controls.size:
        design = np.column_stack([x, controls])
    else:
        design = x.reshape(-1, 1)
    beta = ols_coef_vector(y, design)
    return float(beta[1])


def permutation_test_hybrid_effect(
    df: pd.DataFrame,
    controls: Sequence[str],
    y_col: str = "adoption_residual",
    x_col: str = "hybridization_score",
    n_permutations: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [y_col, x_col] + list(controls)
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(work)

    if n < max(12, len(controls) + 5):
        summary = pd.DataFrame(
            [
                {
                    "n": n,
                    "controls": "|".join(controls),
                    "observed_beta": np.nan,
                    "perm_mean": np.nan,
                    "perm_std": np.nan,
                    "z_vs_null": np.nan,
                    "p_value_two_sided": np.nan,
                    "n_permutations": 0,
                }
            ]
        )
        return summary, pd.DataFrame()

    y = work[y_col].to_numpy(dtype=float)
    x = work[x_col].to_numpy(dtype=float)
    c = work[list(controls)].to_numpy(dtype=float) if controls else np.empty((n, 0), dtype=float)

    observed_beta = ols_beta_for_x(y, x, c)

    rng = np.random.default_rng(seed)
    perm_betas = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_betas[i] = ols_beta_for_x(y, rng.permutation(x), c)

    p_two = (np.sum(np.abs(perm_betas) >= abs(observed_beta)) + 1.0) / (len(perm_betas) + 1.0)
    z_null = (observed_beta - perm_betas.mean()) / (perm_betas.std(ddof=0) + EPS)

    summary = pd.DataFrame(
        [
            {
                "n": n,
                "controls": "|".join(controls),
                "observed_beta": observed_beta,
                "perm_mean": float(perm_betas.mean()),
                "perm_std": float(perm_betas.std(ddof=0)),
                "z_vs_null": float(z_null),
                "p_value_two_sided": float(p_two),
                "n_permutations": int(len(perm_betas)),
            }
        ]
    )

    dist = pd.DataFrame({"beta_perm": perm_betas})
    dist["abs_ge_observed"] = (dist["beta_perm"].abs() >= abs(observed_beta)).astype(int)
    return summary, dist


def choose_benchmark_key(columns: Sequence[str]) -> Optional[str]:
    preferred = ["model_id", "repo_id", "hf_model", "model", "model_name", "name", "id", "checkpoint"]
    for col in preferred:
        if col in columns:
            return col
    return None


def merge_optional_benchmarks(
    df: pd.DataFrame,
    benchmark_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not benchmark_path.exists():
        return df, pd.DataFrame(), pd.DataFrame()

    try:
        bench_raw = pd.read_csv(benchmark_path)
    except Exception as exc:
        report = pd.DataFrame(
            [
                {
                    "status": "read_error",
                    "benchmark_path": str(benchmark_path),
                    "message": str(exc),
                    "rows_in_file": 0,
                    "match_rate": 0.0,
                }
            ]
        )
        return df, report, pd.DataFrame()

    if bench_raw.empty:
        report = pd.DataFrame(
            [
                {
                    "status": "empty_file",
                    "benchmark_path": str(benchmark_path),
                    "message": "",
                    "rows_in_file": 0,
                    "match_rate": 0.0,
                }
            ]
        )
        return df, report, pd.DataFrame()

    bench = bench_raw.copy()
    bench.columns = [str(c).strip() for c in bench.columns]
    key_col = choose_benchmark_key(list(bench.columns))

    if key_col is None:
        report = pd.DataFrame(
            [
                {
                    "status": "no_join_key",
                    "benchmark_path": str(benchmark_path),
                    "message": "No key column found",
                    "rows_in_file": len(bench),
                    "match_rate": 0.0,
                }
            ]
        )
        return df, report, pd.DataFrame()

    value_cols = [c for c in bench.columns if c != key_col]
    rename_map = {c: f"bench_{c}" for c in value_cols}
    bench = bench.rename(columns=rename_map)

    if key_col == "model_id":
        bench["model_id"] = bench["model_id"].astype(str).str.strip()
        bench = bench.drop_duplicates("model_id", keep="first")
        merged = df.merge(bench, on="model_id", how="left")
        strategy = "model_id_exact"
    else:
        bench["_join_short"] = bench[key_col].astype(str).map(short_name).str.lower()
        bench = bench.drop_duplicates("_join_short", keep="first")
        bench = bench.drop(columns=[key_col])
        merged = df.copy()
        merged["_join_short"] = merged["model_id"].astype(str).map(short_name).str.lower()
        merged = merged.merge(bench, on="_join_short", how="left")
        strategy = f"short_name_from_{key_col}"

    bench_cols = [c for c in merged.columns if c.startswith("bench_")]
    if not bench_cols:
        report = pd.DataFrame(
            [
                {
                    "status": "no_value_columns",
                    "benchmark_path": str(benchmark_path),
                    "message": "",
                    "rows_in_file": len(bench_raw),
                    "match_rate": 0.0,
                    "strategy": strategy,
                }
            ]
        )
        if "_join_short" in merged.columns:
            merged = merged.drop(columns=["_join_short"])
        return merged, report, pd.DataFrame()

    matched = merged[bench_cols].notna().any(axis=1)
    match_rate = float(matched.mean()) if len(matched) else 0.0

    used_rows: List[Dict[str, Any]] = []
    score_inputs: Dict[str, pd.Series] = {}

    for col in bench_cols:
        numeric = pd.to_numeric(merged[col], errors="coerce")
        if numeric.notna().sum() < 6:
            continue
        direction = -1.0 if any(tok in col.lower() for tok in LOWER_IS_BETTER_HINTS) else 1.0
        score_inputs[col] = safe_zscore(numeric * direction)
        used_rows.append(
            {
                "benchmark_column": col,
                "direction": "lower_is_better_inverted" if direction < 0 else "higher_is_better",
                "non_null_n": int(numeric.notna().sum()),
            }
        )

    if score_inputs:
        score_df = pd.DataFrame(score_inputs, index=merged.index)
        merged["benchmark_score"] = score_df.mean(axis=1)
        merged["has_benchmark"] = score_df.notna().any(axis=1).astype(float)

    if "_join_short" in merged.columns:
        merged = merged.drop(columns=["_join_short"])

    report = pd.DataFrame(
        [
            {
                "status": "merged",
                "benchmark_path": str(benchmark_path),
                "message": "",
                "rows_in_file": len(bench_raw),
                "match_rate": match_rate,
                "strategy": strategy,
                "numeric_columns_used_for_score": len(used_rows),
            }
        ]
    )
    used_df = pd.DataFrame(used_rows)
    return merged, report, used_df


def summarize_layer_dna(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "raw_config_present",
        "attn_layer_ratio",
        "transition_complexity",
        "attn_periodicity",
        "kv_cache_mb_per_1k_tokens",
        "state_memory_proxy",
    ]
    work = df.copy()
    for col in cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    summary = (
        work.groupby("inferred_paradigm", dropna=False)
        .agg(
            model_count=("model_id", "count"),
            raw_config_coverage=("raw_config_present", "mean"),
            mean_attn_layer_ratio=("attn_layer_ratio", "mean"),
            mean_transition_complexity=("transition_complexity", "mean"),
            mean_attn_periodicity=("attn_periodicity", "mean"),
            mean_kv_cache_mb_per_1k_tokens=("kv_cache_mb_per_1k_tokens", "mean"),
            mean_state_memory_proxy=("state_memory_proxy", "mean"),
        )
        .reset_index()
        .sort_values("model_count", ascending=False)
    )
    return summary


def summarize_compatibility_tax(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("inferred_paradigm", dropna=False)
        .agg(
            model_count=("model_id", "count"),
            mean_framework_support=("framework_support_index", "mean"),
            mean_compatibility_tax=("compatibility_tax", "mean"),
            median_compatibility_tax=("compatibility_tax", "median"),
            mean_hybridization=("hybridization_score", "mean"),
            mean_transition_complexity=("transition_complexity", "mean"),
            mean_adoption_residual=("adoption_residual", "mean"),
        )
        .reset_index()
        .sort_values("mean_compatibility_tax", ascending=False)
    )
    return summary


def write_key_findings(
    df: pd.DataFrame,
    mediation_summary: pd.DataFrame,
    staged_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    benchmark_report: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []

    total = len(df)
    raw_cfg = int(df["raw_config_present"].fillna(False).sum()) if "raw_config_present" in df else 0

    lines.append(f"models_total: {total}")
    lines.append(f"raw_config_available: {raw_cfg} ({(raw_cfg / total) if total else 0:.1%})")

    if "compatibility_tax" in df.columns:
        hybrid_tax = df.loc[df["inferred_paradigm"] == "hybrid", "compatibility_tax"].mean()
        tr_tax = df.loc[df["inferred_paradigm"] == "transformer", "compatibility_tax"].mean()
        lines.append(f"compatibility_tax_mean_hybrid: {hybrid_tax:.4f}")
        lines.append(f"compatibility_tax_mean_transformer: {tr_tax:.4f}")

    if not mediation_summary.empty:
        m = mediation_summary.iloc[0]
        lines.append(f"mediation_indirect_effect: {m['indirect_effect']:+.4f}")
        lines.append(f"mediation_indirect_ci95: [{m['indirect_ci_low']:+.4f}, {m['indirect_ci_high']:+.4f}]")
        lines.append(f"mediation_direct_effect: {m['direct_effect']:+.4f}")
        lines.append(f"mediation_total_effect: {m['total_effect']:+.4f}")
        lines.append(f"mediation_bootstrap_p: {m['bootstrap_p_two_sided']:.4f}")

    if not staged_summary.empty:
        last = staged_summary.iloc[-1]
        first = staged_summary.iloc[0]
        lines.append(f"hybrid_coef_stage1: {first['hybrid_coef']:+.4f}")
        lines.append(f"hybrid_coef_final: {last['hybrid_coef']:+.4f}")
        lines.append(f"hybrid_coef_shift: {last['hybrid_coef_delta_vs_stage1']:+.4f}")
        lines.append(f"final_stage_r2: {last['r2']:.4f}")

    if not permutation_summary.empty:
        p = permutation_summary.iloc[0]
        lines.append(f"permutation_observed_beta: {p['observed_beta']:+.4f}")
        lines.append(f"permutation_p_two_sided: {p['p_value_two_sided']:.4f}")
        lines.append(f"permutation_n: {int(p['n'])}")

    if not benchmark_report.empty:
        b = benchmark_report.iloc[0]
        lines.append(f"benchmark_merge_status: {b.get('status', '')}")
        lines.append(f"benchmark_match_rate: {float(b.get('match_rate', np.nan)):.1%}")

    ranked = (
        df.dropna(subset=["hybridization_score", "compatibility_tax"])
        .sort_values(["hybridization_score", "compatibility_tax"], ascending=[False, False])
        .head(3)
    )
    if not ranked.empty:
        top = "; ".join(
            f"{short_name(row['model_id'])}:{row['compatibility_tax']:.3f}"
            for _, row in ranked.iterrows()
        )
        lines.append(f"top_hybridization_with_tax: {top}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism analysis for why hybridization hurts adoption.")
    parser.add_argument("--input", default="metadata_enriched.json", help="Path to metadata_enriched.json")
    parser.add_argument("--outdir", default="analysis/ecosystem_hybrid_mechanism", help="Output directory")
    parser.add_argument("--benchmarks", default="benchmarks.csv", help="Optional benchmark CSV path")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations for mediation CI")
    parser.add_argument("--permutations", type=int, default=5000, help="Permutation iterations for robustness test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    out_tables = outdir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    meta, df, adoption_baseline = load_mechanism_dataframe(input_path)
    df = add_efficiency_proxies(df)
    df = add_compatibility_tax_indices(df)

    benchmark_path = Path(args.benchmarks)
    df, benchmark_report, benchmark_used = merge_optional_benchmarks(df, benchmark_path)

    mediation_controls = [
        "log1p_kv_cache_mb_per_1k_tokens",
        "log1p_state_memory_proxy",
        "transition_complexity",
        "architecture_quality_index",
        "convergence_score",
        "has_quant_f",
        "is_long_context_128k_f",
    ]
    if "benchmark_score" in df.columns and df["benchmark_score"].notna().sum() >= 8:
        mediation_controls.append("benchmark_score")

    mediation_summary, mediation_boot = mediation_analysis(
        df,
        x_col="hybridization_score",
        m_col="compatibility_tax",
        y_col="adoption_residual",
        controls=mediation_controls,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    staged_summary, staged_coef, stage_defs = staged_ols_analysis(df, y_col="adoption_residual")
    final_controls = [c for c in stage_defs[-1][1] if c != "hybridization_score"] if stage_defs else []
    permutation_summary, permutation_dist = permutation_test_hybrid_effect(
        df,
        controls=final_controls,
        y_col="adoption_residual",
        x_col="hybridization_score",
        n_permutations=args.permutations,
        seed=args.seed,
    )

    layer_summary = summarize_layer_dna(df)
    tax_summary = summarize_compatibility_tax(df)

    df.to_csv(out_tables / "model_hybrid_mechanism_metrics.csv", index=False)
    adoption_baseline.to_csv(out_tables / "adoption_baseline_unstandardized.csv", index=False)
    layer_summary.to_csv(out_tables / "layer_dna_summary.csv", index=False)
    tax_summary.to_csv(out_tables / "compatibility_tax_summary.csv", index=False)
    mediation_summary.to_csv(out_tables / "mediation_summary.csv", index=False)
    mediation_boot.to_csv(out_tables / "mediation_bootstrap_distribution.csv", index=False)
    staged_summary.to_csv(out_tables / "staged_ols_summary.csv", index=False)
    staged_coef.to_csv(out_tables / "staged_ols_coefficients.csv", index=False)
    permutation_summary.to_csv(out_tables / "permutation_test_summary.csv", index=False)
    permutation_dist.to_csv(out_tables / "permutation_test_distribution.csv", index=False)

    layer_model_cols = [
        "model_id",
        "inferred_paradigm",
        "raw_config_present",
        "raw_config_source",
        "layer_dna_source",
        "layer_total_estimate",
        "attn_layer_count",
        "ssm_layer_count",
        "attn_layer_ratio",
        "attn_period",
        "attn_periodicity",
        "layer_transition_rate",
        "transition_complexity",
        "kv_cache_mb_per_1k_tokens",
        "state_memory_proxy",
        "framework_support_index",
        "compatibility_tax",
        "hybridization_score",
        "adoption_residual",
    ]
    layer_model_cols = [c for c in layer_model_cols if c in df.columns]
    df[layer_model_cols].to_csv(out_tables / "layer_dna_model_level.csv", index=False)

    if not benchmark_report.empty:
        benchmark_report.to_csv(out_tables / "benchmark_merge_report.csv", index=False)
    if not benchmark_used.empty:
        benchmark_used.to_csv(out_tables / "benchmark_columns_used.csv", index=False)

    write_key_findings(
        df=df,
        mediation_summary=mediation_summary,
        staged_summary=staged_summary,
        permutation_summary=permutation_summary,
        benchmark_report=benchmark_report,
        out_path=outdir / "key_findings.txt",
    )

    print(f"Analyzed {len(df)} models")
    print(f"Output tables: {out_tables}")
    print(f"Key findings: {outdir / 'key_findings.txt'}")
    if benchmark_path.exists():
        print(f"Benchmark merge attempted: {benchmark_path}")
    if isinstance(meta, dict):
        print(f"source_file={meta.get('source_file')}, generated_at_utc={meta.get('generated_at_utc')}")


if __name__ == "__main__":
    main()
