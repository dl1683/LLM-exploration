#!/usr/bin/env python3
"""
metadata_ingest.py

Step 2 metadata enrichment:
- Reads canonical_registry.json
- Fetches model metadata from Hugging Face Hub
- Parses config.json, tokenizer_config.json, README model card, safetensors index
- Writes metadata_enriched.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping as ABCMapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

try:
    from huggingface_hub.utils import EntryNotFoundError
except Exception:
    EntryNotFoundError = HfHubHTTPError  # type: ignore[assignment]


MAX_REASONABLE_CONTEXT = 10_000_000

HUMAN_MULTIPLIERS = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "t": 1_000_000_000_000,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

HUMAN_NUMBER_RE = re.compile(
    r"^\s*(?P<num>\d+(?:[.,]\d+)?)\s*(?P<suffix>k|m|b|t|thousand|million|billion|trillion)?\s*$",
    re.IGNORECASE,
)

TOKEN_AFTER_NUMBER_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<suffix>k|m|b|t|thousand|million|billion|trillion)?\s*(?:\+|plus)?\s*tokens?\b",
    re.IGNORECASE,
)

TOKEN_AFTER_LABEL_RE = re.compile(
    r"(?:pretrain(?:ing)?|training|train(?:ed)?(?: on)?|corpus|dataset)\s*"
    r"(?:token(?:s)?|count)?\s*[:=]?\s*(?P<num>\d+(?:[.,]\d+)?)\s*"
    r"(?P<suffix>k|m|b|t|thousand|million|billion|trillion)?\b",
    re.IGNORECASE,
)

TOKEN_METADATA_EXCLUDES = (
    "tokenizer",
    "special_token",
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "mask_token",
    "max_token",
    "model_max_length",
    "max_position",
    "context",
    "window",
)

RELEASE_DATE_KEYS = (
    "release_date",
    "released",
    "date",
    "published",
    "model_release_date",
    "created_at",
)

LICENSE_KEYS = (
    "license",
    "license_name",
    "spdx",
    "spdx_id",
)

NUM_LAYERS_KEYS = (
    "num_hidden_layers",
    "n_layer",
    "n_layers",
    "num_layers",
    "decoder_layers",
    "num_decoder_layers",
    "transformer_layers",
    "num_transformer_layers",
)

HIDDEN_SIZE_KEYS = (
    "hidden_size",
    "n_embd",
    "d_model",
    "dim",
    "model_dim",
    "embed_dim",
)

FFN_SIZE_KEYS = (
    "intermediate_size",
    "ffn_dim",
    "ffn_hidden_size",
    "n_inner",
    "d_ff",
    "mlp_dim",
    "feed_forward_proj_size",
)

NUM_HEADS_KEYS = (
    "num_attention_heads",
    "n_head",
    "n_heads",
    "num_heads",
    "attention_heads",
    "n_attn_heads",
)

NUM_KV_HEADS_KEYS = (
    "num_key_value_heads",
    "n_kv_head",
    "n_kv_heads",
    "num_kv_heads",
    "multi_query_group_num",
    "num_key_value_groups",
)

HEAD_DIM_KEYS = (
    "head_dim",
    "attention_head_dim",
    "kv_channels",
)

NORM_KEYS = (
    "norm_type",
    "normalization_type",
    "normalization",
    "norm",
)

ACTIVATION_KEYS = (
    "hidden_act",
    "activation_function",
    "act_fn",
    "activation",
)

MAX_CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "max_seq_length",
    "max_sequence_length",
    "seq_len",
    "seq_length",
    "context_length",
    "max_context_length",
    "model_max_length",
)

SLIDING_WINDOW_KEYS = (
    "sliding_window",
    "sliding_window_size",
    "attention_window",
    "window_size",
)

VOCAB_SIZE_KEYS = (
    "vocab_size",
    "padded_vocab_size",
)

NUM_PARAMS_CONFIG_KEYS = (
    "num_parameters",
    "n_parameters",
    "parameter_count",
    "total_params",
    "n_params",
)

TOKENIZER_TYPE_KEYS = (
    "tokenizer_class",
    "tokenizer_type",
    "tokenizer_model_type",
    "model_type",
)

SSM_STATE_KEYS = (
    "state_size",
    "ssm_state_size",
    "d_state",
    "ssm_d_state",
    "mamba_d_state",
)

SSM_CONV_KEYS = (
    "conv_kernel",
    "ssm_conv_kernel",
    "d_conv",
    "conv_kernel_size",
    "mamba_d_conv",
)

SSM_DT_RANK_KEYS = (
    "dt_rank",
    "ssm_dt_rank",
    "mamba_dt_rank",
    "time_step_rank",
)

SSM_LAYER_KEYS = (
    "num_ssm_layers",
    "ssm_layers",
    "n_ssm_layers",
    "mamba_layers",
    "num_mamba_layers",
)

ATTENTION_LAYER_KEYS = (
    "num_attention_layers",
    "attention_layers",
    "n_attn_layers",
    "num_transformer_layers",
    "transformer_layers",
)

QUANTIZATION_ORDER = [
    "GGUF",
    "GPTQ",
    "AWQ",
    "EXL2",
    "AQLM",
    "HQQ",
    "EETQ",
    "BITSANDBYTES",
    "FP8",
    "INT8",
    "INT4",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich canonical_registry.json with Hugging Face metadata and write "
            "metadata_enriched.json."
        )
    )
    parser.add_argument(
        "--registry-path",
        default="canonical_registry.json",
        help="Path to canonical registry JSON (default: canonical_registry.json).",
    )
    parser.add_argument(
        "--output-path",
        default="metadata_enriched.json",
        help="Output JSON path (default: metadata_enriched.json).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF revision to use for repo files (default: main).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel worker count (default: 4).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for transient HF errors (default: 3).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Base retry backoff in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--hf-token",
        default=(
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        ),
        help="HF token (defaults to HF_TOKEN/HUGGINGFACE_TOKEN env vars).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-model progress logs.",
    )

    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.retries < 1:
        parser.error("--retries must be >= 1")
    if args.retry_backoff_seconds < 0:
        parser.error("--retry-backoff-seconds must be >= 0")
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def split_model_id(model_id: str) -> Tuple[str, str]:
    if "/" in model_id:
        org, name = model_id.split("/", 1)
        return org, name
    return "unknown", model_id


def to_lower_key_map(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in data.items()}


def coerce_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        out = value.strip()
        return out or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        for item in value:
            as_str = coerce_str(item)
            if as_str:
                return as_str
    return None


def parse_human_number(text: str) -> Optional[int]:
    match = HUMAN_NUMBER_RE.match(text.strip())
    if not match:
        return None
    base_text = match.group("num").replace(",", "")
    suffix = (match.group("suffix") or "").lower()
    try:
        base = float(base_text)
    except ValueError:
        return None
    multiplier = HUMAN_MULTIPLIERS.get(suffix, 1)
    return int(base * multiplier)


def numeric_values(value: Any) -> List[float]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return []
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            try:
                return [float(stripped)]
            except ValueError:
                return []
        human = parse_human_number(value)
        if human is not None:
            return [float(human)]
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[float] = []
        for item in value:
            out.extend(numeric_values(item))
        return out
    return []


def coerce_int(value: Any, strategy: str = "first") -> Optional[int]:
    values = numeric_values(value)
    if not values:
        return None
    chosen = max(values) if strategy == "max" else values[0]
    if chosen < 0:
        return None
    return int(chosen)


def coerce_float(value: Any, strategy: str = "first") -> Optional[float]:
    values = numeric_values(value)
    if not values:
        return None
    return max(values) if strategy == "max" else values[0]


def pick_int(data_lc: Mapping[str, Any], keys: Sequence[str], strategy: str = "first") -> Optional[int]:
    for key in keys:
        if key in data_lc:
            val = coerce_int(data_lc[key], strategy=strategy)
            if val is not None:
                return val
    return None


def pick_str(data_lc: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in data_lc:
            val = coerce_str(data_lc[key])
            if val:
                return val
    return None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def valid_context_length(value: Optional[int]) -> bool:
    return value is not None and 4 <= value <= MAX_REASONABLE_CONTEXT


def normalize_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        normalized = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date().isoformat()
        except ValueError:
            match = re.search(r"\d{4}-\d{2}-\d{2}", raw)
            if match:
                return match.group(0)
    return None


def get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, ABCMapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def mapping_from_obj(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, ABCMapping):
        return {str(k): v for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        try:
            data = obj.to_dict()
            if isinstance(data, ABCMapping):
                return {str(k): v for k, v in data.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {str(k): v for k, v in vars(obj).items() if not str(k).startswith("_")}
    return {}


def retry_call(fn: Callable[[], Any], retries: int, backoff_seconds: float) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except (RepositoryNotFoundError, RevisionNotFoundError):
            raise
        except HfHubHTTPError as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status not in (408, 429):
                raise
        except Exception as exc:
            last_exc = exc

        if attempt < retries and backoff_seconds > 0:
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_call exhausted without producing a result")


def load_registry_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Registry JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    rows: Any
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, ABCMapping):
        if isinstance(payload.get("canonical_table"), list):
            rows = payload["canonical_table"]
        elif isinstance(payload.get("models"), list):
            rows = payload["models"]
        else:
            raise RuntimeError(
                f"{path} must contain either a top-level list, 'canonical_table', or 'models'."
            )
    else:
        raise RuntimeError(f"Unsupported registry format in {path}.")

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in rows:
        if not isinstance(row, ABCMapping):
            continue
        model_id = row.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        if model_id in seen:
            continue
        seen.add(model_id)
        out.append(dict(row))

    if not out:
        raise RuntimeError(f"No valid model rows found in {path}.")
    return out


def fetch_model_info(
    api: HfApi,
    model_id: str,
    revision: str,
    token: Optional[str],
    retries: int,
    backoff_seconds: float,
) -> Any:
    return retry_call(
        lambda: api.model_info(
            model_id,
            revision=revision,
            files_metadata=True,
            token=token,
        ),
        retries=retries,
        backoff_seconds=backoff_seconds,
    )


def get_repo_files_and_sizes(
    api: HfApi,
    model_id: str,
    model_info: Any,
    revision: str,
    token: Optional[str],
    retries: int,
    backoff_seconds: float,
) -> Tuple[Set[str], Dict[str, int]]:
    files: Set[str] = set()
    sizes: Dict[str, int] = {}

    siblings = get_attr_or_key(model_info, "siblings", []) or []
    for sibling in siblings:
        filename = get_attr_or_key(sibling, "rfilename")
        if filename is None and isinstance(sibling, str):
            filename = sibling
        if not isinstance(filename, str):
            continue
        files.add(filename)
        size_value = coerce_int(get_attr_or_key(sibling, "size"))
        if size_value is not None:
            sizes[filename] = size_value

    if files:
        return files, sizes

    try:
        listed = retry_call(
            lambda: api.list_repo_files(model_id, revision=revision, token=token),
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        for fname in listed or []:
            if isinstance(fname, str):
                files.add(fname)
    except Exception:
        pass

    return files, sizes


def should_attempt_file(filename: str, repo_files: Set[str]) -> bool:
    return not repo_files or filename in repo_files


def download_json_file(
    model_id: str,
    filename: str,
    revision: str,
    token: Optional[str],
    retries: int,
    backoff_seconds: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        local_path = retry_call(
            lambda: hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                repo_type="model",
                token=token,
            ),
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    except EntryNotFoundError:
        return None, "missing"
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"download_error: {exc}"

    try:
        raw = Path(local_path).read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"read_error: {exc}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_error: {exc}"

    if isinstance(data, ABCMapping):
        return {str(k): v for k, v in data.items()}, None
    return {"_value": data}, None


def download_text_file(
    model_id: str,
    filename: str,
    revision: str,
    token: Optional[str],
    retries: int,
    backoff_seconds: float,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        local_path = retry_call(
            lambda: hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                repo_type="model",
                token=token,
            ),
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    except EntryNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"download_error: {exc}"

    try:
        return Path(local_path).read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"read_error: {exc}"


def find_readme_filename(repo_files: Set[str]) -> Optional[str]:
    if not repo_files:
        return "README.md"

    for candidate in ("README.md", "readme.md", "README.MD"):
        if candidate in repo_files:
            return candidate

    for fname in sorted(repo_files):
        if fname.lower().endswith("readme.md"):
            return fname
    return None


def choose_safetensors_index(repo_files: Set[str]) -> Optional[str]:
    if not repo_files:
        return "model.safetensors.index.json"

    candidates = sorted(
        [f for f in repo_files if f.endswith(".safetensors.index.json")],
        key=lambda x: (x.count("/"), len(x), x),
    )
    if not candidates:
        return None
    if "model.safetensors.index.json" in candidates:
        return "model.safetensors.index.json"
    return candidates[0]


def parse_simple_front_matter(front_matter: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    current_list_key: Optional[str] = None

    for raw_line in front_matter.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if current_list_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip("'\"")
            out.setdefault(current_list_key, []).append(value)
            continue

        kv_match = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*(.*)$", stripped)
        if not kv_match:
            current_list_key = None
            continue

        key = kv_match.group(1)
        value = kv_match.group(2).strip()

        if value == "":
            out.setdefault(key, [])
            current_list_key = key
            continue

        current_list_key = None
        if value.startswith("[") and value.endswith("]"):
            items = [x.strip().strip("'\"") for x in value[1:-1].split(",") if x.strip()]
            out[key] = items
        else:
            out[key] = value.strip("'\"")

    return out


def parse_model_card_front_matter(card_text: Optional[str]) -> Dict[str, Any]:
    if not card_text:
        return {}

    clean_text = card_text.lstrip("\ufeff")
    lines = clean_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end_idx = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}

    front_matter = "\n".join(lines[1:end_idx])

    yaml_module = None
    try:
        import yaml as yaml_module  # type: ignore
    except Exception:
        yaml_module = None

    if yaml_module is not None:
        try:
            parsed = yaml_module.safe_load(front_matter)
            if isinstance(parsed, ABCMapping):
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            pass

    return parse_simple_front_matter(front_matter)


def metadata_value(meta: Mapping[str, Any], keys: Sequence[str]) -> Any:
    meta_lc = to_lower_key_map(meta)
    for key in keys:
        if key in meta_lc:
            return meta_lc[key]
    return None


def normalize_tags(tags: Any) -> List[str]:
    out: List[str] = []
    if isinstance(tags, str):
        out.extend([p.strip() for p in tags.split(",") if p.strip()])
    elif isinstance(tags, list):
        for item in tags:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
    return out


def infer_release_date(model_info: Any, card_meta: Mapping[str, Any]) -> Optional[str]:
    from_info = normalize_date(get_attr_or_key(model_info, "created_at"))
    if from_info:
        return from_info
    return normalize_date(metadata_value(card_meta, RELEASE_DATE_KEYS))


def infer_license(
    card_meta: Mapping[str, Any],
    tags: Sequence[str],
    config_lc: Mapping[str, Any],
) -> Optional[str]:
    raw_license = metadata_value(card_meta, LICENSE_KEYS)

    if isinstance(raw_license, list):
        for item in raw_license:
            val = coerce_str(item)
            if val:
                return val
    else:
        val = coerce_str(raw_license)
        if val:
            return val

    for tag in tags:
        lower_tag = tag.lower()
        if lower_tag.startswith("license:"):
            return tag.split(":", 1)[1].strip()

    config_license = pick_str(config_lc, LICENSE_KEYS)
    if config_license:
        return config_license

    return None


def infer_norm_type(config_lc: Mapping[str, Any]) -> Optional[str]:
    explicit = pick_str(config_lc, NORM_KEYS)
    if explicit:
        lowered = explicit.lower()
        if "rms" in lowered:
            return "rmsnorm"
        if "layer" in lowered:
            return "layernorm"
        if "group" in lowered:
            return "groupnorm"
        return lowered

    if "rms_norm_eps" in config_lc or "rmsnorm_eps" in config_lc or truthy(config_lc.get("use_rms_norm")):
        return "rmsnorm"
    if "layer_norm_eps" in config_lc or "layernorm_eps" in config_lc or "final_layer_norm_eps" in config_lc:
        return "layernorm"

    return None


def infer_activation(config_lc: Mapping[str, Any]) -> Optional[str]:
    act = pick_str(config_lc, ACTIVATION_KEYS)
    return act.lower() if act else None


def infer_positional_scheme(config_lc: Mapping[str, Any]) -> Optional[str]:
    explicit = pick_str(
        config_lc,
        ("position_embedding_type", "positional_embedding_type", "position_embeddings"),
    )
    if explicit:
        lowered = explicit.lower()
        if "rope" in lowered or "rotary" in lowered:
            return "rope"
        if "alibi" in lowered:
            return "alibi"
        if "relative" in lowered:
            return "relative"
        if "absolute" in lowered:
            return "absolute"
        return lowered

    if any(
        key in config_lc
        for key in ("rope_scaling", "rope_theta", "rotary_emb_base", "rotary_pct", "partial_rotary_factor")
    ):
        return "rope"
    if truthy(config_lc.get("alibi")) or truthy(config_lc.get("use_alibi")):
        return "alibi"
    if any(
        key in config_lc
        for key in ("relative_attention_num_buckets", "relative_attention_max_distance", "rel_pos_bins")
    ):
        return "relative"
    if truthy(config_lc.get("no_position_embeddings")):
        return "none"

    model_type = pick_str(config_lc, ("model_type",))
    if model_type and model_type.lower() in {"rwkv", "mamba", "mamba2", "xlstm"}:
        return "implicit_recurrence"

    return None


def infer_max_context(config_lc: Mapping[str, Any], tokenizer_lc: Mapping[str, Any]) -> Optional[int]:
    candidates: List[int] = []

    for key in MAX_CONTEXT_KEYS:
        value = coerce_int(config_lc.get(key), strategy="max")
        if valid_context_length(value):
            candidates.append(value)  # type: ignore[arg-type]

    tokenizer_max = coerce_int(tokenizer_lc.get("model_max_length"), strategy="max")
    if valid_context_length(tokenizer_max):
        candidates.append(tokenizer_max)  # type: ignore[arg-type]

    rope_scaling = config_lc.get("rope_scaling")
    if isinstance(rope_scaling, ABCMapping):
        rope_lc = to_lower_key_map({str(k): v for k, v in rope_scaling.items()})
        factor = coerce_float(rope_lc.get("factor"), strategy="max")
        original = coerce_int(rope_lc.get("original_max_position_embeddings")) or coerce_int(
            config_lc.get("max_position_embeddings")
        )
        if factor is not None and original is not None:
            scaled = int(original * factor)
            if valid_context_length(scaled):
                candidates.append(scaled)

        long_factor = coerce_float(rope_lc.get("long_factor"), strategy="max")
        if long_factor is not None and original is not None:
            scaled = int(original * long_factor)
            if valid_context_length(scaled):
                candidates.append(scaled)

    return max(candidates) if candidates else None


def infer_dtype_from_sources(
    config_lc: Mapping[str, Any],
    safetensors_info: Mapping[str, Any],
    safetensors_index: Optional[Mapping[str, Any]],
) -> Optional[str]:
    dtype = pick_str(config_lc, ("torch_dtype", "dtype", "param_dtype"))
    if dtype:
        return dtype

    params = safetensors_info.get("parameters")
    if isinstance(params, ABCMapping):
        numeric_entries: List[Tuple[int, str]] = []
        for dtype_name, count in params.items():
            count_int = coerce_int(count)
            if count_int is not None:
                numeric_entries.append((count_int, str(dtype_name)))
        if numeric_entries:
            numeric_entries.sort(reverse=True)
            return numeric_entries[0][1]

    if safetensors_index and isinstance(safetensors_index.get("metadata"), ABCMapping):
        meta = safetensors_index["metadata"]  # type: ignore[index]
        meta_lc = to_lower_key_map({str(k): v for k, v in meta.items()})
        dtype = pick_str(meta_lc, ("dtype", "weight_dtype", "tensor_dtype"))
        if dtype:
            return dtype

    return None


def dtype_nbytes(dtype: Optional[str]) -> Optional[float]:
    if not dtype:
        return None

    d = dtype.lower().strip()
    canonical = d.replace("_", "").replace("-", "")

    if canonical in {"f16", "fp16", "float16", "half", "bf16", "bfloat16"}:
        return 2.0
    if canonical in {"f32", "fp32", "float32"}:
        return 4.0
    if canonical in {"f64", "fp64", "float64"}:
        return 8.0
    if canonical in {"f8", "fp8", "float8", "e4m3fn", "e5m2"}:
        return 1.0
    if canonical in {"int8", "uint8", "i8", "u8"}:
        return 1.0
    if canonical in {"int4", "uint4", "nf4", "fp4", "i4", "u4"}:
        return 0.5

    if "bfloat16" in canonical or "float16" in canonical:
        return 2.0
    if "float32" in canonical:
        return 4.0
    if "float64" in canonical:
        return 8.0
    if "float8" in canonical or "fp8" in canonical:
        return 1.0
    if "int8" in canonical:
        return 1.0
    if "int4" in canonical:
        return 0.5

    return None


def infer_num_params(
    model_info: Any,
    config_lc: Mapping[str, Any],
    safetensors_index: Optional[Mapping[str, Any]],
    file_sizes: Mapping[str, int],
) -> Tuple[Optional[int], Optional[str]]:
    safetensors_info = mapping_from_obj(get_attr_or_key(model_info, "safetensors"))

    total_from_api = coerce_int(safetensors_info.get("total"))
    if total_from_api is not None and total_from_api > 0:
        return total_from_api, "hf_api.safetensors.total"

    params_obj = safetensors_info.get("parameters")
    if isinstance(params_obj, ABCMapping):
        summed = 0
        found = False
        for value in params_obj.values():
            parsed = coerce_int(value)
            if parsed is not None:
                summed += parsed
                found = True
        if found and summed > 0:
            return summed, "hf_api.safetensors.parameters"

    for key in NUM_PARAMS_CONFIG_KEYS:
        if key in config_lc:
            value = coerce_int(config_lc.get(key), strategy="max")
            if value is not None and value > 0:
                return value, f"config.{key}"

    dtype = infer_dtype_from_sources(config_lc, safetensors_info, safetensors_index)
    bytes_per_param = dtype_nbytes(dtype) or 2.0

    if safetensors_index and isinstance(safetensors_index.get("metadata"), ABCMapping):
        meta_lc = to_lower_key_map(
            {str(k): v for k, v in safetensors_index["metadata"].items()}  # type: ignore[index]
        )
        total_size = coerce_int(meta_lc.get("total_size"))
        if total_size is not None and total_size > 0:
            approx = int(total_size / bytes_per_param)
            if approx > 0:
                return approx, "safetensors_index.metadata.total_size"

    total_safetensors_bytes = 0
    for filename, size in file_sizes.items():
        lower_name = filename.lower()
        if lower_name.endswith(".safetensors") and not lower_name.endswith(".index.json"):
            total_safetensors_bytes += size
    if total_safetensors_bytes > 0:
        approx = int(total_safetensors_bytes / bytes_per_param)
        if approx > 0:
            return approx, "repo_safetensors_file_sizes"

    return None, None


def extract_architecture(
    model_info: Any,
    config_lc: Mapping[str, Any],
    tokenizer_lc: Mapping[str, Any],
    safetensors_index: Optional[Mapping[str, Any]],
    file_sizes: Mapping[str, int],
) -> Dict[str, Any]:
    num_layers = pick_int(config_lc, NUM_LAYERS_KEYS, strategy="max")
    hidden_size = pick_int(config_lc, HIDDEN_SIZE_KEYS, strategy="max")
    ffn_size = pick_int(config_lc, FFN_SIZE_KEYS, strategy="max")
    num_heads = pick_int(config_lc, NUM_HEADS_KEYS, strategy="max")
    num_kv_heads = pick_int(config_lc, NUM_KV_HEADS_KEYS, strategy="max")
    if num_kv_heads is None and truthy(config_lc.get("multi_query")):
        num_kv_heads = 1

    head_dim = pick_int(config_lc, HEAD_DIM_KEYS, strategy="max")
    if head_dim is None and hidden_size and num_heads and num_heads > 0:
        head_dim = hidden_size // num_heads

    max_context = infer_max_context(config_lc, tokenizer_lc)
    sliding_window = pick_int(config_lc, SLIDING_WINDOW_KEYS, strategy="max")
    vocab_size = pick_int(config_lc, VOCAB_SIZE_KEYS, strategy="max")
    if vocab_size is None:
        vocab_size = pick_int(tokenizer_lc, ("vocab_size",), strategy="max")

    num_params, num_params_source = infer_num_params(
        model_info=model_info,
        config_lc=config_lc,
        safetensors_index=safetensors_index,
        file_sizes=file_sizes,
    )

    out: Dict[str, Any] = {
        "num_params": num_params,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "ffn_size": ffn_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "norm_type": infer_norm_type(config_lc),
        "activation": infer_activation(config_lc),
        "positional_scheme": infer_positional_scheme(config_lc),
        "max_context": max_context,
        "sliding_window": sliding_window,
        "vocab_size": vocab_size,
    }
    if num_params_source:
        out["num_params_source"] = num_params_source
    return out


def extract_paradigm_specific(
    config_lc: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> Dict[str, Any]:
    ssm_state_size = pick_int(config_lc, SSM_STATE_KEYS, strategy="max")
    ssm_conv_kernel = pick_int(config_lc, SSM_CONV_KEYS, strategy="max")
    ssm_dt_rank = pick_int(config_lc, SSM_DT_RANK_KEYS, strategy="max")
    num_ssm_layers = pick_int(config_lc, SSM_LAYER_KEYS, strategy="max")
    num_attention_layers = pick_int(config_lc, ATTENTION_LAYER_KEYS, strategy="max")

    num_total_layers = coerce_int(architecture.get("num_layers"))
    if (
        num_attention_layers is None
        and num_total_layers is not None
        and num_ssm_layers is not None
        and num_total_layers >= num_ssm_layers
    ):
        num_attention_layers = num_total_layers - num_ssm_layers

    ssm_ratio = None
    attention_ratio = None
    if num_total_layers and num_total_layers > 0:
        if num_ssm_layers is not None:
            ssm_ratio = round(float(num_ssm_layers) / float(num_total_layers), 6)
        if num_attention_layers is not None:
            attention_ratio = round(float(num_attention_layers) / float(num_total_layers), 6)

    detected_numeric_fields: Dict[str, int] = {}
    interesting_tokens = ("ssm", "mamba", "rwkv", "xlstm", "lstm", "state", "conv", "hybrid")
    for key in sorted(config_lc.keys()):
        if not any(token in key for token in interesting_tokens):
            continue
        numeric = coerce_int(config_lc[key], strategy="max")
        if numeric is not None:
            detected_numeric_fields[key] = numeric
    if len(detected_numeric_fields) > 32:
        trimmed: Dict[str, int] = {}
        for key in sorted(detected_numeric_fields.keys())[:32]:
            trimmed[key] = detected_numeric_fields[key]
        detected_numeric_fields = trimmed

    return {
        "ssm_state_size": ssm_state_size,
        "ssm_conv_kernel": ssm_conv_kernel,
        "ssm_dt_rank": ssm_dt_rank,
        "num_ssm_layers": num_ssm_layers,
        "num_attention_layers": num_attention_layers,
        "ssm_layer_ratio": ssm_ratio,
        "attention_layer_ratio": attention_ratio,
        "detected_numeric_fields": detected_numeric_fields,
    }


def extract_tokenizer_type(
    tokenizer_lc: Mapping[str, Any],
    config_lc: Mapping[str, Any],
) -> Optional[str]:
    tokenizer_type = pick_str(tokenizer_lc, TOKENIZER_TYPE_KEYS)
    if tokenizer_type:
        return tokenizer_type
    return pick_str(config_lc, ("tokenizer_class", "tokenizer_type"))


def parse_token_candidate(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return int(value)
    if isinstance(value, str):
        direct = parse_human_number(value)
        if direct is not None:
            return direct
        matches = extract_token_candidates_from_text(value)
        if matches:
            return max(matches)
    return None


def extract_token_candidates_from_text(text: Optional[str]) -> List[int]:
    if not text:
        return []

    candidates: List[int] = []

    for regex in (TOKEN_AFTER_NUMBER_RE, TOKEN_AFTER_LABEL_RE):
        for match in regex.finditer(text):
            num = match.group("num")
            suffix = match.group("suffix") or ""
            parsed = parse_human_number(f"{num}{suffix}")
            if parsed is not None:
                candidates.append(parsed)

    broad_regex = re.compile(
        r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<suffix>k|m|b|t|thousand|million|billion|trillion)\s+tokens?\b",
        re.IGNORECASE,
    )
    for match in broad_regex.finditer(text):
        parsed = parse_human_number(f"{match.group('num')}{match.group('suffix')}")
        if parsed is not None:
            candidates.append(parsed)

    return candidates


def should_consider_token_key(key: str) -> bool:
    lowered = key.lower()
    if "token" not in lowered:
        return False
    if any(excluded in lowered for excluded in TOKEN_METADATA_EXCLUDES):
        return False
    if any(token in lowered for token in ("pretrain", "training", "train", "dataset", "corpus")):
        return True
    if lowered in {"tokens", "token_count", "total_tokens", "num_tokens"}:
        return True
    return False


def collect_token_candidates_from_metadata(node: Any) -> List[int]:
    out: List[int] = []

    if isinstance(node, ABCMapping):
        for raw_key, value in node.items():
            key = str(raw_key)
            if should_consider_token_key(key):
                parsed = parse_token_candidate(value)
                if parsed is not None:
                    out.append(parsed)
            out.extend(collect_token_candidates_from_metadata(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(collect_token_candidates_from_metadata(item))
    elif isinstance(node, str) and "token" in node.lower():
        out.extend(extract_token_candidates_from_text(node))

    return out


def extract_pretrain_tokens(card_meta: Mapping[str, Any], card_text: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    meta_candidates = collect_token_candidates_from_metadata(card_meta)
    if meta_candidates:
        return max(meta_candidates), "model_card_metadata"

    text_candidates = extract_token_candidates_from_text(card_text)
    if text_candidates:
        return max(text_candidates), "model_card_text"

    return None, None


def add_quantization_signals_from_text(text: str, formats: Set[str]) -> None:
    lowered = text.lower()
    if ".gguf" in lowered or re.search(r"\bgguf\b", lowered):
        formats.add("GGUF")
    if "gptq" in lowered:
        formats.add("GPTQ")
    if re.search(r"\bawq\b", lowered):
        formats.add("AWQ")
    if "exl2" in lowered:
        formats.add("EXL2")
    if "aqlm" in lowered:
        formats.add("AQLM")
    if re.search(r"\bhqq\b", lowered):
        formats.add("HQQ")
    if "eetq" in lowered:
        formats.add("EETQ")
    if "bitsandbytes" in lowered or re.search(r"\bbnb\b", lowered):
        formats.add("BITSANDBYTES")
    if re.search(r"\bfp8\b|\bfloat8\b", lowered):
        formats.add("FP8")
    if re.search(r"\bint8\b|\b8[- ]bit\b", lowered):
        formats.add("INT8")
    if re.search(r"\bint4\b|\b4[- ]bit\b|\bnf4\b|\bfp4\b", lowered):
        formats.add("INT4")


def detect_quantization_formats(
    repo_files: Set[str],
    tags: Sequence[str],
    card_text: Optional[str],
    config_lc: Mapping[str, Any],
) -> List[str]:
    detected: Set[str] = set()

    for filename in repo_files:
        add_quantization_signals_from_text(filename, detected)
    for tag in tags:
        add_quantization_signals_from_text(tag, detected)
    if card_text:
        add_quantization_signals_from_text(card_text, detected)

    quant_cfg = config_lc.get("quantization_config")
    if isinstance(quant_cfg, ABCMapping):
        quant_cfg_lc = to_lower_key_map({str(k): v for k, v in quant_cfg.items()})
        for value in quant_cfg_lc.values():
            if isinstance(value, (str, int, float, bool)):
                add_quantization_signals_from_text(str(value), detected)
        bits = coerce_int(quant_cfg_lc.get("bits"))
        if bits == 4:
            detected.add("INT4")
        elif bits == 8:
            detected.add("INT8")
    elif quant_cfg is not None:
        add_quantization_signals_from_text(str(quant_cfg), detected)

    ordered = [fmt for fmt in QUANTIZATION_ORDER if fmt in detected]
    remainder = sorted(fmt for fmt in detected if fmt not in QUANTIZATION_ORDER)
    return ordered + remainder


def build_base_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    model_id = str(row["model_id"])
    org, _ = split_model_id(model_id)
    row_org = row.get("org")
    if isinstance(row_org, str) and row_org.strip():
        org = row_org.strip()

    registry_fields: Dict[str, Any] = {}
    for key in ("family", "variant", "tier", "paradigm", "is_quick_test", "is_phase_transition"):
        if key in row:
            registry_fields[key] = row[key]

    return {
        "model_id": model_id,
        "org": org,
        "registry": registry_fields,
        "identity": {
            "model_id": model_id,
            "org": org,
            "release_date": None,
            "license": None,
            "downloads": None,
            "likes": None,
        },
        "architecture": {
            "num_params": None,
            "num_layers": None,
            "hidden_size": None,
            "ffn_size": None,
            "num_heads": None,
            "num_kv_heads": None,
            "head_dim": None,
            "norm_type": None,
            "activation": None,
            "positional_scheme": None,
            "max_context": None,
            "sliding_window": None,
            "vocab_size": None,
        },
        "paradigm_specific": {
            "ssm_state_size": None,
            "ssm_conv_kernel": None,
            "ssm_dt_rank": None,
            "num_ssm_layers": None,
            "num_attention_layers": None,
            "ssm_layer_ratio": None,
            "attention_layer_ratio": None,
            "detected_numeric_fields": {},
        },
        "training": {
            "pretrain_tokens": None,
            "pretrain_tokens_source": None,
            "tokenizer_type": None,
        },
        "quantization": {
            "available_formats": [],
        },
        "sources": {
            "hf_api": False,
            "config_json": None,
            "tokenizer_config_json": None,
            "model_card": None,
            "safetensors_index": None,
        },
        "errors": [],
    }


def enrich_model_row(
    row: Mapping[str, Any],
    api: HfApi,
    revision: str,
    token: Optional[str],
    retries: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    result = build_base_result(row)
    model_id = result["model_id"]

    try:
        model_info = fetch_model_info(
            api=api,
            model_id=model_id,
            revision=revision,
            token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
    except Exception as exc:
        result["errors"].append(f"hf_api_error: {exc}")
        return result

    result["sources"]["hf_api"] = True

    tags = normalize_tags(get_attr_or_key(model_info, "tags", []))
    repo_files, file_sizes = get_repo_files_and_sizes(
        api=api,
        model_id=model_id,
        model_info=model_info,
        revision=revision,
        token=token,
        retries=retries,
        backoff_seconds=backoff_seconds,
    )

    config_json: Dict[str, Any] = {}
    tokenizer_config_json: Dict[str, Any] = {}
    safetensors_index_json: Optional[Dict[str, Any]] = None
    card_text: Optional[str] = None
    card_meta: Dict[str, Any] = {}

    if should_attempt_file("config.json", repo_files):
        parsed, err = download_json_file(
            model_id=model_id,
            filename="config.json",
            revision=revision,
            token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if parsed is not None:
            config_json = parsed
            result["sources"]["config_json"] = "config.json"
        elif err and err != "missing":
            result["errors"].append(f"config.json: {err}")

    if should_attempt_file("tokenizer_config.json", repo_files):
        parsed, err = download_json_file(
            model_id=model_id,
            filename="tokenizer_config.json",
            revision=revision,
            token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if parsed is not None:
            tokenizer_config_json = parsed
            result["sources"]["tokenizer_config_json"] = "tokenizer_config.json"
        elif err and err != "missing":
            result["errors"].append(f"tokenizer_config.json: {err}")

    readme_file = find_readme_filename(repo_files)
    if readme_file:
        text, err = download_text_file(
            model_id=model_id,
            filename=readme_file,
            revision=revision,
            token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if text is not None:
            card_text = text
            result["sources"]["model_card"] = readme_file
            card_meta = parse_model_card_front_matter(card_text)
            tags.extend(normalize_tags(card_meta.get("tags")))
        elif err and err != "missing":
            result["errors"].append(f"{readme_file}: {err}")

    safetensors_index_file = choose_safetensors_index(repo_files)
    if safetensors_index_file:
        parsed, err = download_json_file(
            model_id=model_id,
            filename=safetensors_index_file,
            revision=revision,
            token=token,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        if parsed is not None:
            safetensors_index_json = parsed
            result["sources"]["safetensors_index"] = safetensors_index_file
        elif err and err != "missing":
            result["errors"].append(f"{safetensors_index_file}: {err}")

    config_lc = to_lower_key_map(config_json) if config_json else {}
    tokenizer_lc = to_lower_key_map(tokenizer_config_json) if tokenizer_config_json else {}
    unique_tags = sorted(set(tags))

    result["identity"] = {
        "model_id": model_id,
        "org": result["org"],
        "release_date": infer_release_date(model_info, card_meta),
        "license": infer_license(card_meta, unique_tags, config_lc),
        "downloads": coerce_int(get_attr_or_key(model_info, "downloads")),
        "likes": coerce_int(get_attr_or_key(model_info, "likes")),
    }

    architecture = extract_architecture(
        model_info=model_info,
        config_lc=config_lc,
        tokenizer_lc=tokenizer_lc,
        safetensors_index=safetensors_index_json,
        file_sizes=file_sizes,
    )
    result["architecture"] = architecture
    result["paradigm_specific"] = extract_paradigm_specific(config_lc, architecture)

    pretrain_tokens, pretrain_tokens_source = extract_pretrain_tokens(card_meta, card_text)
    result["training"] = {
        "pretrain_tokens": pretrain_tokens,
        "pretrain_tokens_source": pretrain_tokens_source,
        "tokenizer_type": extract_tokenizer_type(tokenizer_lc, config_lc),
    }

    result["quantization"] = {
        "available_formats": detect_quantization_formats(
            repo_files=repo_files,
            tags=unique_tags,
            card_text=card_text,
            config_lc=config_lc,
        )
    }

    return result


def run_ingest(args: argparse.Namespace) -> Dict[str, Any]:
    registry_path = Path(args.registry_path)
    output_path = Path(args.output_path)

    rows = load_registry_rows(registry_path)
    api = HfApi()

    results_by_model_id: Dict[str, Dict[str, Any]] = {}
    total = len(rows)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_model_id = {
            executor.submit(
                enrich_model_row,
                row,
                api,
                args.revision,
                args.hf_token,
                args.retries,
                args.retry_backoff_seconds,
            ): str(row["model_id"])
            for row in rows
        }

        completed = 0
        for future in as_completed(future_to_model_id):
            model_id = future_to_model_id[future]
            completed += 1
            try:
                enriched = future.result()
            except Exception as exc:
                enriched = build_base_result({"model_id": model_id})
                enriched["errors"].append(f"unexpected_worker_error: {exc}")

            results_by_model_id[model_id] = enriched
            if not args.quiet:
                print(f"[{completed}/{total}] {model_id}", file=sys.stderr)

    ordered_results = [results_by_model_id[str(row["model_id"])] for row in rows]
    failed = sum(1 for item in ordered_results if item.get("errors"))

    payload: Dict[str, Any] = {
        "source_file": str(registry_path),
        "generated_at_utc": utc_now_iso(),
        "model_count": len(ordered_results),
        "failed_count": failed,
        "success_count": len(ordered_results) - failed,
        "models": ordered_results,
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write {output_path}: {exc}") from exc

    if not args.quiet:
        print(f"Wrote {len(ordered_results)} records to {output_path}", file=sys.stderr)

    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_ingest(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
