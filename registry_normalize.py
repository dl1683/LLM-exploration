#!/usr/bin/env python3
"""
registry_normalize.py

Parses model_registry_2026.py, builds a canonical model table, runs integrity checks,
and outputs JSON.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TARGET_VARS: Tuple[str, ...] = (
    "TIER1_2026",
    "TIER2_2026",
    "TIER3_2026",
    "QUICK_TEST",
    "PHASE_TRANSITION_MODELS",
    "PARADIGMS",
)

TIER_VAR_TO_NAME: Tuple[Tuple[str, str], ...] = (
    ("TIER1_2026", "tier1"),
    ("TIER2_2026", "tier2"),
    ("TIER3_2026", "tier3"),
)

TIER_SORT_RANK: Dict[str, int] = {
    "tier1": 1,
    "tier2": 2,
    "tier3": 3,
    "unassigned": 4,
}

SIZE_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?[bBmMkK]$")
NUMERIC_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")

IMBALANCE_LOW_FACTOR = 0.50
IMBALANCE_HIGH_FACTOR = 1.50
IMBALANCE_MAX_MIN_RATIO = 3.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize model_registry_2026.py into a canonical JSON table."
    )
    parser.add_argument(
        "--registry-path",
        default="model_registry_2026.py",
        help="Path to model_registry_2026.py (default: model_registry_2026.py).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if critical integrity issues are found.",
    )
    return parser.parse_args(argv)


def ensure_list_of_strings(name: str, value: Any) -> None:
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must be a list.")
    bad_indices = [idx for idx, item in enumerate(value) if not isinstance(item, str)]
    if bad_indices:
        joined = ", ".join(str(i) for i in bad_indices)
        raise RuntimeError(f"{name} contains non-string items at indices: {joined}")


def validate_registry_shapes(values: Mapping[str, Any]) -> None:
    for var_name, _tier in TIER_VAR_TO_NAME:
        ensure_list_of_strings(var_name, values[var_name])

    ensure_list_of_strings("QUICK_TEST", values["QUICK_TEST"])
    ensure_list_of_strings("PHASE_TRANSITION_MODELS", values["PHASE_TRANSITION_MODELS"])

    paradigms = values["PARADIGMS"]
    if not isinstance(paradigms, dict):
        raise RuntimeError("PARADIGMS must be a dict[str, list[str]].")

    for paradigm_name, models in paradigms.items():
        if not isinstance(paradigm_name, str):
            raise RuntimeError("PARADIGMS keys must be strings.")
        ensure_list_of_strings(f"PARADIGMS[{paradigm_name!r}]", models)


def _extract_assignment_name_and_value(node: ast.AST) -> List[Tuple[str, ast.AST]]:
    pairs: List[Tuple[str, ast.AST]] = []

    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                pairs.append((target.id, node.value))
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.value is not None:
            pairs.append((node.target.id, node.value))

    return pairs


def load_registry_literals(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.exists():
        raise RuntimeError(f"Registry file not found: {registry_path}")

    try:
        source = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {registry_path}: {exc}") from exc

    try:
        tree = ast.parse(source, filename=str(registry_path))
    except SyntaxError as exc:
        raise RuntimeError(f"Failed to parse {registry_path}: {exc}") from exc

    values: Dict[str, Any] = {}
    for node in tree.body:
        for name, value_node in _extract_assignment_name_and_value(node):
            if name not in TARGET_VARS:
                continue
            try:
                values[name] = ast.literal_eval(value_node)
            except (ValueError, SyntaxError) as exc:
                raise RuntimeError(
                    f"Variable {name} in {registry_path} is not a literal list/dict."
                ) from exc

    missing_vars = [name for name in TARGET_VARS if name not in values]
    if missing_vars:
        raise RuntimeError(
            "Missing required variables in registry: " + ", ".join(missing_vars)
        )

    validate_registry_shapes(values)
    return values


def split_org_and_name(model_id: str) -> Tuple[str, str]:
    if "/" in model_id:
        org, model_name = model_id.split("/", 1)
        return org, model_name
    return "unknown", model_id


def infer_family_variant(model_name: str) -> Tuple[str, str]:
    tokens = [t for t in model_name.split("-") if t]
    if not tokens:
        return model_name, ""

    size_idx: Optional[int] = None
    for idx, token in enumerate(tokens):
        if SIZE_TOKEN_RE.match(token):
            size_idx = idx
            break

    if size_idx is not None:
        family = "-".join(tokens[:size_idx]) or tokens[0]
        variant = "-".join(tokens[size_idx:])
        return family, variant

    if len(tokens) == 1:
        return tokens[0], ""

    if NUMERIC_TOKEN_RE.match(tokens[1]):
        family = "-".join(tokens[:2])
        variant = "-".join(tokens[2:])
        return family, variant

    family = tokens[0]
    variant = "-".join(tokens[1:])
    return family, variant


def duplicate_counts(items: List[str]) -> Dict[str, int]:
    counts = Counter(items)
    return {item: counts[item] for item in sorted(counts) if counts[item] > 1}


def build_group_membership(groups: Mapping[str, List[str]]) -> Dict[str, List[str]]:
    membership: Dict[str, List[str]] = defaultdict(list)
    for group_name, models in groups.items():
        for model_id in models:
            membership[model_id].append(group_name)
    return membership


def find_across_group_duplicates(
    membership: Mapping[str, List[str]]
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for model_id, groups in membership.items():
        unique_groups = sorted(set(groups))
        if len(unique_groups) > 1:
            out[model_id] = unique_groups
    return dict(sorted(out.items()))


def membership_label(groups: List[str], unassigned: str = "unassigned") -> str:
    unique_groups = sorted(set(groups))
    if not unique_groups:
        return unassigned
    if len(unique_groups) == 1:
        return unique_groups[0]
    return "MULTI:" + ",".join(unique_groups)


def build_canonical_table(values: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tier_groups: Dict[str, List[str]] = {
        tier_name: list(values[var_name]) for var_name, tier_name in TIER_VAR_TO_NAME
    }
    paradigm_groups: Dict[str, List[str]] = {
        str(name): list(models) for name, models in values["PARADIGMS"].items()
    }

    tier_membership = build_group_membership(tier_groups)
    paradigm_membership = build_group_membership(paradigm_groups)

    quick_test_set = set(values["QUICK_TEST"])
    phase_transition_set = set(values["PHASE_TRANSITION_MODELS"])

    all_models = sorted(
        set(tier_membership.keys())
        | set(paradigm_membership.keys())
        | quick_test_set
        | phase_transition_set
    )

    rows: List[Dict[str, Any]] = []
    for model_id in all_models:
        org, model_name = split_org_and_name(model_id)
        family, variant = infer_family_variant(model_name)

        row = {
            "model_id": model_id,
            "org": org,
            "family": family,
            "variant": variant,
            "tier": membership_label(tier_membership.get(model_id, [])),
            "paradigm": membership_label(paradigm_membership.get(model_id, [])),
            "is_quick_test": model_id in quick_test_set,
            "is_phase_transition": model_id in phase_transition_set,
        }
        rows.append(row)

    def sort_key(row: Mapping[str, Any]) -> Tuple[int, str, str]:
        tier_val = str(row["tier"])
        if tier_val.startswith("MULTI:"):
            rank = 0
        else:
            rank = TIER_SORT_RANK.get(tier_val, 99)
        return rank, tier_val, str(row["model_id"])

    rows.sort(key=sort_key)
    return rows


def run_integrity_checks(values: Mapping[str, Any]) -> Dict[str, Any]:
    tier_groups: Dict[str, List[str]] = {
        tier_name: list(values[var_name]) for var_name, tier_name in TIER_VAR_TO_NAME
    }
    paradigm_groups: Dict[str, List[str]] = {
        str(name): list(models) for name, models in values["PARADIGMS"].items()
    }

    tier_membership = build_group_membership(tier_groups)
    paradigm_membership = build_group_membership(paradigm_groups)

    tier_model_set = set(tier_membership.keys())
    paradigm_model_set = set(paradigm_membership.keys())

    missing_from_paradigms = sorted(tier_model_set - paradigm_model_set)

    within_tier_duplicates: Dict[str, Dict[str, int]] = {}
    for tier_name, models in tier_groups.items():
        dups = duplicate_counts(models)
        if dups:
            within_tier_duplicates[tier_name] = dups

    within_paradigm_duplicates: Dict[str, Dict[str, int]] = {}
    for paradigm_name, models in paradigm_groups.items():
        dups = duplicate_counts(models)
        if dups:
            within_paradigm_duplicates[paradigm_name] = dups

    quick_test_duplicates = duplicate_counts(list(values["QUICK_TEST"]))
    phase_transition_duplicates = duplicate_counts(list(values["PHASE_TRANSITION_MODELS"]))

    across_tier_duplicates = find_across_group_duplicates(tier_membership)
    across_paradigm_duplicates = find_across_group_duplicates(paradigm_membership)

    paradigm_counts = {
        paradigm_name: len(models)
        for paradigm_name, models in sorted(paradigm_groups.items())
    }

    if paradigm_counts:
        counts = list(paradigm_counts.values())
        total = sum(counts)
        mean = total / len(counts)
        min_count = min(counts)
        max_count = max(counts)

        max_to_min_ratio = None if min_count == 0 else round(max_count / min_count, 4)
        underrepresented = sorted(
            [
                name
                for name, count in paradigm_counts.items()
                if count < (mean * IMBALANCE_LOW_FACTOR)
            ]
        )
        overrepresented = sorted(
            [
                name
                for name, count in paradigm_counts.items()
                if count > (mean * IMBALANCE_HIGH_FACTOR)
            ]
        )

        is_imbalanced = (
            min_count == 0
            or (max_to_min_ratio is not None and max_to_min_ratio > IMBALANCE_MAX_MIN_RATIO)
            or bool(underrepresented)
            or bool(overrepresented)
        )
    else:
        total = 0
        mean = 0.0
        min_count = 0
        max_count = 0
        max_to_min_ratio = None
        underrepresented = []
        overrepresented = []
        is_imbalanced = False

    has_critical_issues = (
        bool(missing_from_paradigms)
        or bool(within_tier_duplicates)
        or bool(within_paradigm_duplicates)
        or bool(across_tier_duplicates)
        or bool(across_paradigm_duplicates)
        or bool(quick_test_duplicates)
        or bool(phase_transition_duplicates)
    )

    return {
        "missing_from_paradigms": missing_from_paradigms,
        "duplicates": {
            "within_tiers": within_tier_duplicates,
            "across_tiers": across_tier_duplicates,
            "within_paradigms": within_paradigm_duplicates,
            "across_paradigms": across_paradigm_duplicates,
            "quick_test": quick_test_duplicates,
            "phase_transition": phase_transition_duplicates,
        },
        "paradigm_imbalance": {
            "counts": paradigm_counts,
            "total_models": total,
            "mean_count": round(mean, 4),
            "min_count": min_count,
            "max_count": max_count,
            "max_to_min_ratio": max_to_min_ratio,
            "underrepresented": underrepresented,
            "overrepresented": overrepresented,
            "thresholds": {
                "underrepresented_below_mean_factor": IMBALANCE_LOW_FACTOR,
                "overrepresented_above_mean_factor": IMBALANCE_HIGH_FACTOR,
                "max_to_min_ratio": IMBALANCE_MAX_MIN_RATIO,
            },
            "is_imbalanced": is_imbalanced,
        },
        "has_critical_issues": has_critical_issues,
    }


def normalize_registry(registry_path: Path) -> Dict[str, Any]:
    values = load_registry_literals(registry_path)
    canonical_table = build_canonical_table(values)
    integrity_checks = run_integrity_checks(values)

    return {
        "source_file": str(registry_path),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "canonical_table": canonical_table,
        "integrity_checks": integrity_checks,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    registry_path = Path(args.registry_path)

    try:
        output = normalize_registry(registry_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.strict and output["integrity_checks"]["has_critical_issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
