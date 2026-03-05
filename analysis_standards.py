#!/usr/bin/env python3
"""
analysis_standards.py

Standards toolkit for analysis quality, scoring, and portfolio tracking.

Core goals:
1) Force explicit investor and research value statements.
2) Enforce a minimum quality gate for credible claims.
3) Compute two comparable portfolio scores:
   - investor_signal_score (0-100)
   - research_depth_score (0-100)
4) Write a standardized result card and append to a scoreboard.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


INVESTOR_WEIGHTS: Dict[str, float] = {
    "actionability": 0.20,
    "deployment_relevance": 0.20,
    "cost_signal": 0.15,
    "adoption_signal": 0.15,
    "timeliness": 0.10,
    "data_quality": 0.10,
    "identification_strength": 0.10,
}

RESEARCH_WEIGHTS: Dict[str, float] = {
    "novelty": 0.15,
    "theoretical_depth": 0.15,
    "measurement_quality": 0.15,
    "causal_identification": 0.20,
    "robustness": 0.15,
    "reproducibility": 0.10,
    "generality": 0.10,
}

QUALITY_GATE_KEYS: Tuple[str, ...] = (
    "non_circular_metrics",
    "causal_design_matches_claim",
    "out_of_sample_validation",
    "uncertainty_reported",
    "robustness_checks",
    "reproducible_pipeline",
)

REQUIRED_TEXT_FIELDS: Tuple[str, ...] = (
    "analysis_id",
    "title",
    "track",
    "owner",
    "status",
    "hypothesis",
    "value_to_investors",
    "value_to_researchers",
    "identification_strategy",
    "out_of_sample_validation_plan",
    "business_implication",
    "scientific_implication",
    "evidence_strength",
)

REQUIRED_LIST_FIELDS: Tuple[str, ...] = (
    "falsification_tests",
    "success_criteria",
    "uncertainty_methods",
    "robustness_checks",
)

VALID_TRACKS = {"investor", "research", "dual"}
VALID_STATUS = {"draft", "in_progress", "complete", "archived"}
VALID_EVIDENCE_STRENGTH = {"exploratory", "suggestive", "strong", "causal"}

PLACEHOLDER_TOKENS = ("todo", "tbd", "replace_me", "placeholder")

SCOREBOARD_HEADERS = [
    "scored_at_utc",
    "analysis_id",
    "title",
    "track",
    "status",
    "investor_signal_score",
    "research_depth_score",
    "quality_gate_pass",
    "evidence_strength",
    "analysis_dir",
    "business_implication",
    "scientific_implication",
    "validation_notes",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_placeholder_text(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in PLACEHOLDER_TOKENS)


def validate_rating_block(
    ratings: Dict[str, Any],
    weights: Dict[str, float],
    block_name: str,
) -> List[str]:
    errors: List[str] = []
    expected = set(weights.keys())
    got = set(ratings.keys())

    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        errors.append(f"{block_name} rubric missing keys: {', '.join(missing)}")
    if extra:
        errors.append(f"{block_name} rubric has unexpected keys: {', '.join(extra)}")

    for key in sorted(expected & got):
        raw = ratings[key]
        if not isinstance(raw, (int, float)):
            errors.append(f"{block_name}.{key} must be numeric 0-5")
            continue
        if raw < 0 or raw > 5:
            errors.append(f"{block_name}.{key} must be in [0, 5]")
    return errors


def validate_manifest(
    manifest: Dict[str, Any],
    allow_placeholders: bool = False,
) -> List[str]:
    errors: List[str] = []

    for field in REQUIRED_TEXT_FIELDS:
        value = manifest.get(field)
        if not is_non_empty_text(value):
            errors.append(f"Missing/empty required field: {field}")
            continue
        if not allow_placeholders and has_placeholder_text(str(value)):
            errors.append(f"Field contains placeholder text: {field}")

    for field in REQUIRED_LIST_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, list) or len(value) == 0:
            errors.append(f"Field must be a non-empty list: {field}")

    track = str(manifest.get("track", "")).strip().lower()
    if track not in VALID_TRACKS:
        errors.append(f"Invalid track: {track} (expected one of {sorted(VALID_TRACKS)})")

    status = str(manifest.get("status", "")).strip().lower()
    if status not in VALID_STATUS:
        errors.append(f"Invalid status: {status} (expected one of {sorted(VALID_STATUS)})")

    evidence_strength = str(manifest.get("evidence_strength", "")).strip().lower()
    if evidence_strength not in VALID_EVIDENCE_STRENGTH:
        errors.append(
            "Invalid evidence_strength: "
            f"{evidence_strength} (expected one of {sorted(VALID_EVIDENCE_STRENGTH)})"
        )

    quality_gates = manifest.get("quality_gates")
    if not isinstance(quality_gates, dict):
        errors.append("Missing quality_gates block")
    else:
        for key in QUALITY_GATE_KEYS:
            if key not in quality_gates:
                errors.append(f"Missing quality gate key: quality_gates.{key}")
            elif not isinstance(quality_gates[key], bool):
                errors.append(f"Quality gate must be boolean: quality_gates.{key}")

    rubric = manifest.get("rubric")
    if not isinstance(rubric, dict):
        errors.append("Missing rubric block")
    else:
        investor = rubric.get("investor")
        research = rubric.get("research")
        if not isinstance(investor, dict):
            errors.append("Missing rubric.investor block")
        else:
            errors.extend(validate_rating_block(investor, INVESTOR_WEIGHTS, "rubric.investor"))
        if not isinstance(research, dict):
            errors.append("Missing rubric.research block")
        else:
            errors.extend(validate_rating_block(research, RESEARCH_WEIGHTS, "rubric.research"))

    return errors


def parse_key_findings_implications(
    key_findings_path: Path,
    allow_placeholders: bool = False,
) -> List[str]:
    errors: List[str] = []
    if not key_findings_path.exists():
        return [f"Missing key findings file: {key_findings_path}"]

    text = key_findings_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    business_lines = [
        line for line in lines if line.lower().startswith("business_implication:")
    ]
    scientific_lines = [
        line for line in lines if line.lower().startswith("scientific_implication:")
    ]

    if not business_lines:
        errors.append("key_findings.txt missing `business_implication:` line")
    if not scientific_lines:
        errors.append("key_findings.txt missing `scientific_implication:` line")

    if not allow_placeholders:
        for line in business_lines + scientific_lines:
            value = line.split(":", 1)[1].strip() if ":" in line else ""
            if has_placeholder_text(value) or not value:
                errors.append("key_findings implication line still has placeholder text")
                break

    return errors


def weighted_score(ratings: Dict[str, Any], weights: Dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        raw = float(ratings.get(key, 0.0))
        clipped = min(5.0, max(0.0, raw))
        score += (clipped / 5.0) * weight
    return round(score * 100.0, 2)


def quality_gate_pass(quality_gates: Dict[str, Any]) -> bool:
    return all(bool(quality_gates.get(key, False)) for key in QUALITY_GATE_KEYS)


def normalize_one_line(value: Any) -> str:
    return " ".join(str(value).split())


def build_result_card(
    manifest: Dict[str, Any],
    investor_score: float,
    research_score: float,
    gate_passed: bool,
    validation_errors: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# Result Card: {manifest['title']}")
    lines.append("")
    lines.append(f"- Analysis ID: `{manifest['analysis_id']}`")
    lines.append(f"- Track: `{manifest['track']}`")
    lines.append(f"- Owner: `{manifest['owner']}`")
    lines.append(f"- Status: `{manifest['status']}`")
    lines.append(f"- Evidence Strength: `{manifest['evidence_strength']}`")
    lines.append(f"- Scored At (UTC): `{utc_now_iso()}`")
    lines.append("")
    lines.append("## Scores")
    lines.append(f"- Investor Signal Score: `{investor_score:.2f}` / 100")
    lines.append(f"- Research Depth Score: `{research_score:.2f}` / 100")
    lines.append(f"- Quality Gate Pass: `{gate_passed}`")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append(f"- {normalize_one_line(manifest['hypothesis'])}")
    lines.append("")
    lines.append("## Implications")
    lines.append(f"- Business: {normalize_one_line(manifest['business_implication'])}")
    lines.append(f"- Scientific: {normalize_one_line(manifest['scientific_implication'])}")
    lines.append("")
    lines.append("## Identification and Checks")
    lines.append(f"- Identification Strategy: {normalize_one_line(manifest['identification_strategy'])}")
    lines.append("- Falsification Tests:")
    for item in manifest.get("falsification_tests", []):
        lines.append(f"  - {normalize_one_line(item)}")
    lines.append("- Robustness Checks:")
    for item in manifest.get("robustness_checks", []):
        lines.append(f"  - {normalize_one_line(item)}")
    lines.append("- Uncertainty Methods:")
    for item in manifest.get("uncertainty_methods", []):
        lines.append(f"  - {normalize_one_line(item)}")
    lines.append("")
    lines.append("## Validation")
    if validation_errors:
        for err in validation_errors:
            lines.append(f"- FAIL: {err}")
    else:
        lines.append("- PASS: manifest and implication checks passed")
    lines.append("")
    return "\n".join(lines)


def append_scoreboard_row(scoreboard_path: Path, row: Dict[str, Any]) -> None:
    scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    exists = scoreboard_path.exists()

    with scoreboard_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCOREBOARD_HEADERS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SCOREBOARD_HEADERS})


def init_analysis(args: argparse.Namespace) -> int:
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    template_path = Path(args.manifest_template)
    manifest = load_json(template_path)

    manifest["analysis_id"] = args.analysis_id or manifest["analysis_id"]
    manifest["title"] = args.title or manifest["title"]
    manifest["track"] = args.track or manifest["track"]
    manifest["owner"] = args.owner or manifest["owner"]
    manifest["created_at_utc"] = utc_now_iso()

    manifest_path = analysis_dir / "analysis_manifest.json"
    key_findings_path = analysis_dir / "key_findings.txt"
    result_card_path = analysis_dir / "result_card.md"

    if manifest_path.exists() and not args.force:
        print(f"SKIP (exists): {manifest_path}")
    else:
        write_json(manifest_path, manifest)
        print(f"WROTE: {manifest_path}")

    if key_findings_path.exists() and not args.force:
        print(f"SKIP (exists): {key_findings_path}")
    else:
        key_findings_path.write_text(
            "\n".join(
                [
                    "business_implication: TODO",
                    "scientific_implication: TODO",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"WROTE: {key_findings_path}")

    template_result_card = Path(args.result_card_template).read_text(encoding="utf-8")
    if result_card_path.exists() and not args.force:
        print(f"SKIP (exists): {result_card_path}")
    else:
        result_card_path.write_text(template_result_card, encoding="utf-8")
        print(f"WROTE: {result_card_path}")

    return 0


def validate_one_dir(
    analysis_dir: Path,
    allow_placeholders: bool = False,
    require_manifest: bool = True,
) -> List[str]:
    errors: List[str] = []
    manifest_path = analysis_dir / "analysis_manifest.json"
    key_findings_path = analysis_dir / "key_findings.txt"

    if require_manifest and not manifest_path.exists():
        errors.append(f"Missing file: {manifest_path}")
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed to parse manifest JSON: {exc}")
            manifest = None
        if manifest is not None:
            errors.extend(validate_manifest(manifest, allow_placeholders=allow_placeholders))

    errors.extend(
        parse_key_findings_implications(
            key_findings_path,
            allow_placeholders=allow_placeholders,
        )
    )
    return errors


def validate_analysis(args: argparse.Namespace) -> int:
    analysis_dir = Path(args.analysis_dir)
    errors = validate_one_dir(
        analysis_dir=analysis_dir,
        allow_placeholders=args.allow_placeholders,
        require_manifest=True,
    )
    if errors:
        print(f"VALIDATION FAILED: {analysis_dir}")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"VALIDATION PASSED: {analysis_dir}")
    return 0


def score_analysis(args: argparse.Namespace) -> int:
    analysis_dir = Path(args.analysis_dir)
    manifest_path = analysis_dir / "analysis_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Missing {manifest_path}")
        return 1

    manifest = load_json(manifest_path)
    errors = validate_one_dir(
        analysis_dir=analysis_dir,
        allow_placeholders=args.allow_placeholders,
        require_manifest=True,
    )
    if errors and not args.allow_incomplete:
        print("ERROR: Validation failed. Fix issues or pass --allow-incomplete.")
        for err in errors:
            print(f"  - {err}")
        return 1

    investor_score = weighted_score(manifest["rubric"]["investor"], INVESTOR_WEIGHTS)
    research_score = weighted_score(manifest["rubric"]["research"], RESEARCH_WEIGHTS)
    gate_passed = quality_gate_pass(manifest["quality_gates"])

    if args.apply_gate_penalty and not gate_passed:
        investor_score = round(investor_score * 0.70, 2)
        research_score = round(research_score * 0.70, 2)

    result_card = build_result_card(
        manifest=manifest,
        investor_score=investor_score,
        research_score=research_score,
        gate_passed=gate_passed,
        validation_errors=errors,
    )
    result_card_path = analysis_dir / "result_card.md"
    result_card_path.write_text(result_card, encoding="utf-8")

    row = {
        "scored_at_utc": utc_now_iso(),
        "analysis_id": manifest["analysis_id"],
        "title": manifest["title"],
        "track": manifest["track"],
        "status": manifest["status"],
        "investor_signal_score": f"{investor_score:.2f}",
        "research_depth_score": f"{research_score:.2f}",
        "quality_gate_pass": str(gate_passed),
        "evidence_strength": manifest["evidence_strength"],
        "analysis_dir": str(analysis_dir),
        "business_implication": normalize_one_line(manifest["business_implication"]),
        "scientific_implication": normalize_one_line(manifest["scientific_implication"]),
        "validation_notes": "validation_passed" if not errors else " | ".join(errors[:5]),
    }
    append_scoreboard_row(Path(args.scoreboard), row)

    print(f"SCORED: {analysis_dir}")
    print(f"  investor_signal_score={investor_score:.2f}")
    print(f"  research_depth_score={research_score:.2f}")
    print(f"  quality_gate_pass={gate_passed}")
    print(f"WROTE: {result_card_path}")
    print(f"APPENDED: {args.scoreboard}")
    return 0


def audit_analyses(args: argparse.Namespace) -> int:
    root = Path(args.analysis_root)
    if not root.exists():
        print(f"ERROR: Missing analysis root: {root}")
        return 1

    analysis_dirs: List[Path] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and (entry / "key_findings.txt").exists():
            analysis_dirs.append(entry)

    total = len(analysis_dirs)
    failed = 0

    for directory in analysis_dirs:
        errors = validate_one_dir(
            analysis_dir=directory,
            allow_placeholders=args.allow_placeholders,
            require_manifest=not args.allow_missing_manifest,
        )
        if errors:
            failed += 1
            print(f"[FAIL] {directory}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"[PASS] {directory}")

    print("")
    print(f"AUDIT SUMMARY: {total - failed}/{total} passed")
    if failed and args.strict:
        return 1
    return 0


def backfill_implications(args: argparse.Namespace) -> int:
    root = Path(args.analysis_root)
    if not root.exists():
        print(f"ERROR: Missing analysis root: {root}")
        return 1

    touched = 0
    scanned = 0
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        key_findings = entry / "key_findings.txt"
        if not key_findings.exists():
            continue
        scanned += 1

        text = key_findings.read_text(encoding="utf-8")
        lines = text.splitlines()
        non_empty = [line.strip() for line in lines if line.strip()]
        has_business = any(
            line.lower().startswith("business_implication:") for line in non_empty
        )
        has_scientific = any(
            line.lower().startswith("scientific_implication:") for line in non_empty
        )

        additions: List[str] = []
        if not has_business:
            additions.append(f"business_implication: {args.business_placeholder}")
        if not has_scientific:
            additions.append(f"scientific_implication: {args.scientific_placeholder}")

        if additions:
            touched += 1
            print(f"[UPDATE] {key_findings}")
            for item in additions:
                print(f"  + {item}")
            if not args.dry_run:
                new_text = text.rstrip("\n")
                if new_text:
                    new_text += "\n"
                new_text += "\n".join(additions) + "\n"
                key_findings.write_text(new_text, encoding="utf-8")

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print("")
    print(f"BACKFILL SUMMARY ({mode}): updated {touched}/{scanned} key_findings files")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analysis standards, validation, scoring, and portfolio tracking"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Initialize a new analysis directory with standards files")
    init_cmd.add_argument("--analysis-dir", required=True, help="Target analysis directory")
    init_cmd.add_argument("--analysis-id", default=None, help="Analysis ID override")
    init_cmd.add_argument("--title", default=None, help="Title override")
    init_cmd.add_argument("--track", choices=sorted(VALID_TRACKS), default=None, help="Track override")
    init_cmd.add_argument("--owner", default=None, help="Owner override")
    init_cmd.add_argument(
        "--manifest-template",
        default="standards/analysis_manifest_template.json",
        help="Path to manifest JSON template",
    )
    init_cmd.add_argument(
        "--result-card-template",
        default="standards/templates/result_card_template.md",
        help="Path to result card markdown template",
    )
    init_cmd.add_argument("--force", action="store_true", help="Overwrite existing files")
    init_cmd.set_defaults(func=init_analysis)

    validate_cmd = sub.add_parser("validate", help="Validate one analysis directory")
    validate_cmd.add_argument("--analysis-dir", required=True, help="Analysis directory to validate")
    validate_cmd.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow TODO/placeholder text while drafting",
    )
    validate_cmd.set_defaults(func=validate_analysis)

    score_cmd = sub.add_parser("score", help="Score one analysis and append to scoreboard")
    score_cmd.add_argument("--analysis-dir", required=True, help="Analysis directory to score")
    score_cmd.add_argument(
        "--scoreboard",
        default="analysis/portfolio_scoreboard.csv",
        help="CSV scoreboard path",
    )
    score_cmd.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow TODO/placeholder text while drafting",
    )
    score_cmd.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Score even when validation errors exist",
    )
    score_cmd.add_argument(
        "--apply-gate-penalty",
        action="store_true",
        help="Apply 30% penalty when quality gates are not all true",
    )
    score_cmd.set_defaults(func=score_analysis)

    audit_cmd = sub.add_parser(
        "audit",
        help="Audit all analysis subdirectories containing key_findings.txt",
    )
    audit_cmd.add_argument(
        "--analysis-root",
        default="analysis",
        help="Root directory containing analysis outputs",
    )
    audit_cmd.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow TODO/placeholder text",
    )
    audit_cmd.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Do not fail directories that do not yet have analysis_manifest.json",
    )
    audit_cmd.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any audit checks fail",
    )
    audit_cmd.set_defaults(func=audit_analyses)

    backfill_cmd = sub.add_parser(
        "backfill-implications",
        help="Append missing implication lines to existing key_findings files",
    )
    backfill_cmd.add_argument(
        "--analysis-root",
        default="analysis",
        help="Root directory containing analysis outputs",
    )
    backfill_cmd.add_argument(
        "--business-placeholder",
        default="TODO",
        help="Placeholder text for missing business implication lines",
    )
    backfill_cmd.add_argument(
        "--scientific-placeholder",
        default="TODO",
        help="Placeholder text for missing scientific implication lines",
    )
    backfill_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned edits without writing files",
    )
    backfill_cmd.set_defaults(func=backfill_implications)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
