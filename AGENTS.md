# AGENTS.md

This file is the canonical agent policy for this repository.
All coding agents (Codex, Claude, Cursor, etc.) should follow it.

## Mission
Produce analysis outputs that are:
1. Useful for investors.
2. Meaningful for fundamental LLM research.
3. Methodologically credible and reproducible.

## Mandatory Standards Workflow
For any new or updated analysis in `analysis/*`, agents must run this workflow:

1. Scaffold (new analyses only):
```bash
python analysis_standards.py init --analysis-dir analysis/<analysis_name> --analysis-id <id> --title "<title>" --track <investor|research|dual> --owner <owner>
```

2. Fill required artifacts:
- `analysis/<analysis_name>/analysis_manifest.json`
- `analysis/<analysis_name>/key_findings.txt`
  - Must include:
    - `business_implication: ...`
    - `scientific_implication: ...`

3. Validate:
```bash
python analysis_standards.py validate --analysis-dir analysis/<analysis_name>
```

4. Score + register portfolio impact:
```bash
python analysis_standards.py score --analysis-dir analysis/<analysis_name> --scoreboard analysis/portfolio_scoreboard.csv --apply-gate-penalty
```

5. Result card is required:
- `analysis/<analysis_name>/result_card.md`

## Portfolio Audit Requirement
Before closing large analysis updates, run:
```bash
python analysis_standards.py audit --analysis-root analysis --strict
```

If legacy folders are missing implication lines, agents may backfill placeholders:
```bash
python analysis_standards.py backfill-implications --analysis-root analysis
```

## Quality Rules (Hard Requirements)
Agents must not present strong claims unless these are explicitly handled:
- Non-circular metrics.
- Claim type matches identification design (descriptive vs causal).
- Out-of-sample validation.
- Uncertainty quantification (CI/bootstrap/permutation/etc).
- Robustness checks.
- Reproducible pipeline and data lineage.

## Source of Truth for Templates
- `standards/analysis_manifest_template.json`
- `standards/templates/analysis_spec_template.md`
- `standards/templates/result_card_template.md`
- `ANALYSIS_STANDARDS.md`

If any instruction file conflicts, follow this precedence:
1. User/developer/system instructions at runtime.
2. `AGENTS.md` (this file).
3. Other static docs in repo.

