# Analysis Standards and Scoring

This repo now supports a two-track analysis standard:

1. Investor-relevant signal generation.
2. Fundamental LLM research depth.

Each analysis should be created and tracked through the same workflow.

## Workflow

1) Initialize analysis scaffolding:

```bash
python analysis_standards.py init \
  --analysis-dir analysis/my_new_study \
  --analysis-id my_new_study_v1 \
  --title "My New Study" \
  --track dual \
  --owner devan
```

2) Fill:
- `analysis/my_new_study/analysis_manifest.json`
- `analysis/my_new_study/key_findings.txt`
  - must include:
    - `business_implication: ...`
    - `scientific_implication: ...`

3) Validate:

```bash
python analysis_standards.py validate --analysis-dir analysis/my_new_study
```

4) Score and register in portfolio scoreboard:

```bash
python analysis_standards.py score \
  --analysis-dir analysis/my_new_study \
  --scoreboard analysis/portfolio_scoreboard.csv \
  --apply-gate-penalty
```

This writes:
- `analysis/my_new_study/result_card.md`
- a new row in `analysis/portfolio_scoreboard.csv`

5) Optional full audit:

```bash
python analysis_standards.py audit --analysis-root analysis --strict
```

## Scoring Model

Two 0-100 scores are computed from 0-5 rubric ratings in the manifest:

- `investor_signal_score`
  - actionability
  - deployment_relevance
  - cost_signal
  - adoption_signal
  - timeliness
  - data_quality
  - identification_strength

- `research_depth_score`
  - novelty
  - theoretical_depth
  - measurement_quality
  - causal_identification
  - robustness
  - reproducibility
  - generality

If `--apply-gate-penalty` is enabled, both scores receive a 30% penalty when quality gates are not all true.

## Required Quality Gates

- `non_circular_metrics`
- `causal_design_matches_claim`
- `out_of_sample_validation`
- `uncertainty_reported`
- `robustness_checks`
- `reproducible_pipeline`

## Templates

- `standards/analysis_manifest_template.json`
- `standards/templates/analysis_spec_template.md`
- `standards/templates/result_card_template.md`

## Agent Integration

To ensure coding agents consistently apply this workflow, repository-level
instruction guides are provided:

- `AGENTS.md` (canonical agent policy)
- `CLAUDE.md` (Claude-specific pointer to `AGENTS.md`)

Agents should enforce the same requirements for all new/updated analyses.
