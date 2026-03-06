# Result Card: Orthogonality Scaling Law at 7B+

- Analysis ID: `orthogonality-scale`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-06T01:00:55+00:00`

## Scores
- Investor Signal Score: `86.00` / 100
- Research Depth Score: `94.00` / 100
- Quality Gate Pass: `True`

## Hypothesis
- The null interaction between PR expansion surgery and Gaussian jitter stress observed at sub-3B scale (exp-012/013) generalizes to 7B+ models across all four paradigms (transformer, SSM, hybrid, reasoning). Pre-specified SESOI: interaction OR within [0.90, 1.11], target ROPE fraction >= 0.95.

## Implications
- Business: Interaction detected at 7B+ scale (OR=0.92, p=0.013) — orthogonality between geometric and dynamic axes may not generalize to larger models. Deployment testing at scale requires joint geometric-dynamic validation, not independent axis testing.
- Scientific: Scale transition reveals negative interaction between PR surgery and jitter stress in 7B+ transformers (mean=-0.026) that was absent at sub-3B scale. Reasoning models show no interaction (-0.003). Scale-moderation meta-regression shows a non-significant negative trend (slope=-0.018, p=0.30). Incomplete paradigm coverage (missing SSM/hybrid) limits generalization.

## Identification and Checks
- Identification Strategy: Focused 2x2 factorial at mid-layer (0.5) with fixed strengths (surgery=0.08, jitter=0.08). 7 seeds for precision. 64 short-answer prompts (32 cal + 32 holdout). Interaction tested via clustered logistic regression, bootstrap (5000), permutation (5000), LOO, and LOPO. Scale-moderation via linear regression of interaction on log10(params_b) pooled across exp-012/013/014.
- Falsification Tests:
  - Interaction OR outside ROPE [0.90, 1.11] at 7B+ scale would falsify scale generalization
  - Reasoning-only interaction significantly different from zero would indicate paradigm-specific coupling
  - Scale-moderation slope significantly different from zero (p<0.05) would indicate scale-dependent interaction
  - Calibration-holdout gap >5pp would indicate prompt-level overfitting
- Robustness Checks:
  - Paradigm-specific interaction analysis (transformer vs SSM vs hybrid vs reasoning)
  - Reasoning-only interaction estimate
  - LOO and LOPO influence analysis
  - Calibration vs holdout split comparison
  - Scale-moderation meta-regression with prior experiments
  - Main effects analysis (surgery and jitter independently)
- Uncertainty Methods:
  - Bootstrap CIs (5000 resamples over model-level means)
  - Permutation null distribution (5000 shuffles of surgery/jitter labels within model)
  - Leave-one-model-out sensitivity analysis
  - Leave-one-paradigm-out stability analysis
  - ROPE equivalence testing with pre-specified bounds [0.90, 1.11]
  - Scale-moderation meta-regression pooling 3 experiments

## Validation
- PASS: manifest and implication checks passed
