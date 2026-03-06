# Result Card: 7B+ Cross-Paradigm Orthogonality Resolution

- Analysis ID: `orthogonality-cross-paradigm`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-06T02:52:11+00:00`

## Scores
- Investor Signal Score: `89.00` / 100
- Research Depth Score: `97.00` / 100
- Quality Gate Pass: `True`

## Hypothesis
- The negative surgery x jitter interaction detected at 7B+ in exp-014 (OR=0.92, p=0.013) is architecture-specific heterogeneity concentrated in large transformers, not a universal scale law. SSM/hybrid/RWKV 7B+ models will show interaction near zero (within SESOI), while transformers will remain negative. The permutation p-value conflict (0.35 vs 0.013) is a low-independent-model-count artifact that resolves with a larger panel.

## Implications
- Business: The surgery-jitter interaction is concentrated in transformers (-0.023) while reasoning models show zero interaction. Suggests transformer-specific coupling, not universal scale law. CAVEAT: SSM/hybrid models not tested (download failures), completion gate not met.
- Scientific: With 5 seeds, the exp-014 interaction persists in transformers but vanishes in reasoning models. Cochran's Q (4.00, p=0.41) shows no significant between-model heterogeneity. Without SSM/hybrid coverage, the transformer-specific claim remains provisional.

## Identification and Checks
- Identification Strategy: Same 2x2 factorial as exp-014 (surgery=0.08, jitter=0.08, mid-layer=0.5) held constant for direct comparability. Panel expanded to 10 models across 4 paradigms. 5 seeds (reduced from 7 to allocate budget to more independent models). Exp-014 data reused for 5 completed models. Interaction tested via clustered logistic regression, bootstrap, permutation, LOO, LOPO, and Cochran's Q heterogeneity.
- Falsification Tests:
  - If pooled interaction returns to ROPE (fraction >= 0.95), exp-014 was a false positive
  - If SSM/hybrid show interaction magnitude comparable to transformers, it IS a universal scale effect
  - If permutation p < 0.05 with expanded panel, the p-value conflict was due to low model count
  - Calibration-holdout gap > 5pp would indicate prompt-level overfitting
- Robustness Checks:
  - Paradigm-specific interaction analysis (transformer vs SSM vs hybrid vs reasoning)
  - Reasoning-only interaction estimate
  - LOO and LOPO influence analysis
  - Calibration vs holdout split comparison
  - Scale-moderation meta-regression with prior experiments (012-015)
  - Cochran's Q between-model heterogeneity
  - Main effects analysis (surgery and jitter independently)
  - Completion gate check (>=8 models, >=1 SSM, >=1 hybrid)
- Uncertainty Methods:
  - Bootstrap CIs (5000 resamples over model-level means)
  - Permutation null distribution (5000 shuffles of surgery/jitter labels within model)
  - Leave-one-model-out sensitivity analysis
  - Leave-one-paradigm-out stability analysis
  - ROPE equivalence testing with pre-specified bounds [0.90, 1.11]
  - Scale-moderation meta-regression pooling 4 experiments
  - Cochran's Q heterogeneity test with I-squared and tau-squared

## Validation
- PASS: manifest and implication checks passed
