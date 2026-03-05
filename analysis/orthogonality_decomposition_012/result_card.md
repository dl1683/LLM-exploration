# Result Card: Mechanistic Orthogonality Decomposition

- Analysis ID: `orthogonality-decomp`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-05T19:53:47+00:00`

## Scores
- Investor Signal Score: `80.00` / 100
- Research Depth Score: `85.00` / 100
- Quality Gate Pass: `True`

## Hypothesis
- PR surgery (geometric manipulation) and jitter stress (dynamic perturbation) operate through independent mechanisms. Their combined effect on accuracy should be additive (no interaction), confirming the orthogonality observed in exp-011's correlation analysis.

## Implications
- Business: No detectable interaction between PR surgery and noise stress — deployment testing should treat geometric quality and dynamic stability as separate risk axes. Current evidence supports independent testing protocols rather than a single combined assessment.
- Scientific: 2x2 factorial data are consistent with weak or no interaction between representation geometry manipulation and dynamic perturbation (interaction OR=1.10, 95% CI includes 1, permutation p=0.50, ROPE fraction=0.998). This does not prove mechanistic independence but provides the strongest evidence to date that these measurement axes are not confounded.

## Identification and Checks
- Identification Strategy: Pre-registered 2x2 factorial causal design: Factor A = PR expansion surgery at mid-layer, Factor B = Gaussian jitter stress at mid-layer. 15 models x 32 prompts x 3 seeds x 4 conditions = 5,760 trials. Mixed-effects logistic regression with random intercepts for model and prompt.
- Falsification Tests:
  - If surgery and jitter interacted, the interaction term CI would exclude zero (FALSIFIED: CI=[-0.015, 0.042])
  - If interaction was significant, permutation p < 0.05 (FALSIFIED: p=0.50)
  - If interaction was large, OR outside ROPE [0.8, 1.25] (FALSIFIED: OR=1.10, ROPE fraction=0.998)
- Robustness Checks:
  - LOO stability: std=0.004, most influential model DSR1-1.5B (delta=0.010)
  - ROPE analysis: 99.8% of bootstrap OR within [0.8, 1.25]
  - Consistent across paradigms: all 4 paradigm-level interactions within [-0.04, +0.04]
  - 3 seeds per condition to average over stochastic noise
- Uncertainty Methods:
  - Bootstrap (2000 resamples over models) for interaction CI
  - Permutation test (5000 shuffles) for non-parametric significance
  - BinomialBayesMixedGLM with variational Bayes for mixed-effects uncertainty
  - Leave-one-model-out sensitivity analysis

## Validation
- PASS: manifest and implication checks passed
