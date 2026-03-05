# Result Card: Orthogonality Grid (Expanded Factorial)

- Analysis ID: `orthogonality-grid`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-05T22:37:12+00:00`

## Scores
- Investor Signal Score: `82.00` / 100
- Research Depth Score: `92.00` / 100
- Quality Gate Pass: `True`

## Hypothesis
- The null interaction between PR expansion surgery and Gaussian jitter stress observed in exp-012 (single layer, single strength) generalizes across multiple layer positions (0.2/0.5/0.8), perturbation strengths (4x4 grid), and task domains (math/factual/logic/hard). Pre-specified SESOI: interaction OR within [0.90, 1.11].

## Implications
- Business: Orthogonality between geometric (PR) and dynamic (noise) axes holds across multiple layer positions, perturbation strengths, and task families. Deployment risk requires independent testing on both axes.
- Scientific: Expanded grid factorial (3 layers x 3x3 strengths x 48 prompts, batched) confirms no detectable interaction between PR surgery and jitter stress. Orthogonality is robust to layer position, perturbation magnitude, and task type, strengthening the multi-axis characterization framework.

## Identification and Checks
- Identification Strategy: Expanded 2x2 factorial within a 3x3 strength grid across 3 layer positions. PR expansion surgery (orthogonal noise injection into low-variance dimensions) is Factor A; Gaussian jitter stress (scaled noise proportional to activation norm) is Factor B. Interaction tested via mixed-effects logistic regression with cluster-robust SEs by model. Bootstrap (5000 resamples) and permutation (5000 shuffles) for non-parametric validation.
- Falsification Tests:
  - Interaction OR outside ROPE [0.90, 1.11] at any layer position would falsify orthogonality at that operating point
  - Systematic calibration-holdout divergence (>5pp accuracy gap) would indicate prompt-level overfitting
  - LOO analysis showing any single model moves interaction outside ROPE would indicate fragile conclusion
  - Layer-specific interaction tests: if interaction emerges only at specific layers, orthogonality is partial not global
- Robustness Checks:
  - Multi-layer replication (0.2/0.5/0.8 of total layers)
  - Multi-strength replication (4x4 grid from zero to moderate)
  - Multi-domain replication (math, factual, logic, hard)
  - Calibration vs holdout split comparison
  - Paradigm-specific interaction analysis (transformer vs SSM vs hybrid vs reasoning)
  - LOO influence analysis
- Uncertainty Methods:
  - Bootstrap CIs (5000 resamples over model-level means)
  - Permutation null distribution (5000 shuffles of surgery/jitter labels within model)
  - Leave-one-model-out sensitivity analysis
  - ROPE equivalence testing with pre-specified bounds [0.90, 1.11]
  - Confidence scoring on answer parsing (exact=1.0, substring=0.5)

## Validation
- PASS: manifest and implication checks passed
