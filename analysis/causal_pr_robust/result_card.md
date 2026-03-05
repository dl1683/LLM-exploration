# Result Card: Over-Compression + JPIS Stability Bridge

- Analysis ID: `overcompression-bridge`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-05T09:50:58+00:00`

## Scores
- Investor Signal Score: `56.00` / 100
- Research Depth Score: `58.10` / 100
- Quality Gate Pass: `False`

## Hypothesis
- Over-compressed models (low core PR) gain from mild PR expansion, and that expansion gain predicts lower pressure fragility (pressure_auc from JPIS). The mediation path core_pr -> expansion_gain -> pressure_auc links representation geometry to dynamic stability.

## Implications
- Business: Deployment risk requires multi-axis testing: representation geometry (PR) predicts expansion resilience but NOT noise robustness. A model can be geometrically well-structured yet dynamically fragile, or vice versa.
- Scientific: PR dimensionality and dynamic stability are orthogonal axes. Core PR strongly predicts expansion resilience (r=0.905, p=0.002) but does not predict noise fragility. This falsifies simple compression-causes-fragility models.

## Identification and Checks
- Identification Strategy: Interventional: inject orthogonal subspace expansion at varying strengths into hidden states, measure accuracy change. Cross-reference with JPIS perturbation data (Gaussian noise injection). Coarse sweep: 6 layers x 4 strengths x 3 seeds per model.
- Falsification Tests:
  - If expansion gain correlated with pressure_auc, Spearman p < 0.05 (FALSIFIED: p=0.61)
  - If core_pr predicted expansion gain, Spearman p < 0.05 (CONFIRMED: r=0.905, p=0.002)
  - If mediation path existed, indirect effect CI would exclude zero (FALSIFIED: CI=[-0.83, 0.62])
- Robustness Checks:
  - 3 random seeds per expansion condition
  - Multiple layer positions tested (6 coarse layers per model)
  - Both Spearman and Pearson correlations reported
  - Per-domain (math vs factual) correlations checked separately
- Uncertainty Methods:
  - Bootstrap 5000 resamples for correlation CI
  - Permutation test 10000 for correlation p-value
  - Bootstrap 5000 for mediation indirect effect CI

## Validation
- PASS: manifest and implication checks passed
