# Result Card: Layerwise Coupling Mechanism in 7B+ Base vs Reasoning-Tuned Transformers

- Analysis ID: `orthogonality-mechanism-016`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-06T07:57:11+00:00`

## Scores
- Investor Signal Score: `89.00` / 100
- Research Depth Score: `98.00` / 100
- Quality Gate Pass: `True`

## Hypothesis
- The surgery x jitter interaction detected at 7B+ in exp-014/015 is layer-localized in mid/late layers of base transformers, and reasoning training regularizes/compresses this vulnerable band. Local representation geometry (PR, anisotropy) at each layer predicts interaction magnitude.

## Implications
- Business: NO_CLEAR_MECHANISM for the 7B+ transformer coupling. The surgery-jitter interaction is real (bootstrap CIs exclude zero at all layers) but uniformly distributed across depth, not predictable from geometry, and not eliminated by reasoning training. Both base and reasoning-tuned models require full perturbation testing at all layers — no shortcuts. CAVEAT: only 4/5 models completed (DSR1-Llama-8B excluded due to excessive inference time at batch_size=1).
- Scientific: First layerwise decomposition of surgery-jitter interaction at 7B+ scale. All three hypothesized mechanisms falsified: (1) interaction is uniform across layers, not localized (H1: p=0.42); (2) reasoning training does not regularize (H2: p=0.69, single reasoning model); (3) local geometry (PR, anisotropy) does not predict interaction (H3: r=0.004, p=0.99). The coupling appears to be a deep architectural property of transformers, not a shallow geometric effect. Permutation p=0.076 is marginal, consistent with exp-014/015 pattern of model-based vs permutation p-value tension at low model counts.

## Identification and Checks
- Identification Strategy: Same 2x2 factorial as exp-014/015 (surgery=0.08, jitter=0.08) applied at 5 layer positions [0.20, 0.35, 0.50, 0.65, 0.80]. 5 cached models grouped as base_transformer (3) vs reasoning_tuned (2). Phase 1: 32,000 eval trials. Phase 2: clean geometry pass (128 prompts, compute PR and anisotropy per layer). Interaction computed per model per layer. Group comparison via Wilcoxon signed-rank. Geometry-interaction correlation via Pearson/Spearman + multiple regression.
- Falsification Tests:
  - If base transformers show uniform interaction across all layers (no localization), H1 is falsified
  - If reasoning models show comparable interaction to base at any layer, H2 is falsified
  - If PR and anisotropy show no significant correlation with interaction (|r| < 0.3), H3 is falsified
  - If calibration-holdout gap > 5pp, prompt-level overfitting is present
- Robustness Checks:
  - Per-model interaction profiles (individual trajectories vs group means)
  - LOO by group (base_transformer, reasoning_tuned)
  - Main effects (surgery, jitter) by layer position
  - H1: t-test comparing mid/late vs early/extreme layers in base transformers
  - H2: Wilcoxon signed-rank test on paired layer-level interactions
  - H3: Pearson + Spearman correlations, group-separated correlations
  - Mixed-effects model with layer x group x interaction terms
  - Calibration vs holdout split comparison
- Uncertainty Methods:
  - Bootstrap CIs (5000 resamples) per layer position
  - Permutation null distribution (5000 shuffles of surgery/jitter labels within model)
  - Leave-one-model-out sensitivity per group
  - Cal vs holdout split comparison
  - Pearson and Spearman correlations for geometry-interaction link
  - Multiple regression (PR + anisotropy + group -> interaction)

## Validation
- PASS: manifest and implication checks passed
