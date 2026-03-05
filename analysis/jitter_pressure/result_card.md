# Result Card: Jitter-Pressure Inference Stability (JPIS)

- Analysis ID: `jpis-stage4`
- Track: `dual`
- Owner: `claude`
- Status: `complete`
- Evidence Strength: `causal`
- Scored At (UTC): `2026-03-05T05:24:09+00:00`

## Scores
- Investor Signal Score: `58.10` / 100
- Research Depth Score: `57.40` / 100
- Quality Gate Pass: `False`

## Hypothesis
- Models across different architectures (Transformer, SSM, Hybrid, Reasoning) exhibit different dynamic stability profiles under hidden-state perturbation, and this stability axis is orthogonal to representation geometry metrics (PR, anisotropy).

## Implications
- Business: Pressure sensitivity reveals deployment risk: models appearing equivalent on benchmarks may have vastly different robustness to inference-time noise (quantization drift, temperature, stochastic decoding). Hybrid architectures are the most robust paradigm.
- Scientific: Dynamic stability under perturbation is a new axis orthogonal to representation geometry. 'Compressed but stable' vs 'compact but hypersensitive' cannot be predicted from PR alone. Math is more fragile than factual under pressure (p=0.008).

## Identification and Checks
- Identification Strategy: Interventional: inject calibrated Gaussian noise into hidden states at specific layers, measure output-level effects (accuracy, flip rate, edit distance, entropy, logit margin, KL divergence). Dose-response design across 6 perturbation strengths.
- Falsification Tests:
  - If perturbation had no functional effect, flip rates would be 0 at all strengths
  - If all architectures responded identically, Kruskal-Wallis p would be >0.5
  - If math and factual shared the same stability profile, Wilcoxon p would be >0.5
- Robustness Checks:
  - Multiple layer positions tested (5 per model at 10%, 25%, 50%, 75%, 90%)
  - 3 random seeds for noise direction (full-config models)
  - Consistent patterns across transformers with different baselines (Qwen3, Gemma3, Gemma2)
- Uncertainty Methods:
  - Bootstrap 2000 resamples for model-level CIs on max-strength accuracy
  - Multiple seeds (3) for noise direction averaging (first 4 models)
  - Wilcoxon signed-rank for paired math vs factual comparison

## Validation
- PASS: manifest and implication checks passed
