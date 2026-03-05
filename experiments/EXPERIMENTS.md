# Experiments Log

Reverse chronological order. Only Codex-validated conclusions are listed.

## exp-012: Mechanistic Orthogonality Decomposition
**Status:** Complete
**Date:** 2026-03-05
**Commit:** `0164ae8`
**Script:** `mechanistic_orthogonality_decomposition.py`
**Output:** `analysis/orthogonality_decomposition_012/`
**Config:** 15 models (Hymba failed), 2×2 factorial (PR surgery × jitter stress), 32 prompts, 3 seeds, 5,760 trials
**What we learned:** No detectable interaction between PR expansion surgery and Gaussian jitter stress (interaction OR=1.10, p=0.186, 95% bootstrap CI includes zero, permutation p=0.50, ROPE fraction=0.998). Data are consistent with the orthogonality hypothesis from exp-011. Key model-level observations: Gemma2-2B and Zamba2-1.2B are immune to both interventions; Nemotron-1.5B-R is devastated by surgery but resistant to jitter; Mamba models are largely insensitive to both. LOO analysis confirms stability (std=0.004).

---

## exp-011: Over-Compression + JPIS Stability Bridge — Stage 3b
**Status:** Complete
**Date:** 2026-03-05
**Commit:** `d15b372`
**Script:** `causal_pr_robust.py`
**Output:** `analysis/causal_pr_robust/`
**Config:** 8 models, Stage A coarse sweep (6 layers x 4 strengths x 3 seeds), bridge with JPIS
**What we learned:** Core PR strongly predicts expansion resilience (r=0.905, p=0.002) — high-PR models are immune to dimensional perturbation. BUT expansion gain does NOT predict noise fragility (r=-0.214, p=0.61). These are genuinely orthogonal measurement axes. Mediation path falsified (indirect CI spans zero). This refutes the simple "compression causes fragility" model and supports multi-axis characterization.

---

## exp-010: Jitter-Pressure Inference Stability (JPIS) — Stage 4
**Status:** Complete
**Date:** 2026-03-05
**Commit:** `7788d29`
**Script:** `jitter_pressure_analysis.py`
**Output:** `analysis/jitter_pressure/`
**Config:** 8 models (of 10 attempted), 6 strengths (0.01-0.5), 3 seeds, 5 layer positions, 20,160 total trials
**What we learned:** Dynamic stability under hidden-state perturbation is a new axis orthogonal to representation geometry. Hybrids are the most robust paradigm (mean pressure_auc=0.173), followed by transformers (0.503), then SSMs (0.830). Math is more fragile than factual under pressure (Wilcoxon p=0.008). Gemma2-2B is essentially pressure-proof (auc=0.018); Gemma3-1B is a "glass cannon" (auc=0.924). DSR1-1.5B reasoning model is highly pressure-sensitive (auc=0.972 on math). Mamba-1.4B hung during perturbation (sequential SSM + hook overhead); RWKV7 missing triton dependency.

---

## exp-009: Temporal Causal Analysis
**Status:** Complete
**Date:** 2026-02-25
**Commit:** `88554d0`
**Script:** `temporal_causal_analysis.py`
**Output:** `analysis/temporal_causal/`, `analysis/temporal_causal_relaxed/`
**What we learned:** IV regression shows strong instrument (F=23.27) for framework support as predictor of adoption. Strict analysis: 3 natural experiments, placebo p=0.04. Relaxed: 34 events, 100% pass pretrend test. Temporal dynamics consistent with compatibility tax driving adoption patterns.

---

## exp-008: Measured Compatibility Tax (v2, 129 models)
**Status:** Complete
**Date:** 2026-02-24
**Commit:** `3fe5eb3`
**Script:** `measured_compatibility_tax.py`
**Output:** `analysis/measured_tax_v2/`
**What we learned:** 59% of hybrid adoption penalty is mediated by measured framework compatibility tax (indirect effect=-0.998, CI95=[-1.47, -0.60], p<0.001). No sign flip in mediation (critical validity check). xLSTM has highest tax (0.925), Reasoning has lowest (0.033).

---

## exp-007: Deep Ecosystem Dynamics (v2, 129 models)
**Status:** Complete
**Date:** 2026-02-23
**Commit:** `1a8fe2e`
**Script:** `ecosystem_deep_dynamics.py`
**Output:** `analysis/ecosystem_deep_v2/`
**What we learned:** Convergence index doubled from 0.234 to 0.493 over study period. 16/129 (12.4%) hidden hybrid candidates detected. 35 changepoints, peak at 2025-09-01.

---

## exp-006: Ecosystem Analysis (50 models)
**Status:** Complete
**Date:** 2026-02-22
**Commit:** `1a8fe2e`
**Script:** `ecosystem_analysis.py`
**Output:** `analysis/ecosystem/`
**What we learned:** 40% of models show SSM signals. GQA dominance at 94.4%. Context window inflation (67.6% have 128K+). Architecture landscape rapidly converging.

---

## exp-005: Causal PR Intervention — Stage 3
**Status:** Complete
**Date:** 2026-02-21
**Commit:** `73ed313`
**Script:** `causal_pr_intervention.py`
**Output:** `analysis/causal_pr_intervention/`
**What we learned:** PR expansion causes monotonic accuracy decline in all 4 tested models. OVER-COMPRESSION EVIDENCE: DSR1-1.5B improves from 0.375 to 0.500 accuracy at mild PR expansion, suggesting it has been compressed below optimal dimensionality. Math uses higher-rank subspaces; factual recall is low-rank retrieval-like.

---

## exp-004: Multi-Pair Reasoning Geometry — Stage 2b
**Status:** Complete
**Date:** 2026-02-20
**Commit:** `712c813`
**Script:** `reasoning_pairs_analysis.py`
**Output:** `analysis/reasoning_pairs/`
**What we learned:** Universal reasoning compression: 6/6 matched pairs show PR compression (mean delta=-0.211, Wilcoxon p=0.0156). Holds across 5 training methods: distillation, RL, SFT+alignment. Unanimity is significant (permutation p=0.016), not magnitude (p=0.46). Core CKA preserved (0.9613) — structure maintained despite compression.

---

## exp-003: Reasoning Activation Divergence — Stage 2
**Status:** Complete
**Date:** 2026-02-19
**Commit:** `aa63997`
**Script:** `reasoning_activation_divergence.py`
**Output:** `analysis/reasoning_activation_divergence/`
**What we learned:** DSR1-1.5B diverges from base models only in early layers (max RDI=0.2528 at layer 2). Factual recall maintains PR=1.01 (high-rank), narrative compresses to PR=1.19 (lowest). High CKA with base models (0.948 mean) despite dimensional compression.

---

## exp-002: Representation Geometry Robustness — Stage 1.5
**Status:** Complete
**Date:** 2026-02-18
**Commit:** `f255057`
**Script:** `representation_geometry_robustness.py`
**Output:** `analysis/representation_geometry_robustness/`
**What we learned:** SSM/Transformer intrinsic dimensionality ratio holds under 200-resample bootstrap (p=4.47e-39 on normalized PR/d_model). Anisotropy stable across paradigms. Reasoning models show extreme anisotropy (0.9924).

---

## exp-001: Representation Geometry — Stage 1
**Status:** Complete
**Date:** 2026-02-17
**Commit:** `e3a8ce1`
**Script:** `representation_geometry.py`
**Output:** `analysis/representation_geometry/`
**What we learned:** 5.3x intrinsic dimensionality gap between SSM (PR=11.30) and Transformer (PR=2.13), p=1.08e-40. Hybrids are genuinely intermediate (not just sum of parts). Reasoning distillation compresses to near-1D (DSR1-1.5B PR=1.19). Within-paradigm CKA=0.766 vs cross-paradigm=0.539.
