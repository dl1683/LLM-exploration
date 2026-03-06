# Experiments Log

Reverse chronological order. Only Codex-validated conclusions are listed.

## exp-016: Layerwise Coupling Mechanism at 7B+
**Status:** Complete
**Date:** 2026-03-06
**Commit:** `9592c74`
**Script:** `orthogonality_mechanism_016.py`
**Output:** `analysis/orthogonality_mechanism_016/`
**Config:** 4 models (3 base_transformer: Qwen2.5-7B, Qwen3-8B, OLMo2-7B; 1 reasoning_tuned: DSR1-7B; DSR1-Llama-8B excluded due to 0.2 t/s inference), 2×2 factorial at 5 layer positions [0.20, 0.35, 0.50, 0.65, 0.80], surgery=0.08, jitter=0.08, 5 seeds, 64 prompts, 25,600 trials + 25 geometry measurements.
**What we learned:** NO_CLEAR_MECHANISM. All three mechanistic hypotheses falsified: (1) Interaction is uniform across layers, not localized (H1: t=-0.204, p=0.42); (2) Reasoning training does not regularize (H2: Wilcoxon p=0.69, single reasoning model); (3) Local geometry (PR, anisotropy) does not predict interaction (H3: r=0.004, p=0.99). Bootstrap CIs exclude zero at ALL layers — the coupling is real but unexplained. Permutation p=0.076 (marginal). Cal-holdout gap 1.5pp. Mixed-effects: no significant layer×interaction (p=0.19) or group×interaction (p=0.60). The surgery-jitter coupling at 7B+ appears to be a deep architectural property of transformers, not a shallow geometric or layer-specific effect. Investor=89, Research=98.

---

## exp-015: 7B+ Cross-Paradigm Orthogonality Resolution
**Status:** Complete
**Date:** 2026-03-06
**Commit:** `ef33cec`
**Script:** `orthogonality_cross_paradigm_015.py`
**Output:** `analysis/orthogonality_cross_paradigm_015/`
**Config:** 5 models (3 transformer, 2 reasoning; SSM/hybrid downloads all failed), 2×2 factorial at mid-layer, surgery=0.08, jitter=0.08, 5 seeds, 64 prompts, 6,400 trials. Reused exp-014 data for 5 completed models. Completion gate NOT met (missing SSM/hybrid).
**What we learned:** TRANSFORMER-SPECIFIC COUPLING confirmed. The exp-014 interaction is concentrated in transformers (mean=-0.023) while reasoning models show zero interaction (-0.000). Mixed-effects OR=0.94, p=0.017. Bootstrap CI [-0.023, -0.003] excludes zero. Permutation p=0.53 still conflicts (low model count). Cochran's Q=4.00 (p=0.41, I²=0%) shows no significant between-model heterogeneity. Verdict: NOT a universal scale law — it's architecture-specific. CAVEAT: SSM/hybrid paradigms could not be tested due to persistent HuggingFace download failures; the claim is provisional. Investor=89, Research=97.

---

## exp-014: Orthogonality Scaling Law at 7B+
**Status:** Complete
**Date:** 2026-03-05
**Commit:** `0f6c68f`
**Script:** `orthogonality_scale_014.py`
**Output:** `analysis/orthogonality_scale_014/`
**Config:** 5 models (3 transformer, 2 reasoning; ARWKV failed deps, Zamba2/FalconMamba stalled download, FalconH1 hung), 2×2 factorial at mid-layer, surgery=0.08, jitter=0.08, 7 seeds, 64 prompts, 8,960 trials
**What we learned:** INTERACTION DETECTED AT 7B+ SCALE. Orthogonality breaks: OR=0.92, p=0.013, bootstrap CI [-0.029, -0.006] excludes zero. The effect is driven by transformers (mean interaction=-0.026) not reasoning models (-0.003). Surgery has a much stronger main effect at 7B+ (-0.149 accuracy drop vs negligible at sub-3B). Scale-moderation meta-regression shows a non-significant negative trend (slope=-0.018, p=0.30). Caveat: only 5 models, missing SSM/hybrid paradigms. Permutation p=0.35 (not significant) conflicts with model-based p=0.013.

---

## exp-013: Orthogonality Grid (Expanded Factorial)
**Status:** Complete
**Date:** 2026-03-05
**Commit:** `9352dbb`
**Script:** `orthogonality_grid_013.py`
**Output:** `analysis/orthogonality_grid_013/`
**Config:** 12 models (DSR1-1.5B & Nemotron OOM), 3×3 strength grid (surgery [0,0.06,0.12] × jitter [0,0.04,0.08]), 3 layer positions (0.2/0.5/0.8), 48 prompts, 3 seeds, 46,656 trials
**What we learned:** Orthogonality between PR surgery and jitter stress is robust across the full operating grid — no detectable interaction (OR=1.02, p=0.53, bootstrap CI [-0.008, 0.021], permutation p=0.58). Holds at all layer positions, all strength combinations, and all task domains. ROPE fraction=0.82 (below 0.95 target but consistent with null). Calibration-holdout divergence <2pp. LOO std=0.002. Missing reasoning paradigm models due to CUDA OOM cascade. Three paradigms confirmed: transformer, SSM, hybrid.

---

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
