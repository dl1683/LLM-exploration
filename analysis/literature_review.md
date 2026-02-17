# Literature Review: Representation Geometry in LLMs (2025-2026)
## Compiled Feb 17, 2026

### Key Independent Validation
- **Li et al. (NeurIPS 2025)** "Tracing the Representation Geometry of Language Models from Pretraining to Post-training" — RLVR training drives compression-seeking (lower RankMe). Independent corroboration of our reasoning compression finding. They observed it; we show it's harmful.
- **Joshi et al. (NeurIPS 2025)** "Geometry of Decision Making in Language Models" — Final-layer ID correlates with reasoning performance across 28 transformers. Correlation only; we provide causal evidence.

### Methodological References
- **Chun et al. (2025)** "Estimating Dimensionality from Finite Samples" — PR estimators have O(1/P + 1/Q) bias. Corrected estimators available. Could sharpen our measurements.
- **Jha & Reagen (EMNLP 2025)** "Spectral Scaling Laws" — PR ("hard rank") saturates early, Shannon rank ("soft rank") grows with FFN width. eDim metric proposed.

### Our Novelty Claims (confirmed by literature gap)
1. **Cross-architecture PR comparison** — nobody has measured PR across transformer/SSM/hybrid
2. **Causal dimensionality manipulation** — all prior causal work operates on concept directions, not geometric properties
3. **Over-compression as functionally harmful** — observational→causal gap we fill
4. **Systematic matched-pair methodology** across 6 pairs — most rigorous comparative design

### Related but Distinct Work
- **Zhang et al. (2025)** "When Reasoning Meets Compression" — external compression (quant/prune), not intrinsic
- **Lei et al. (2025)** "Revisiting LLM Reasoning via Information Bottleneck" — token-level entropy, not representation geometry
- **Naderi et al. (ICLR 2025)** "Mind the Gap: Spectral Analysis of Rank Collapse" — initialization-time, not training-induced
- **Goomba Lab (2025)** "Tradeoffs of SSMs and Transformers" — conceptual framework, no dimensionality measurements
