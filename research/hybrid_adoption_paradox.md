# Hybrid Adoption Paradox: Why SSM+Attention Models Have Lower Downloads

## Key Finding
Hybridization score has -0.584 partial correlation with downloads (strongest signal in our data).
Pure transformers: +1.49 log-scale premium. Hybrids: -0.78. RWKV: -2.66.

## Root Causes (Ranked by Impact)

### 1. TOOLING GAP (Very High Impact)
- **vLLM V0**: Treated hybrids as "experimental hacks". KV cache architecture assumed every layer needs it. Mamba layers broke this. Fixed in V1 (late 2025) but legacy perception persists.
- **llama.cpp**: Mamba was CPU-only initially. GGUF conversions required re-conversion between versions.
- **TGI**: Config mismatches (model_type "mamba" vs "ssm"). TGI entered maintenance mode Dec 2025.
- **llm-compressor**: Open issue - does NOT fully support hybrid structures (Bamba, Nemotron-H, Zamba2).

### 2. QUANTIZATION BARRIER (Very High Impact)
- Standard PTQ (GPTQ, AWQ) **fails catastrophically on SSM layers**
- Static W8A8 on Mamba-2.8B: 78.63 perplexity (vs 9.45 FP16) = 8.3x degradation
- Causal recurrence propagates quantization errors through entire sequence
- Specialized methods (Quamba, Quamba2) exist but are research-stage
- LoRA adapters difficult to apply to SSM layers (prefix scan memory layout)

### 3. RETRIEVAL WEAKNESS (High Impact)
- Removing attention from hybrid: retrieval accuracy drops to **literally 0%**
- Pure Mamba fails on copy tasks beyond ~500 tokens
- Single attention layer at layer 4 restores perfect copying
- Most users run short-context tasks where transformers are 1.8x faster

### 4. SPECULATIVE DECODING INCOMPATIBILITY (Medium Impact)
- Hidden state backtracking impossible (states discarded after each update)
- Tree-based parallel verification impossible with sequential SSM processing
- Research solutions only (SpecMamba on FPGA)

### 5. ECOSYSTEM LOCK-IN (Medium Impact)
- 1M+ transformer checkpoints on HuggingFace
- 6+ years of transformer tuning recipes
- PEFT/LoRA works seamlessly with transformers, buggy with Mamba
- Community tutorials/courses all assume transformer architecture

## Why Labs Still Invest
- **IBM**: Enterprise cost reduction. Granite 4.0's 9:1 ratio cuts memory 70%. ISO 42001 certified.
- **NVIDIA**: Shows their hardware excels at both. Nemotron-H 3x faster than Llama-3.1-8B.
- **TII**: "High-density reasoning" - Falcon-H1R 7B beats Qwen3-32B on math (73.96% vs 63.66%).
- **Microsoft**: Phi-4-mini-flash 10x throughput, 2-3x latency reduction.

Common thread: optimizing for **inference cost at enterprise scale**, not HuggingFace downloads.

## Sources
- vLLM RFC #17140, PyTorch Blog on hybrid models
- Mamba-PTQ (ICML 2024), Q-Mamba, Quamba, Quamba2
- Goomba Lab tradeoffs analysis
- Falcon-H1R blog, NVIDIA Nemotron-H, IBM Granite 4.0
- SpecMamba (arxiv 2509.19873)
