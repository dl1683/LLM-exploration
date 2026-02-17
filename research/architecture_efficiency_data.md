# Architecture Efficiency Data: Transformer vs SSM vs Hybrid

## Inference Throughput (tokens/second)

| Model | Type | Config | Throughput | vs Transformer |
|-------|------|--------|-----------|---------------|
| Falcon-H1R 7B | Hybrid | 512→32K, batch 64 | ~1500 tok/s/GPU | **2x** vs Qwen3-8B |
| Hymba-1.5B | Hybrid | 8K, batch 128 | 2696 tok/s | **3.5x** vs Llama-3.2-3B |
| Nemotron-H 56B | Hybrid | Long context | - | **2.4-2.9x** vs Llama-3.1-70B |
| Phi-4-flash | Hybrid | 2K+32K gen | - | **10x** vs Phi-4-mini |
| Bamba-9B | Hybrid | vLLM | - | **2.5x throughput, 2x latency** |
| Pure Mamba | SSM | General | - | **4-5x** (no KV cache) |

## Memory Efficiency (24GB GPU)

| Model | Architecture | Max Tokens |
|-------|-------------|------------|
| Mamba-790M | Pure SSM | **220,000** |
| Llama-3.2-1B | Transformer | 65,536 |
| Qwen2.5-0.5B | Transformer | 57,344 |

SSMs process ~4x longer sequences on same GPU.

## Cache Size (8K context, FP16)

| Model | Cache Size |
|-------|-----------|
| Hymba-1.5B (hybrid) | 79 MB |
| Qwen2.5-1.5B (transformer) | 229 MB |
| Llama-3.2-3B (transformer) | 918 MB |

## Quantization Sensitivity (Mamba-2.8B, WikiText2 PPL)

| Method | Bits | Perplexity | vs FP16 (9.45) |
|--------|------|-----------|----------------|
| Static W8A8 (naive) | 8 | 78.63 | **+731%** |
| Quamba W8A8 | 8 | 9.91 | +4.9% |
| Quamba2 | 4 | ~9.55 | ~+1% |

## Training MFU (H100, 8B params)

| Architecture | MFU |
|-------------|-----|
| Transformer (FA2) | 30.7% |
| Mamba-2-Hybrid | 29.9% |
| Pure Mamba-1 | 10-15% |

## Optimal Hybrid Ratios (from ablations)

| Model | Attention:Mamba Ratio | Notes |
|-------|----------------------|-------|
| Jamba | 1:7 (12.5%) | Production-proven |
| Nemotron-H | ~1:12 (8%) | NVIDIA optimized |
| Granite 4.0 | 1:9 (~10%) | IBM enterprise |
| **NVIDIA sweet spot** | **7-8%** | Ablation-determined |

Going below 5% attention degrades in-context learning.
Going above 15% wastes compute.

## Scaling Laws
- SSMs need ~3x more training tokens for knowledge tasks (MMLU)
- On perplexity, SSMs match with equal or fewer tokens
- Bamba-9B matched Llama-3.1-8B with 7x less data (data quality matters)
- At byte-level (no tokenization), SSMs scale substantially better

## Phase Transitions
- 0% attention: Fails at associative recall/copy >500 tokens
- 7-8% attention: **Phase transition** - ICL capability jumps dramatically
- 12.5%: Robust across all tasks
- 50%+: Diminishing returns

## Speed Crossover
- Short sequences (<4K): Transformers 1.8x faster
- ~4-8K tokens: Crossover point
- Long sequences (>57K): SSMs up to 4x faster

## Sources
- TransMamba (arxiv 2503.24067), Jamba (ICLR 2025)
- Falcon-H1R (arxiv 2601.02346), Hymba (arxiv 2411.13676)
- Nemotron-H (arxiv 2504.03624), Bamba (HF blog)
- Mamba-2 (Princeton), Quamba (arxiv 2410.13229)
- Goomba Lab tradeoffs, SSM Long Context (arxiv 2507.12442)
