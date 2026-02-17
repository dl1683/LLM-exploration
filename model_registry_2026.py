"""
Model Registry 2026 - Expanded for Ecosystem Analysis
Updated: February 2026
Policy: Focus on 2024-2026 models. Expanded from 50 to ~130 for statistical power.
"""

# TIER 1 - Fast experiments (all < 3B)
TIER1_2026 = [
    # Transformers - Qwen3
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    # Transformers - Qwen2.5
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    # Transformers - Google
    "google/gemma-3-1b-it",
    "google/gemma-2-2b-it",
    # Transformers - Other
    "HuggingFaceTB/SmolLM3-3B",
    "ibm-granite/granite-4.0-350m",
    "ibm-granite/granite-4.0-1b",
    "tiiuae/Falcon3-1B-Instruct",

    # SSM - Mamba v1
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
    # SSM - Mamba v2
    "state-spaces/mamba2-130m",
    "state-spaces/mamba2-370m",
    "state-spaces/mamba2-780m",
    "state-spaces/mamba2-1.3b",
    "state-spaces/mamba2-2.7b",
    # SSM - Other
    "cartesia-ai/Rene-v0.1-1.3b-pytorch",

    # Hybrid
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Instruct",
    "ibm-granite/granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-1b",
    "Zyphra/Zamba2-1.2B",
    "Zyphra/Zamba2-1.2B-instruct",
    "nvidia/Hymba-1.5B-Base",                     # Mamba + attention hybrid, 1.5B

    # Liquid AI
    "LiquidAI/LFM2-350M-Exp",
    "LiquidAI/LFM2-1.2B-Exp",
    "LiquidAI/LFM2-2.6B-Exp",

    # RWKV
    "RWKV/RWKV7-Goose-World3-1.5B-HF",
    "RWKV/v6-Finch-1B6-HF",

    # MoE
    "Qwen/Qwen1.5-MoE-A2.7B",                    # 14B total / 2.7B active

    # Reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

# TIER 2 - Standard experiments (3B-8B)
TIER2_2026 = [
    # Transformers - Qwen3
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    # Transformers - Qwen2.5
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-Coder-7B",
    # Transformers - Google
    "google/gemma-3-4b-it",
    # Transformers - Microsoft
    "microsoft/Phi-4-mini-instruct",
    # Transformers - Meta
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    # Transformers - Mistral
    "mistralai/Ministral-8B-Instruct-2410",
    # Transformers - Cohere
    "CohereForAI/aya-expanse-8b",
    "CohereForAI/c4ai-command-r7b-12-2024",
    # Transformers - InternLM
    "internlm/internlm2_5-7b-chat",
    "internlm/internlm3-8b-instruct",
    # Transformers - Allen AI
    "allenai/Olmo-3-7B",
    # Transformers - IBM
    "ibm-granite/granite-4.0-micro",
    "ibm-granite/granite-4.0-tiny-preview",
    "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-3.2-8b-instruct",
    # Transformers - TII Falcon3
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",

    # SSM
    "tiiuae/falcon-mamba-7b",
    "tiiuae/falcon-mamba-7b-instruct",

    # Hybrid
    "tiiuae/Falcon-H1-3B-Instruct",
    "tiiuae/Falcon-H1-7B-Instruct",
    "nvidia/Nemotron-H-4B-Instruct-128K",
    "Zyphra/Zamba2-2.7B",
    "Zyphra/Zamba2-2.7B-instruct",
    "Zyphra/Zamba2-7B",
    "Zyphra/Zamba2-7B-Instruct",
    "ibm-granite/granite-4.0-h-tiny",
    "togethercomputer/StripedHyena-Nous-7B",      # Hyena + attention hybrid
    "togethercomputer/StripedHyena-Hessian-7B",   # Hyena + attention hybrid

    # xLSTM
    "NX-AI/xLSTM-7b",

    # RWKV
    "RWKV-Red-Team/ARWKV-R1-7B",
    "RWKV/v5-Eagle-7B-HF",
    "RWKV/v6-Finch-3B-HF",
    "RWKV/v6-Finch-7B-HF",

    # MoE
    "allenai/OLMoE-1B-7B-0924",                   # 7B total / 1B active

    # Reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-mini-reasoning",
    "AIDC-AI/Marco-o1",

    # Diffusion
    "ML-GSAI/LLaDA-8B-Base",

    # RetNet
    "Spiral-AI/Spiral-RetNet-3b-base",
]

# TIER 3 - Validation (8B+, quantize aggressively)
TIER3_2026 = [
    # Transformers - Qwen3
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    # Transformers - Qwen2.5
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    # Transformers - Google
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    # Transformers - Meta
    "meta-llama/Llama-3.3-70B-Instruct",
    # Transformers - Mistral
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # Transformers - Cohere
    "CohereForAI/c4ai-command-r-08-2024",
    "CohereForAI/aya-expanse-32b",
    # Transformers - Other
    "openai/gpt-oss-20b",
    "allenai/Olmo-3-1125-32B",
    "allenai/OLMo-2-0325-32B-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "nvidia/Llama-3_1-Nemotron-51B-Instruct",
    "microsoft/Phi-4",
    "baichuan-inc/Baichuan-M1-14B-Base",
    "tiiuae/Falcon3-10B-Instruct",

    # Hybrid
    "tiiuae/Falcon-H1-34B-Instruct",
    "tiiuae/Falcon-H1-34B-Base",
    "ibm-granite/granite-4.0-h-small",
    "ai21labs/Jamba-v0.1",                         # Mamba + attention + MoE, 52B total
    "ai21labs/AI21-Jamba-1.5-Mini",                # Mamba + attention + MoE, 52B total
    "ai21labs/AI21-Jamba-1.5-Large",               # Mamba + attention + MoE, 398B total

    # RWKV
    "RWKV/v6-Finch-14B-HF",

    # MoE
    "mistralai/Mixtral-8x7B-Instruct-v0.1",       # 47B total / 13B active
    "mistralai/Mixtral-8x22B-Instruct-v0.1",      # 141B total / 39B active
    "mistralai/Mistral-Large-Instruct-2411",       # 123B dense
    "deepseek-ai/DeepSeek-V2.5",                   # 236B total / 21B active
    "deepseek-ai/DeepSeek-V3",                     # 671B total / 37B active
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-235B-A22B",                        # 235B total / 22B active
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",   # 109B total / 17B active, 10M ctx
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",  # 400B+ total / 17B active
    "Snowflake/snowflake-arctic-instruct",          # 480B total / 17B active

    # Reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1",                     # 671B MoE reasoning
    "microsoft/Phi-4-reasoning",
    "microsoft/phi-4-reasoning-plus",
    "Qwen/QwQ-32B",
    "Qwen/QwQ-32B-Preview",

    # Diffusion
    "inclusionAI/LLaDA2.0-mini",
]

# Quick test set for rapid iteration
QUICK_TEST = [
    "Qwen/Qwen3-0.6B",                           # Transformer
    "state-spaces/mamba-370m-hf",                  # SSM (Mamba 1 HF format)
    "tiiuae/Falcon-H1-0.5B-Instruct",             # Hybrid (Mamba+Transformer)
    "LiquidAI/LFM2-1.2B-Exp",                     # Liquid (LIV convolution+GQA)
    "RWKV/RWKV7-Goose-World3-1.5B-HF",            # RWKV
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Reasoning
]

# For phase transition analysis (proven working)
PHASE_TRANSITION_MODELS = [
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Legacy but validated
]

# Paradigm groups (representative samples for quick paradigm-specific tests)
PARADIGMS = {
    "transformer": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen2.5-7B",
        "google/gemma-3-1b-it",
        "google/gemma-2-2b-it",
        "ibm-granite/granite-4.0-1b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "tiiuae/Falcon3-7B-Instruct",
    ],
    "ssm": [
        "state-spaces/mamba-370m-hf",
        "state-spaces/mamba-790m-hf",
        "state-spaces/mamba-1.4b-hf",
        "state-spaces/mamba-2.8b-hf",
        "state-spaces/mamba2-1.3b",
        "state-spaces/mamba2-2.7b",
        "tiiuae/falcon-mamba-7b",
        "cartesia-ai/Rene-v0.1-1.3b-pytorch",
    ],
    "hybrid": [
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "tiiuae/Falcon-H1-1.5B-Instruct",
        "tiiuae/Falcon-H1-7B-Instruct",
        "ibm-granite/granite-4.0-h-1b",
        "Zyphra/Zamba2-1.2B",
        "Zyphra/Zamba2-2.7B",
        "nvidia/Hymba-1.5B-Base",
        "togethercomputer/StripedHyena-Nous-7B",
        "ai21labs/AI21-Jamba-1.5-Mini",
    ],
    "liquid": [
        "LiquidAI/LFM2-350M-Exp",
        "LiquidAI/LFM2-1.2B-Exp",
        "LiquidAI/LFM2-2.6B-Exp",
    ],
    "xlstm": [
        "NX-AI/xLSTM-7b",
    ],
    "rwkv": [
        "RWKV/RWKV7-Goose-World3-1.5B-HF",
        "RWKV/v5-Eagle-7B-HF",
        "RWKV/v6-Finch-1B6-HF",
        "RWKV/v6-Finch-7B-HF",
    ],
    "moe": [
        "Qwen/Qwen1.5-MoE-A2.7B",
        "allenai/OLMoE-1B-7B-0924",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-235B-A22B",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ],
    "reasoning": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "microsoft/Phi-4-mini-reasoning",
        "Qwen/QwQ-32B",
        "deepseek-ai/DeepSeek-R1",
    ],
    "retnet": [
        "Spiral-AI/Spiral-RetNet-3b-base",
    ],
}


def get_tier1_models():
    """Get all Tier 1 models for fast experiments."""
    return TIER1_2026


def get_quick_test_models():
    """Get quick test set covering all paradigms."""
    return QUICK_TEST


def get_models_by_paradigm(paradigm: str):
    """Get models for a specific paradigm."""
    return PARADIGMS.get(paradigm, [])


def get_all_paradigms():
    """Get list of all paradigms."""
    return list(PARADIGMS.keys())
