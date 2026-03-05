#!/usr/bin/env python3
"""Quick script to verify which model IDs exist on HuggingFace and get their dates."""

import json
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi

api = HfApi()

CANDIDATES = [
    # Transformers - Meta Llama
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",

    # Transformers - Mistral
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Codestral-25.01-2501",

    # Transformers - Qwen2.5
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-32B-Instruct",

    # Transformers - Google Gemma 2
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",

    # Transformers - Cohere
    "CohereForAI/c4ai-command-r-08-2024",
    "CohereForAI/c4ai-command-r7b-12-2024",
    "CohereForAI/aya-expanse-8b",
    "CohereForAI/aya-expanse-32b",

    # Transformers - InternLM
    "internlm/internlm2_5-7b-chat",
    "internlm/internlm3-8b-instruct",

    # Transformers - Other 2025
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "microsoft/Phi-4",
    "Nexusflow/Athene-V2-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "stabilityai/stablelm-2-12b-chat",
    "baichuan-inc/Baichuan-M1-14B-Base",

    # MoE - Mixture of Experts
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V2.5",
    "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen/Qwen2.5-72B-Instruct",  # dense but large
    "allenai/OLMoE-1B-7B-0924",
    "databricks/dbrx-instruct",
    "Snowflake/snowflake-arctic-instruct",
    "mistralai/Mistral-Large-Instruct-2411",
    "Qwen/Qwen3-235B-A22B",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",

    # Hybrid models
    "ai21labs/AI21-Jamba-1.5-Mini",
    "ai21labs/AI21-Jamba-1.5-Large",
    "ai21labs/Jamba-v0.1",
    "Zyphra/Zamba2-1.2B-instruct",
    "Zyphra/Zamba2-2.7B-instruct",
    "Zyphra/Zamba2-7B-Instruct",
    "togethercomputer/StripedHyena-Nous-7B",
    "togethercomputer/StripedHyena-Hessian-7B",
    "tiiuae/Falcon-H1-34B-Base",
    "nvidia/Hymba-1.5B-Base",
    "nvidia/Nemotron-H-8B-Base-128K",
    "nvidia/Nemotron-H-47B-Instruct-128K",
    "nvidia/Nemotron-H-56B-Instruct-128K",

    # Pure SSM
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-2.8b-hf",
    "state-spaces/mamba2-130m",
    "state-spaces/mamba2-370m",
    "state-spaces/mamba2-780m",
    "state-spaces/mamba2-1.3b",
    "state-spaces/mamba2-2.7b",
    "tiiuae/falcon-mamba-7b-instruct",
    "cartesia-ai/Rene-v0.1-1.3b-pytorch",

    # RWKV
    "RWKV/v5-Eagle-7B-HF",
    "RWKV/v6-Finch-1B6-HF",
    "RWKV/v6-Finch-3B-HF",
    "RWKV/v6-Finch-7B-HF",
    "RWKV/v6-Finch-14B-HF",
    "RWKV/v7-Goose-3B-HF",
    "RWKV/v7-Goose-7B-HF",
    "RWKV/RWKV7-Goose-World3-3B-HF",

    # Reasoning
    "Qwen/QwQ-32B",
    "Qwen/QwQ-32B-Preview",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "AIDC-AI/Marco-o1",
    "allenai/OLMo-2-0325-32B-Instruct",

    # xLSTM
    "NX-AI/xLSTM-1b",
    "NX-AI/xLSTM-2.9b",

    # Diffusion
    "ML-GSAI/LLaDA-8B-Instruct",
    "inclusionAI/LLaDA2.0-8B",

    # RetNet / Alternative
    "microsoft/retnet-3b",
    "Spiral-AI/Spiral-RetNet-3b-base",

    # More 2025 models
    "Writer/Palmyra-X-004-mini",
    "Writer/Palmyra-X-004",
    "nvidia/Llama-3_1-Nemotron-51B-Instruct",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-10B-Instruct",
    "microsoft/phi-4-reasoning-plus",
    "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-3.2-8b-instruct",
    "Zyphra/Zamba3-Instruct-Mini-3.2B",
    "Zyphra/Zamba3-Instruct-Nano-1.3B",
]


def check_model(model_id: str) -> dict:
    try:
        info = api.model_info(model_id)
        created = info.created_at
        downloads = info.downloads or 0
        likes = info.likes or 0
        tags = info.tags or []
        return {
            "model_id": model_id,
            "exists": True,
            "created": created.isoformat() if created else None,
            "year": created.year if created else None,
            "month": created.month if created else None,
            "downloads": downloads,
            "likes": likes,
            "pipeline_tag": info.pipeline_tag,
            "library": info.library_name,
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "exists": False,
            "error": str(e)[:100],
        }


def main():
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check_model, mid): mid for mid in CANDIDATES}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            status = "OK" if r["exists"] else "MISSING"
            date_str = f"{r.get('year', '?')}-{r.get('month', '?'):02d}" if r.get("year") else "?"
            print(f"  [{status}] {r['model_id']}  date={date_str}  dl={r.get('downloads', 0)}")

    results.sort(key=lambda x: (not x["exists"], x["model_id"]))

    print("\n\n=== SUMMARY ===")
    exist = [r for r in results if r["exists"]]
    missing = [r for r in results if not r["exists"]]

    print(f"Total candidates: {len(results)}")
    print(f"Exist: {len(exist)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMISSING models:")
        for r in missing:
            print(f"  - {r['model_id']}: {r.get('error', 'unknown')}")

    # Filter to 2024+ (we'll be flexible with late 2024)
    recent = [r for r in exist if r.get("year") and r["year"] >= 2024]
    print(f"\n2024+ models: {len(recent)}")

    # Save to JSON for later use
    with open("verified_models.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to verified_models.json")


if __name__ == "__main__":
    main()
