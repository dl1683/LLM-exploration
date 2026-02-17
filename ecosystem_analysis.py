#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PARADIGM_ORDER = [
    "transformer",
    "ssm",
    "hybrid",
    "liquid",
    "xlstm",
    "rwkv",
    "reasoning",
    "diffusion",
    "unassigned",
]

SUSPICIOUS_PRETRAIN_VALUES = {
    256, 512, 1024, 2000, 2048, 4096, 8192, 16384, 32768, 38912, 65536, 131072
}


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_payload(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "models" in payload:
        return payload, payload["models"]
    if isinstance(payload, list):
        return {"model_count": len(payload)}, payload
    raise ValueError("Unsupported JSON format")


def flatten_model(m: dict) -> dict:
    registry = m.get("registry") or {}
    ident = m.get("identity") or {}
    arch = m.get("architecture") or {}
    ps = m.get("paradigm_specific") or {}
    train = m.get("training") or {}
    quant = m.get("quantization") or {}
    src = m.get("sources") or {}

    detected = ps.get("detected_numeric_fields") or {}
    detected_keys = [str(k).lower() for k in detected.keys()]
    has_mamba_key = any(k.startswith("mamba") for k in detected_keys)
    has_ssm_key = any("ssm" in k for k in detected_keys)

    qformats = quant.get("available_formats") or []

    return {
        "model_id": m.get("model_id"),
        "org": m.get("org"),
        "family": registry.get("family"),
        "variant": registry.get("variant"),
        "tier": registry.get("tier"),
        "registry_paradigm": registry.get("paradigm"),
        "is_quick_test": registry.get("is_quick_test"),
        "is_phase_transition": registry.get("is_phase_transition"),
        "release_date": ident.get("release_date"),
        "license": ident.get("license"),
        "downloads": ident.get("downloads"),
        "likes": ident.get("likes"),
        "num_params": arch.get("num_params"),
        "num_layers": arch.get("num_layers"),
        "hidden_size": arch.get("hidden_size"),
        "ffn_size": arch.get("ffn_size"),
        "num_heads": arch.get("num_heads"),
        "num_kv_heads": arch.get("num_kv_heads"),
        "head_dim": arch.get("head_dim"),
        "norm_type": arch.get("norm_type"),
        "activation": arch.get("activation"),
        "positional_scheme": arch.get("positional_scheme"),
        "max_context": arch.get("max_context"),
        "sliding_window": arch.get("sliding_window"),
        "vocab_size": arch.get("vocab_size"),
        "num_params_source": arch.get("num_params_source"),
        "ssm_state_size": ps.get("ssm_state_size"),
        "ssm_conv_kernel": ps.get("ssm_conv_kernel"),
        "ssm_dt_rank": ps.get("ssm_dt_rank"),
        "detected_numeric_keys": "|".join(sorted(detected_keys)),
        "detected_numeric_count": len(detected_keys),
        "has_mamba_key": has_mamba_key,
        "has_ssm_key": has_ssm_key,
        "pretrain_tokens": train.get("pretrain_tokens"),
        "pretrain_tokens_source": train.get("pretrain_tokens_source"),
        "tokenizer_type": train.get("tokenizer_type"),
        "quant_format_count": len(qformats),
        "quant_formats": "|".join(qformats),
        "hf_api": src.get("hf_api"),
        "has_config_json": src.get("config_json") is not None,
        "has_model_card": src.get("model_card") is not None,
        "has_safetensors_index": src.get("safetensors_index") is not None,
        "error_count": len(m.get("errors") or []),
        "has_errors": len(m.get("errors") or []) > 0,
    }


def infer_paradigm(row: pd.Series) -> str:
    model_id = str(row.get("model_id", "")).lower()
    org = str(row.get("org", "")).lower()
    family = str(row.get("family", "")).lower()
    reg = str(row.get("registry_paradigm", "")).lower()

    if "llada" in model_id or "diffusion" in family:
        return "diffusion"
    if "rwkv" in model_id:
        return "rwkv"
    if "xlstm" in model_id:
        return "xlstm"
    if "reasoning" in model_id or "r1-distill" in model_id:
        return "reasoning"
    if "lfm" in model_id or "liquid" in org or reg == "liquid":
        return "liquid"

    has_ssm_signals = bool(row.get("has_ssm_signals", False))
    name_ssm_hint = any(
        x in model_id for x in ["mamba", "zamba", "falcon-h1", "nemotron-h"]
    )
    has_attention = pd.notna(row.get("num_heads"))

    if has_ssm_signals or name_ssm_hint:
        return "hybrid" if has_attention else "ssm"

    if reg in {"transformer", "ssm", "hybrid", "liquid", "xlstm", "rwkv", "reasoning", "diffusion"}:
        return reg
    return "transformer"


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "downloads", "likes", "num_params", "num_layers", "hidden_size", "ffn_size",
        "num_heads", "num_kv_heads", "head_dim", "max_context", "sliding_window",
        "vocab_size", "ssm_state_size", "ssm_conv_kernel", "ssm_dt_rank",
        "pretrain_tokens", "detected_numeric_count", "quant_format_count", "error_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_month"] = df["release_date"].dt.to_period("M").dt.to_timestamp()
    df["release_year"] = df["release_date"].dt.year

    df["has_ssm_signals"] = (
        df["ssm_state_size"].notna()
        | df["ssm_conv_kernel"].notna()
        | df["has_mamba_key"].fillna(False)
        | df["has_ssm_key"].fillna(False)
    )

    df["param_b"] = df["num_params"] / 1e9
    df["context_k"] = df["max_context"] / 1024
    df["log10_params"] = np.log10(df["num_params"])
    df["log10_context"] = np.log10(df["max_context"])
    df["log10_downloads"] = np.log10(df["downloads"] + 1)

    df["gqa_ratio"] = df["num_heads"] / df["num_kv_heads"]
    df["kv_compression"] = 1 - (df["num_kv_heads"] / df["num_heads"])
    df["is_gqa"] = np.where(df["gqa_ratio"].notna(), (df["gqa_ratio"] > 1).astype(float), np.nan)
    df["is_mha"] = np.where(df["gqa_ratio"].notna(), np.isclose(df["gqa_ratio"], 1.0).astype(float), np.nan)

    df["ffn_expansion"] = df["ffn_size"] / df["hidden_size"]
    df["params_per_layer_m"] = df["num_params"] / df["num_layers"] / 1e6
    df["params_per_head_m"] = df["num_params"] / df["num_heads"] / 1e6
    df["depth_width_ratio"] = df["num_layers"] / df["hidden_size"]

    df["context_per_param"] = df["max_context"] / df["num_params"]
    df["context_per_param_b"] = df["max_context"] / df["param_b"]
    df["downloads_per_param_b"] = df["downloads"] / df["param_b"]

    df["is_long_context_128k"] = df["max_context"] >= 128000
    df["is_extreme_context"] = df["max_context"] >= 1_000_000
    df["has_sliding_window"] = df["sliding_window"].notna()
    df["sliding_window_ratio"] = df["sliding_window"] / df["max_context"]

    df["pretrain_tokens_suspect"] = (
        df["pretrain_tokens"].notna()
        & (
            (df["pretrain_tokens"] < 1_000_000_000)
            | df["pretrain_tokens"].isin(SUSPICIOUS_PRETRAIN_VALUES)
        )
    )
    df["pretrain_tokens_clean"] = np.where(df["pretrain_tokens_suspect"], np.nan, df["pretrain_tokens"])
    df["tokens_per_param"] = df["pretrain_tokens_clean"] / df["num_params"]

    df["has_quant"] = df["quant_format_count"] > 0
    df["has_gguf"] = df["quant_formats"].str.contains("GGUF", na=False)
    df["has_bitsandbytes"] = df["quant_formats"].str.contains("BITSANDBYTES", na=False)
    df["has_int8"] = df["quant_formats"].str.contains("INT8", na=False)

    df["canonical_head_dim"] = df["head_dim"].isin([64, 128])
    df["canonical_norm"] = df["norm_type"].eq("rmsnorm")
    df["canonical_activation"] = df["activation"].eq("silu")
    df["canonical_position"] = df["positional_scheme"].eq("rope")
    df["convergence_score"] = (
        df[["canonical_head_dim", "canonical_norm", "canonical_activation", "canonical_position"]]
        .astype(float)
        .mean(axis=1)
    )

    completeness_fields = [
        "release_date", "license", "downloads", "num_params", "num_layers", "hidden_size",
        "ffn_size", "num_heads", "num_kv_heads", "max_context", "vocab_size",
        "pretrain_tokens", "tokenizer_type",
    ]
    df["metadata_completeness"] = df[completeness_fields].notna().mean(axis=1)

    df["inferred_paradigm"] = df.apply(infer_paradigm, axis=1)
    df["taxonomy_gap"] = (df["registry_paradigm"] == "unassigned") & (df["inferred_paradigm"] != "transformer")
    df["explicit_label_disagreement"] = (
        df["registry_paradigm"].ne("unassigned")
        & df["registry_paradigm"].ne(df["inferred_paradigm"])
    )

    df["size_bucket"] = pd.cut(
        df["param_b"],
        bins=[-np.inf, 1, 3, 10, np.inf],
        labels=["sub-1B", "1-3B", "3-10B", "10B+"],
    )
    df["context_bucket"] = pd.cut(
        df["max_context"],
        bins=[-np.inf, 8192, 32768, 131072, np.inf],
        labels=["<=8k", "8k-32k", "32k-131k", ">131k"],
    )
    return df


def pareto_frontier_min_x_max_y(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    work = frame[["model_id", "inferred_paradigm", x, y]].dropna().sort_values(by=x, ascending=True)
    if work.empty:
        return work
    frontier_rows = []
    best_y = -np.inf
    for _, row in work.iterrows():
        if row[y] > best_y:
            frontier_rows.append(row)
            best_y = row[y]
    return pd.DataFrame(frontier_rows)


def save_tables(df: pd.DataFrame, out_tables: Path) -> dict:
    out_tables.mkdir(parents=True, exist_ok=True)

    registry_counts = (
        df["registry_paradigm"].value_counts(dropna=False)
        .rename_axis("paradigm")
        .reset_index(name="count")
    )
    inferred_counts = (
        df["inferred_paradigm"].value_counts(dropna=False)
        .rename_axis("paradigm")
        .reset_index(name="count")
    )
    registry_counts.to_csv(out_tables / "registry_paradigm_counts.csv", index=False)
    inferred_counts.to_csv(out_tables / "inferred_paradigm_counts.csv", index=False)

    taxonomy_matrix = pd.crosstab(df["registry_paradigm"], df["inferred_paradigm"]).reindex(
        index=PARADIGM_ORDER, columns=PARADIGM_ORDER, fill_value=0
    )
    taxonomy_matrix.to_csv(out_tables / "registry_vs_inferred_matrix.csv")

    tier_gap = (
        df.assign(
            is_unassigned=df["registry_paradigm"].eq("unassigned"),
            hidden_non_transformer=lambda d: d["registry_paradigm"].eq("unassigned")
            & d["inferred_paradigm"].ne("transformer"),
        )
        .groupby("tier", dropna=False)
        .agg(
            model_count=("model_id", "count"),
            unassigned_count=("is_unassigned", "sum"),
            unassigned_share=("is_unassigned", "mean"),
            hidden_non_transformer_count=("hidden_non_transformer", "sum"),
            hidden_non_transformer_share=("hidden_non_transformer", "mean"),
        )
        .reset_index()
    )
    tier_gap.to_csv(out_tables / "tier_assignment_gap.csv", index=False)

    paradigm_stats = (
        df.groupby("inferred_paradigm", dropna=False)
        .agg(
            model_count=("model_id", "count"),
            median_param_b=("param_b", "median"),
            median_context_k=("context_k", "median"),
            median_gqa_ratio=("gqa_ratio", "median"),
            gqa_rate=("is_gqa", "mean"),
            ssm_signal_rate=("has_ssm_signals", "mean"),
            quantized_rate=("has_quant", "mean"),
            median_downloads=("downloads", "median"),
            median_downloads_per_param_b=("downloads_per_param_b", "median"),
            mean_convergence_score=("convergence_score", "mean"),
            mean_metadata_completeness=("metadata_completeness", "mean"),
            suspect_token_rate=("pretrain_tokens_suspect", "mean"),
            long_context_rate=("is_long_context_128k", "mean"),
        )
        .reset_index()
    )
    paradigm_stats.to_csv(out_tables / "paradigm_comparison_metrics.csv", index=False)

    monthly = (
        df.dropna(subset=["release_month"])
        .groupby(["release_month", "inferred_paradigm"])
        .size()
        .reset_index(name="count")
    )
    monthly.to_csv(out_tables / "monthly_release_trends.csv", index=False)

    coverage_fields = [
        "release_date", "license", "downloads", "likes", "num_params", "num_layers",
        "hidden_size", "ffn_size", "num_heads", "num_kv_heads", "head_dim",
        "max_context", "sliding_window", "vocab_size", "pretrain_tokens", "tokenizer_type",
    ]
    coverage = pd.DataFrame({
        "field": coverage_fields,
        "coverage_rate": [df[c].notna().mean() for c in coverage_fields],
    })
    coverage.to_csv(out_tables / "metadata_coverage.csv", index=False)

    frontier_downloads = pareto_frontier_min_x_max_y(df, "param_b", "downloads")
    frontier_context = pareto_frontier_min_x_max_y(df, "param_b", "max_context")
    frontier_downloads.to_csv(out_tables / "pareto_downloads_vs_params.csv", index=False)
    frontier_context.to_csv(out_tables / "pareto_context_vs_params.csv", index=False)

    df.to_csv(out_tables / "model_level_derived_metrics.csv", index=False)

    return {
        "registry_counts": registry_counts,
        "inferred_counts": inferred_counts,
        "taxonomy_matrix": taxonomy_matrix,
        "tier_gap": tier_gap,
        "paradigm_stats": paradigm_stats,
        "monthly": monthly,
        "frontier_downloads": frontier_downloads,
        "frontier_context": frontier_context,
    }


def save_figures(df: pd.DataFrame, tables: dict, out_figs: Path) -> None:
    out_figs.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    counts = (
        pd.concat(
            [
                tables["registry_counts"].set_index("paradigm")["count"].rename("registry"),
                tables["inferred_counts"].set_index("paradigm")["count"].rename("inferred"),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )
    counts_long = counts.melt(id_vars="paradigm", var_name="label_type", value_name="count")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=counts_long, x="paradigm", y="count", hue="label_type", order=PARADIGM_ORDER)
    plt.xticks(rotation=30, ha="right")
    plt.title("Registry vs Inferred Paradigm Counts")
    plt.tight_layout()
    plt.savefig(out_figs / "01_paradigm_counts_registry_vs_inferred.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.heatmap(tables["taxonomy_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title("Registry Paradigm vs Inferred Paradigm")
    plt.tight_layout()
    plt.savefig(out_figs / "02_registry_vs_inferred_heatmap.png", dpi=220)
    plt.close()

    tier_plot = tables["tier_gap"].melt(
        id_vars="tier",
        value_vars=["unassigned_share", "hidden_non_transformer_share"],
        var_name="metric",
        value_name="share",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=tier_plot, x="tier", y="share", hue="metric")
    plt.ylim(0, 1)
    plt.title("Assignment Debt by Tier")
    plt.tight_layout()
    plt.savefig(out_figs / "03_tier_assignment_gap.png", dpi=220)
    plt.close()

    scatter_df = df.dropna(subset=["param_b", "max_context"]).copy()
    scatter_df["marker_size"] = np.clip(np.log10(scatter_df["downloads"].fillna(1) + 1) * 22, 25, 280)
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=scatter_df,
        x="param_b",
        y="max_context",
        hue="inferred_paradigm",
        size="marker_size",
        sizes=(20, 320),
        alpha=0.85,
        legend="brief",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Parameters (billions, log scale)")
    plt.ylabel("Max Context (tokens, log scale)")
    plt.title("Parameter Scale vs Context Capacity")
    outliers = scatter_df.nlargest(6, "context_per_param_b")
    for _, row in outliers.iterrows():
        plt.annotate(short_name(row["model_id"]), (row["param_b"], row["max_context"]), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_figs / "04_params_vs_context_scatter.png", dpi=220)
    plt.close()

    gqa_df = df.dropna(subset=["gqa_ratio"]).copy()
    if not gqa_df.empty:
        gqa_order = [p for p in PARADIGM_ORDER if p in set(gqa_df["inferred_paradigm"])]
        plt.figure(figsize=(12, 5))
        sns.violinplot(data=gqa_df, x="inferred_paradigm", y="gqa_ratio", order=gqa_order, inner="quartile", cut=0)
        plt.xticks(rotation=30, ha="right")
        plt.title("GQA Ratio Distribution by Inferred Paradigm")
        plt.tight_layout()
        plt.savefig(out_figs / "05_gqa_ratio_violin.png", dpi=220)
        plt.close()

    eff_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["downloads_per_param_b"])
    if not eff_df.empty:
        eff_order = [p for p in PARADIGM_ORDER if p in set(eff_df["inferred_paradigm"])]
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=eff_df, x="inferred_paradigm", y="downloads_per_param_b", order=eff_order)
        plt.yscale("log")
        plt.xticks(rotation=30, ha="right")
        plt.title("Downloads per Parameter by Inferred Paradigm")
        plt.tight_layout()
        plt.savefig(out_figs / "06_downloads_per_param_boxplot.png", dpi=220)
        plt.close()

    monthly = tables["monthly"]
    if not monthly.empty:
        monthly_pivot = (
            monthly.pivot(index="release_month", columns="inferred_paradigm", values="count")
            .fillna(0)
            .reindex(columns=[p for p in PARADIGM_ORDER if p in set(monthly["inferred_paradigm"])], fill_value=0)
        )
        ax = monthly_pivot.plot.area(figsize=(12, 6), colormap="tab20")
        ax.set_title("Monthly Model Releases by Inferred Paradigm")
        ax.set_xlabel("Release Month")
        ax.set_ylabel("Model Count")
        plt.tight_layout()
        plt.savefig(out_figs / "07_monthly_release_trends.png", dpi=220)
        plt.close()

    feature_cols = [
        "median_param_b", "median_context_k", "median_gqa_ratio", "ssm_signal_rate",
        "quantized_rate", "mean_convergence_score", "mean_metadata_completeness", "suspect_token_rate",
    ]
    feat = tables["paradigm_stats"].set_index("inferred_paradigm")[feature_cols]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(feat.median(numeric_only=True))
    feat_norm = (feat - feat.mean()) / feat.std(ddof=0).replace(0, np.nan)
    feat_norm = feat_norm.fillna(0.0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(feat_norm, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Normalized Cross-Paradigm Feature Profile")
    plt.tight_layout()
    plt.savefig(out_figs / "08_paradigm_feature_heatmap.png", dpi=220)
    plt.close()

    frontier = tables["frontier_downloads"].sort_values("param_b")
    frontier_base = df.dropna(subset=["param_b", "downloads"])
    if not frontier_base.empty:
        plt.figure(figsize=(11, 7))
        sns.scatterplot(
            data=frontier_base,
            x="param_b",
            y="downloads",
            hue="inferred_paradigm",
            alpha=0.7,
        )
        if not frontier.empty:
            plt.plot(frontier["param_b"], frontier["downloads"], color="black", linewidth=2.5, label="Pareto Frontier")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Parameters (billions, log scale)")
        plt.ylabel("Downloads (log scale)")
        plt.title("Adoption Efficiency Frontier (Downloads vs Parameters)")
        plt.tight_layout()
        plt.savefig(out_figs / "09_adoption_pareto_frontier.png", dpi=220)
        plt.close()


def write_key_findings(df: pd.DataFrame, tables: dict, out_path: Path) -> None:
    total = len(df)
    unassigned = int((df["registry_paradigm"] == "unassigned").sum())
    inferred_non_transformer_unassigned = int(
        ((df["registry_paradigm"] == "unassigned") & (df["inferred_paradigm"] != "transformer")).sum()
    )
    ssm_signals = int(df["has_ssm_signals"].sum())
    ssm_in_unassigned = int(((df["registry_paradigm"] == "unassigned") & df["has_ssm_signals"]).sum())
    context_known = int(df["max_context"].notna().sum())
    long_context = int((df["max_context"] >= 128000).sum())
    gqa_known = int(df["gqa_ratio"].notna().sum())
    gqa_models = int((df["gqa_ratio"] > 1).sum())
    pretrain_known = int(df["pretrain_tokens"].notna().sum())
    pretrain_suspect = int(df["pretrain_tokens_suspect"].sum())
    quantized = int(df["has_quant"].sum())
    gguf = int(df["has_gguf"].sum())

    lines = [
        f"models_total: {total}",
        f"registry_unassigned: {unassigned} ({unassigned / total:.1%})",
        f"unassigned_but_inferred_non_transformer: {inferred_non_transformer_unassigned}",
        f"models_with_ssm_signals: {ssm_signals} ({ssm_signals / total:.1%})",
        f"ssm_signals_inside_unassigned: {ssm_in_unassigned}",
        f"context_known: {context_known}",
        f"context_ge_128k: {long_context} ({(long_context / context_known) if context_known else 0:.1%} of known)",
        f"gqa_known: {gqa_known}",
        f"gqa_ratio_gt_1: {gqa_models} ({(gqa_models / gqa_known) if gqa_known else 0:.1%} of known)",
        f"pretrain_tokens_known: {pretrain_known}",
        f"pretrain_tokens_suspect: {pretrain_suspect} ({(pretrain_suspect / pretrain_known) if pretrain_known else 0:.1%} of known)",
        f"quantized_models: {quantized}",
        f"gguf_models: {gguf}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-paradigm ecosystem analysis for metadata_enriched.json")
    parser.add_argument("--input", default="metadata_enriched.json", help="Path to metadata_enriched.json")
    parser.add_argument("--outdir", default="analysis/ecosystem", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    out_tables = outdir / "tables"
    out_figs = outdir / "figures"

    meta, models = load_payload(input_path)
    df = pd.DataFrame([flatten_model(m) for m in models])
    df = add_derived_metrics(df)

    tables = save_tables(df, out_tables)
    save_figures(df, tables, out_figs)
    write_key_findings(df, tables, outdir / "key_findings.txt")

    print(f"Analyzed {len(df)} models.")
    print(f"Tables: {out_tables}")
    print(f"Figures: {out_figs}")
    print(f"Key findings: {outdir / 'key_findings.txt'}")
    if isinstance(meta, dict):
        print(f"source_file: {meta.get('source_file')}, generated_at_utc: {meta.get('generated_at_utc')}")


if __name__ == "__main__":
    main()
