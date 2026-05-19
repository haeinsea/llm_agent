import json
import numpy as np
import pandas as pd

def summarize_ndss_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 기본 성능 지표
    ACT_mean = df["ACT"].mean()
    KC_mean = df["K-Concord"].mean()
    SGR_mean = df["SGR"].mean()

    # Top-k Recall
    def top1_ok(row):
        return 1 if row["true_var"] == row["top5"][0] else 0

    def top3_ok(row):
        return 1 if row["true_var"] in row["top5"][:3] else 0

    def top5_ok(row):
        return 1 if row["true_var"] in row["top5"] else 0

    df["Top1"] = df.apply(top1_ok, axis=1)
    df["Top3"] = df.apply(top3_ok, axis=1)
    df["Top5"] = df.apply(top5_ok, axis=1)

    result = {
        "n_attacks": len(df),
        "ACT_mean": float(ACT_mean),
        "KConcord_mean": float(KC_mean),
        "SGR_mean": float(SGR_mean),
        "Top1_recall": float(df["Top1"].mean()),
        "Top3_recall": float(df["Top3"].mean()),
        "Top5_recall": float(df["Top5"].mean()),
        "true_var_distribution": df["true_var"].value_counts().to_dict(),
    }

    print("\n========== NDSS NO-LLM FINAL PERFORMANCE ==========")
    print(f"Attacks evaluated : {result['n_attacks']}")
    print(f"ACT (Hit@5)      : {result['ACT_mean']:.4f}")
    print(f"K-Concord        : {result['KConcord_mean']:.4f}")
    print(f"SGR              : {result['SGR_mean']:.4f}")
    print(f"Top-1 Recall     : {result['Top1_recall']:.4f}")
    print(f"Top-3 Recall     : {result['Top3_recall']:.4f}")
    print(f"Top-5 Recall     : {result['Top5_recall']:.4f}")
    print("\nTrue-var distribution:")
    for k, v in result["true_var_distribution"].items():
        print(f"  {k}: {v}")
    print("=====================================================")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--json", required=True, help="./ndss_reasoning_no_llm.json")

    args = p.parse_args()
    summarize_ndss_results(args.json)
