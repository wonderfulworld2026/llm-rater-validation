"""
run_analysis.py
---------------
The one command that runs the whole study.

  1. Load the data (data/synthetic_sample.csv by default).
  2. Take a stratified sample of N answers (to control cost).
  3. Score each answer with the chosen rater (claude or heuristic).
  4. Compute agreement with the human scores (overall + by subgroup).
  5. Save a scored CSV, a metrics JSON, and figures into results/.

TYPICAL USAGE
  # 1) Free test of the whole pipeline, no API key needed:
  python run_analysis.py --rater heuristic

  # 2) The real study with Claude on ~150 answers:
  python run_analysis.py --data data/mohler.csv --rater claude --n 150

Run `python run_analysis.py --help` to see all options.
"""

from __future__ import annotations
import argparse
import json
import os

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm just draws a progress bar; fall back to identity
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_prep import load_data, sample_data
from src.rater import get_rater
from src.metrics import agreement_metrics, by_group, add_length_bins


def score_dataframe(df: pd.DataFrame, rater) -> pd.DataFrame:
    """Add 'machine_score' and 'machine_rationale' columns by grading each row."""
    scores, rationales = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring with {rater.name}"):
        result = rater.grade(
            question=row["question"],
            reference=row["reference_answer"],
            answer=row["student_answer"],
        )
        scores.append(result.score)
        rationales.append(result.rationale)
    out = df.copy()
    out["machine_score"] = scores
    out["machine_rationale"] = rationales
    return out


def make_figures(df: pd.DataFrame, out_dir: str) -> None:
    """Three simple, screener-friendly figures. Matplotlib only (no seaborn)."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed; write straight to files
    import matplotlib.pyplot as plt
    import numpy as np

    # 1) Human vs machine scatter (jittered so overlapping points are visible).
    fig, ax = plt.subplots(figsize=(5, 5))
    jit = lambda a: np.asarray(a) + np.random.uniform(-0.12, 0.12, size=len(a))
    ax.scatter(jit(df["human_score"]), jit(df["machine_score"]), alpha=0.5, s=30)
    ax.plot([0, 5], [0, 5], "--", color="gray", linewidth=1)
    ax.set_xlabel("Human score"); ax.set_ylabel("Machine score")
    ax.set_title("Human vs. machine scores"); ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.3, 5.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "scatter_human_vs_machine.png"), dpi=150)
    plt.close(fig)

    # 2) Confusion heatmap of rounded scores.
    from sklearn.metrics import confusion_matrix
    labels = list(range(0, 6))
    h = df["human_score"].round().clip(0, 5).astype(int)
    m = df["machine_score"].round().clip(0, 5).astype(int)
    cm = confusion_matrix(h, m, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels); ax.set_yticks(labels)
    ax.set_xlabel("Machine score"); ax.set_ylabel("Human score")
    ax.set_title("Score confusion matrix")
    for i in labels:
        for j in labels:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 3) QWK by answer-length bin (the robustness slice).
    binned = add_length_bins(df)
    grp = by_group(binned, "length_bin", min_n=1)
    if not grp.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(grp["length_bin"].astype(str), grp["qwk"])
        ax.set_ylabel("QWK"); ax.set_ylim(0, 1)
        ax.set_title("Agreement (QWK) by answer length")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "qwk_by_length.png"), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Validate an LLM as a rater for short answers.")
    p.add_argument("--data", default="data/synthetic_sample.csv", help="Path to the input CSV.")
    p.add_argument("--rater", default="heuristic", choices=["claude", "heuristic"],
                   help="Which grader to use. 'heuristic' needs no API key.")
    p.add_argument("--n", type=int, default=None, help="Sample size (default: all rows).")
    p.add_argument("--model", default=None, help="Override the Claude model string.")
    p.add_argument("--out", default="results", help="Output directory.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.data)
    df = sample_data(df, args.n, seed=args.seed)

    rater = get_rater(args.rater, model=args.model) if args.model else get_rater(args.rater)
    scored = score_dataframe(df, rater)

    # --- overall metrics ---
    overall = agreement_metrics(scored["human_score"], scored["machine_score"])

    # --- by-subgroup metrics (length + per question) ---
    binned = add_length_bins(scored)
    length_tbl = by_group(binned, "length_bin")
    question_tbl = by_group(scored, "question")

    # --- save everything ---
    scored_path = os.path.join(args.out, "scored_responses.csv")
    scored.to_csv(scored_path, index=False)

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "rater": rater.name,
            "model": getattr(rater, "model", None),
            "overall": overall,
            "by_length": length_tbl.to_dict(orient="records"),
            "by_question": question_tbl.to_dict(orient="records"),
        }, f, indent=2)

    make_figures(scored, args.out)

    # --- print a readable summary ---
    print("\n" + "=" * 52)
    print(f"RESULTS  (rater = {rater.name}, n = {overall['n']})")
    print("=" * 52)
    print(f"  Quadratic Weighted Kappa : {overall['qwk']}")
    print(f"  Pearson r                : {overall['pearson_r']}")
    print(f"  Spearman r               : {overall['spearman_r']}")
    print(f"  Exact agreement          : {overall['exact_agreement']:.0%}")
    print(f"  Adjacent (within 1)      : {overall['adjacent_agreement']:.0%}")
    print(f"  Mean absolute error      : {overall['mae']}")
    print("-" * 52)
    if not length_tbl.empty:
        print("  QWK by answer length:")
        for _, r in length_tbl.iterrows():
            print(f"    {str(r['length_bin']):8s} n={r['n']:<4d} QWK={r['qwk']}")
    print("=" * 52)
    print(f"\nSaved: {scored_path}\n       {metrics_path}\n       {args.out}/*.png\n")


if __name__ == "__main__":
    main()
