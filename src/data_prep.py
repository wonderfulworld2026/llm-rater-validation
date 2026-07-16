"""
data_prep.py
------------
Load a short-answer-grading dataset, check it has the columns we need, and
draw a (cost-controlling) sample for scoring.

R analogy: think of `df` below as a data.frame. pandas is R's data.frame with
slightly different syntax. `df["col"]` is like `df$col`; `df.head()` is like
head(df).

EXPECTED INPUT SCHEMA (a plain .csv with these columns):

    id                a unique id for each student answer
    question          the prompt/question the student answered
    reference_answer  the "model"/correct answer used as a grading reference
    student_answer    the student's actual response (this is what gets graded)
    human_score       the human grade, on a 0-5 scale (may be fractional)

Any real dataset you download just needs to be reshaped into these five
columns (see README for how). The synthetic sample in data/ already matches it.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# The columns every input file must contain.
REQUIRED_COLUMNS = ["id", "question", "reference_answer", "student_answer", "human_score"]

# The grading scale. Mohler and most short-answer sets use 0-5.
MIN_SCORE = 0
MAX_SCORE = 5


def load_data(path: str) -> pd.DataFrame:
    """Read the CSV, validate its columns, and clean obvious problems.

    Raises a clear error if the file is missing a required column, so you find
    schema problems immediately instead of three steps later.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Your data file is missing required column(s): {missing}.\n"
            f"It has: {list(df.columns)}.\n"
            f"Rename/reshape your file to the 5-column schema in data_prep.py."
        )

    # Keep only the columns we use, in a known order.
    df = df[REQUIRED_COLUMNS].copy()

    # Coerce the score to a number; drop rows we can't grade.
    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["student_answer", "human_score"])
    df = df[df["student_answer"].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"[data_prep] Dropped {dropped} row(s) with empty answers or missing scores.")

    # Clip any out-of-range human scores into the expected band.
    df["human_score"] = df["human_score"].clip(MIN_SCORE, MAX_SCORE)

    df = df.reset_index(drop=True)
    print(f"[data_prep] Loaded {len(df)} gradable responses across "
          f"{df['question'].nunique()} question(s).")
    return df


def sample_data(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Return a sample of up to `n` rows, spread across the score range.

    Why stratify: if you random-sample 150 rows and 130 happen to be high
    scores, your agreement metrics are computed on a lopsided slice. Sampling
    evenly across the rounded score keeps the sample representative and your
    kappa honest. If n is None or >= len(df), returns everything.
    """
    if n is None or n >= len(df):
        return df.reset_index(drop=True)

    rounded = df["human_score"].round().astype(int)
    # Proportional allocation across score buckets, then top up any rounding gap.
    frac = n / len(df)
    parts = []
    for score_val, grp in df.groupby(rounded):
        take = max(1, int(round(len(grp) * frac)))
        parts.append(grp.sample(n=min(take, len(grp)), random_state=seed))
    sample = pd.concat(parts)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    sample = sample.sort_values("id").reset_index(drop=True)
    print(f"[data_prep] Sampled {len(sample)} responses (stratified by score) "
          f"from {len(df)} total.")
    return sample
