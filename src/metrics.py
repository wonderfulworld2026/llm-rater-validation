"""
metrics.py
----------
How well do the machine scores agree with the human scores? These are the
standard rater-agreement measures from the automated-scoring literature — the
same ones you'd use to validate a second human rater.

  * Quadratic Weighted Kappa (QWK): the headline metric in automated essay/
    short-answer scoring (it's the ASAP competition metric). 1.0 = perfect
    agreement, 0 = chance, negative = worse than chance. QWK penalizes big
    disagreements more than small ones (hence "quadratic").
  * Pearson / Spearman correlation: linear and rank association.
  * Exact agreement: % of answers where machine == human (rounded).
  * Adjacent agreement: % within 1 point — the usual "close enough" bar.
  * MAE: average absolute point gap.

`by_group` recomputes these within subgroups (e.g., short vs long answers, or
per question) so you can show WHERE the LLM rater holds up and where it slips —
that conditional story is the interesting part of a validation.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

MIN_SCORE = 0
MAX_SCORE = 5
LABELS = list(range(MIN_SCORE, MAX_SCORE + 1))


def _round_int(a):
    return np.clip(np.round(np.asarray(a, dtype=float)), MIN_SCORE, MAX_SCORE).astype(int)


def agreement_metrics(human, machine) -> dict:
    """Compute all agreement metrics for one pair of score vectors.

    `human` may be fractional (averaged human grades); correlations use the raw
    values, while kappa/agreement use rounded integers (kappa needs categories).
    """
    human = np.asarray(human, dtype=float)
    machine = np.asarray(machine, dtype=float)

    h_int = _round_int(human)
    m_int = _round_int(machine)

    # QWK. Fixing `labels` guards against a subgroup that happens to use only
    # part of the 0-5 range.
    qwk = cohen_kappa_score(h_int, m_int, weights="quadratic", labels=LABELS)

    # Correlations need variance in both vectors; guard the degenerate case.
    if np.std(human) == 0 or np.std(machine) == 0:
        pearson = spearman = float("nan")
    else:
        pearson = pearsonr(human, machine)[0]
        spearman = spearmanr(human, machine)[0]

    exact = float(np.mean(h_int == m_int))
    adjacent = float(np.mean(np.abs(h_int - m_int) <= 1))
    mae = mean_absolute_error(human, machine)

    return {
        "n": int(len(human)),
        "qwk": round(float(qwk), 3),
        "pearson_r": round(float(pearson), 3),
        "spearman_r": round(float(spearman), 3),
        "exact_agreement": round(exact, 3),
        "adjacent_agreement": round(adjacent, 3),
        "mae": round(float(mae), 3),
    }


def by_group(df: pd.DataFrame, group_col: str,
             human_col: str = "human_score", machine_col: str = "machine_score",
             min_n: int = 8) -> pd.DataFrame:
    """Agreement metrics computed within each level of `group_col`.

    Groups smaller than `min_n` are skipped (kappa is noisy on tiny samples).
    Returns a tidy data.frame, one row per group.
    """
    rows = []
    for level, grp in df.groupby(group_col):
        if len(grp) < min_n:
            continue
        m = agreement_metrics(grp[human_col], grp[machine_col])
        m = {group_col: level, **m}
        rows.append(m)
    return pd.DataFrame(rows)


def add_length_bins(df: pd.DataFrame, answer_col: str = "student_answer",
                    n_bins: int = 3) -> pd.DataFrame:
    """Add a 'length_bin' column (word count bucketed into short/medium/long).

    Used for the robustness slice: does the LLM rater agree with humans equally
    well on terse vs verbose answers?
    """
    df = df.copy()
    wc = df[answer_col].astype(str).str.split().apply(len)
    labels = ["short", "medium", "long"][:n_bins]
    try:
        df["length_bin"] = pd.qcut(wc, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        # Too few distinct lengths to bin; fall back to a single bucket.
        df["length_bin"] = "all"
    return df
