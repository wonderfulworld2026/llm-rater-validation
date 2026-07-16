"""
prepare_mohler.py
-----------------
Reshape a downloaded Mohler ("Texas") short-answer dataset into the 5-column
schema this project expects, and write it to data/mohler.csv.

The Mohler dataset is distributed in a few slightly different layouts depending
on the mirror. So this is a TEMPLATE: download a copy, open it, look at the
actual column names, and set the four variables in the CONFIG block below to
match. Everything else is handled for you.

Where to get the data (any one of these):
  * Rada Mihalcea's dataset page (University of Michigan / UNT) — search
    "Rada Mihalcea short answer grading dataset".
  * GitHub mirrors — search GitHub for "Mohler short answer grading csv";
    several repos host a cleaned CSV with columns like question / desired
    answer / student answer / score.

The raw data has TWO human grader scores per answer. The community standard is
to use their AVERAGE as the gold human score, which is what we do here.
"""

import pandas as pd

# ============================ CONFIG — EDIT THESE ============================
RAW_PATH = "data/mohler_raw.csv"   # the file you downloaded

# Set these to the ACTUAL column names in your downloaded file:
COL_QUESTION   = "question"          # the question text
COL_REFERENCE  = "desired_answer"    # the reference / model answer
COL_STUDENT    = "student_answer"    # the student's response
COL_GRADER_1   = "score_me"          # first human grader's score
COL_GRADER_2   = "score_other"       # second human grader's score
# If your file already has a single averaged score column instead of two,
# set COL_GRADER_2 = None and point COL_GRADER_1 at that averaged column.
# ============================================================================

OUT_PATH = "data/mohler.csv"


def main():
    raw = pd.read_csv(RAW_PATH)
    print(f"Raw columns found: {list(raw.columns)}")

    out = pd.DataFrame()
    out["id"] = [f"M{i:04d}" for i in range(len(raw))]
    out["question"] = raw[COL_QUESTION].astype(str)
    out["reference_answer"] = raw[COL_REFERENCE].astype(str)
    out["student_answer"] = raw[COL_STUDENT].astype(str)

    if COL_GRADER_2:
        g1 = pd.to_numeric(raw[COL_GRADER_1], errors="coerce")
        g2 = pd.to_numeric(raw[COL_GRADER_2], errors="coerce")
        out["human_score"] = (g1 + g2) / 2.0
    else:
        out["human_score"] = pd.to_numeric(raw[COL_GRADER_1], errors="coerce")

    out = out.dropna(subset=["human_score"])
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(out)} rows in the project schema.")
    print("Now run:  python run_analysis.py --data data/mohler.csv --rater claude --n 150")


if __name__ == "__main__":
    main()
