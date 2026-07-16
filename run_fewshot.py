"""
run_fewshot.py
--------------
Experiment: does adding a few "worked examples" to the grading rubric
(few-shot prompting) improve Claude's agreement with human scores?

This reuses everything in your project. The ONLY change from the normal
Claude run is that the grader is shown three hand-authored anchor answers
(scored 5, 3, and 1) to calibrate the 0-5 scale before it grades. Those
anchors are NOT from the Mohler dataset, so there is no data leakage.

It grades the SAME 150-answer sample as your zero-shot run (same data, same
sample size, same random seed), so the comparison is apples-to-apples.

USAGE (after your normal Claude run already produced results_claude/):
    python run_fewshot.py

It saves to results_claude_fewshot/ and prints a RESULTS block. Compare its
QWK to the 0.673 from your zero-shot results_claude/ run.
"""

from __future__ import annotations
import json
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_prep import load_data, sample_data
from src.rater import ClaudeRater, ScoreResult, _parse_score, build_messages, RUBRIC, DEFAULT_MODEL
from src.metrics import agreement_metrics, by_group, add_length_bins
from run_analysis import make_figures  # reuse the exact same figures

# ----------------------------------------------------------------------------
# The few-shot anchors. Three worked examples that calibrate the 0-5 scale.
# Hand-authored (a hash-table question) so they are NOT in the Mohler test set.
# ----------------------------------------------------------------------------
FEWSHOT_EXAMPLES = [
    {
        "question": "What is the purpose of a hash table?",
        "reference": "A hash table stores key-value pairs and uses a hash function to map keys to array indices, giving near constant-time lookup.",
        "answer": "It stores keys with their values and uses a hash function so you can look things up in roughly constant time.",
        "score": 5,
        "reason": "Captures all key ideas: key-value pairs, hash function, fast (constant-time) lookup.",
    },
    {
        "question": "What is the purpose of a hash table?",
        "reference": "A hash table stores key-value pairs and uses a hash function to map keys to array indices, giving near constant-time lookup.",
        "answer": "It is a structure that lets you store data and find it quickly.",
        "score": 3,
        "reason": "Partially correct (fast retrieval) but omits key-value pairs and the hash-function mechanism.",
    },
    {
        "question": "What is the purpose of a hash table?",
        "reference": "A hash table stores key-value pairs and uses a hash function to map keys to array indices, giving near constant-time lookup.",
        "answer": "It arranges numbers in order from smallest to largest.",
        "score": 1,
        "reason": "On-topic (a data structure) but the described behavior is wrong.",
    },
]

# The exact JSON instruction the normal grader uses, so the ONLY difference
# between zero-shot and few-shot is the presence of the anchor examples.
JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object, no other text, in exactly this form:\n"
    '{"score": <integer 0-5>, "rationale": "<one short sentence>"}'
)


def _fewshot_block() -> str:
    lines = ["\n\nHere are worked examples showing how to apply the rubric:"]
    for ex in FEWSHOT_EXAMPLES:
        lines.append(
            f"\n\nQUESTION: {ex['question']}"
            f"\nREFERENCE ANSWER: {ex['reference']}"
            f"\nSTUDENT ANSWER: {ex['answer']}"
            f"\nCORRECT SCORE: {ex['score']}  (reason: {ex['reason']})"
        )
    return "".join(lines)


FEWSHOT_SYSTEM = RUBRIC + _fewshot_block() + JSON_INSTRUCTION


class FewShotClaudeRater(ClaudeRater):
    """Same as ClaudeRater, but the system prompt includes the anchor examples."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "claude_fewshot"

    def grade(self, question: str, reference: str, answer: str) -> ScoreResult:
        # Same user message as the zero-shot grader; only the system differs.
        _, user = build_messages(question, reference, answer)
        last_err = None
        import time
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    system=FEWSHOT_SYSTEM,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                score, rationale = _parse_score(text)
                return ScoreResult(score, rationale)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Grading failed after {self.max_retries} tries: {last_err}")


def run(rater, data_path="data/mohler.csv", n=150, out_dir="results_claude_fewshot", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_path)
    df = sample_data(df, n, seed=seed)

    scores, rationales = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring with {rater.name}"):
        r = rater.grade(row["question"], row["reference_answer"], row["student_answer"])
        scores.append(r.score)
        rationales.append(r.rationale)
    scored = df.copy()
    scored["machine_score"] = scores
    scored["machine_rationale"] = rationales

    overall = agreement_metrics(scored["human_score"], scored["machine_score"])
    binned = add_length_bins(scored)
    length_tbl = by_group(binned, "length_bin")

    scored.to_csv(os.path.join(out_dir, "scored_responses.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"rater": rater.name, "model": getattr(rater, "model", None),
                   "overall": overall,
                   "by_length": length_tbl.to_dict(orient="records")}, f, indent=2)
    make_figures(scored, out_dir)

    print("\n" + "=" * 52)
    print(f"RESULTS  (rater = {rater.name}, n = {overall['n']})")
    print("=" * 52)
    print(f"  Quadratic Weighted Kappa : {overall['qwk']}")
    print(f"  Pearson r                : {overall['pearson_r']}")
    print(f"  Spearman r               : {overall['spearman_r']}")
    print(f"  Exact agreement          : {overall['exact_agreement']:.0%}")
    print(f"  Adjacent (within 1)      : {overall['adjacent_agreement']:.0%}")
    print(f"  Mean absolute error      : {overall['mae']}")
    if not length_tbl.empty:
        print("-" * 52)
        print("  QWK by answer length:")
        for _, r in length_tbl.iterrows():
            print(f"    {str(r['length_bin']):8s} n={r['n']:<4d} QWK={r['qwk']}")
    print("=" * 52)
    print(f"\nSaved to {out_dir}/\n")


def main():
    run(FewShotClaudeRater())


if __name__ == "__main__":
    main()
