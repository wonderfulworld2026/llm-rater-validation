"""
rater.py
--------
Turns a (question, reference_answer, student_answer) into a 0-5 score.

Two raters are provided:

  * ClaudeRater    -> asks Claude to grade against a rubric (the real study).
  * HeuristicRater -> a no-API baseline that scores by word overlap with the
                      reference answer. Needs no key and no network, so you can
                      (a) test the whole pipeline for free, and (b) report it as
                      a naive baseline the LLM should beat.

Design choice: the grader NEVER sees the human_score. It only sees what a real
rater would see (question, reference, student answer). That keeps the
comparison honest.
"""

from __future__ import annotations
import os
import re
import json
import time
from dataclasses import dataclass

MIN_SCORE = 0
MAX_SCORE = 5

# Default Claude model for grading. Haiku is the cheapest tier and is plenty
# for rubric scoring, which keeps you well under your monthly cap. If this
# model string is ever out of date, check https://docs.claude.com for the
# current Haiku model name and change it here (or pass --model on the CLI).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# The rubric the model must apply. Editing this prompt is the main lever you'll
# tune on Saturday: clearer rubric -> better agreement with humans.
RUBRIC = """You are grading a student's short answer against a reference answer.

Score the student answer from 0 to 5 using this rubric:
5 = fully correct and complete; matches the key ideas of the reference answer.
4 = correct with a minor omission or imprecision.
3 = partially correct; captures some key ideas but misses others.
2 = mostly incorrect; only a small relevant fragment.
1 = incorrect but on-topic / shows minimal relevant content.
0 = blank, irrelevant, or entirely wrong.

Judge the MEANING, not the wording. A correct answer phrased differently from
the reference still earns full marks. Do not reward length for its own sake."""


def _clip_int(x: float) -> int:
    """Round to an int and force it into the 0-5 band."""
    return int(max(MIN_SCORE, min(MAX_SCORE, round(x))))


def build_messages(question: str, reference: str, answer: str):
    """Build the system + user messages sent to Claude.

    We ask for JSON only so the score is easy to parse. `rationale` is optional
    but useful: it lets you eyeball WHY the model gave a score when you audit
    disagreements.
    """
    system = (
        RUBRIC
        + "\n\nRespond with ONLY a JSON object, no other text, in exactly this form:\n"
        '{"score": <integer 0-5>, "rationale": "<one short sentence>"}'
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"STUDENT ANSWER:\n{answer}\n\n"
        "Grade the student answer."
    )
    return system, user


def _parse_score(text: str) -> tuple[int, str]:
    """Pull the score (and rationale if present) out of the model's reply.

    Tries JSON first; if the model wrapped it in prose, falls back to the first
    0-5 integer we can find. Returns (score, rationale).
    """
    text = text.strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return _clip_int(float(obj["score"])), str(obj.get("rationale", ""))
    except Exception:
        m = re.search(r"[0-5]", text)
        if m:
            return int(m.group()), text[:120]
        raise ValueError(f"Could not parse a 0-5 score from model reply: {text[:200]!r}")


@dataclass
class ScoreResult:
    score: int
    rationale: str = ""


class HeuristicRater:
    """No-API baseline: score = 5 * (fraction of reference words the answer covers).

    This is deliberately dumb. Its job is to (1) let you test the pipeline with
    zero cost and (2) give you a baseline number the LLM should clearly beat —
    a small but real piece of validity evidence.
    """

    name = "heuristic"

    @staticmethod
    def _tokens(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(s).lower()))

    def grade(self, question: str, reference: str, answer: str) -> ScoreResult:
        ref = self._tokens(reference)
        ans = self._tokens(answer)
        if not ref:
            return ScoreResult(0, "no reference tokens")
        overlap = len(ref & ans) / len(ref)
        return ScoreResult(_clip_int(5 * overlap), f"overlap={overlap:.2f}")


class ClaudeRater:
    """Grades with Claude via the Anthropic API.

    The `anthropic` package and an ANTHROPIC_API_KEY are only needed here, and
    only when you actually run the real study — they're imported lazily so the
    heuristic path works without them.
    """

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 max_retries: int = 3):
        try:
            import anthropic  # lazy import
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package isn't installed. Run:\n"
                "    pip install anthropic\n"
                "Or use --rater heuristic to test without the API."
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key found. Set it first:\n"
                "  Windows PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
                "  macOS/Linux:         export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_retries = max_retries

    def grade(self, question: str, reference: str, answer: str) -> ScoreResult:
        system, user = build_messages(question, reference, answer)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
                score, rationale = _parse_score(text)
                return ScoreResult(score, rationale)
            except Exception as e:  # network hiccup, rate limit, parse fail
                last_err = e
                time.sleep(1.5 * (attempt + 1))  # simple backoff
        raise RuntimeError(f"Grading failed after {self.max_retries} tries: {last_err}")


def get_rater(name: str, model: str = DEFAULT_MODEL):
    """Factory: 'claude' or 'heuristic'."""
    name = name.lower()
    if name == "claude":
        return ClaudeRater(model=model)
    if name == "heuristic":
        return HeuristicRater()
    raise ValueError(f"Unknown rater '{name}'. Use 'claude' or 'heuristic'.")
