"""
make_synthetic.py
-----------------
Writes a tiny synthetic dataset (data/synthetic_sample.csv) in the exact schema
the pipeline expects. This exists ONLY so you can test the plumbing end-to-end
before you download the real dataset or spend any API credits.

It is NOT research data — the "human_score" here is fabricated to correlate with
answer quality so the agreement metrics come out sensible. Replace it with a
real dataset (see README) for anything you'd actually show.
"""

import random
import pandas as pd

random.seed(7)

# Each item: a question, a reference answer, and answers at descending quality.
ITEMS = [
    {
        "question": "What is the purpose of a stack data structure?",
        "reference": "A stack stores elements in last in first out order using push and pop operations.",
        "answers": [
            ("A stack keeps elements in last in first out order using push and pop.", 5),
            ("It stores items and you push and pop them in last in first out order.", 5),
            ("A stack uses push and pop to add and remove elements from the top.", 4),
            ("It stores elements and removes the most recent one first.", 3),
            ("A stack holds data you can push onto.", 2),
            ("It is a kind of list.", 1),
            ("It sorts numbers from low to high.", 0),
        ],
    },
    {
        "question": "Why do we use indexing in a database?",
        "reference": "An index speeds up data retrieval by letting the database find rows without scanning the whole table.",
        "answers": [
            ("An index speeds up retrieval so the database finds rows without scanning the whole table.", 5),
            ("Indexing lets the database locate rows faster instead of scanning every row.", 5),
            ("It makes queries faster by helping the database find data quickly.", 4),
            ("Indexes help you search the database more quickly.", 3),
            ("It organizes the database.", 2),
            ("So the data looks nicer.", 0),
            ("It deletes duplicate rows.", 1),
        ],
    },
    {
        "question": "What does recursion mean in programming?",
        "reference": "Recursion is when a function calls itself to solve smaller instances of the same problem until a base case stops it.",
        "answers": [
            ("Recursion is when a function calls itself on smaller instances until a base case stops it.", 5),
            ("A function that calls itself to solve smaller versions of the problem with a base case.", 5),
            ("It is when a function calls itself repeatedly to break a problem down.", 4),
            ("A function calling itself.", 3),
            ("Looping over data many times.", 1),
            ("Recursion means the program crashes.", 0),
            ("It is a way functions repeat smaller problems.", 3),
        ],
    },
    {
        "question": "What is the role of a compiler?",
        "reference": "A compiler translates source code written in a high level language into machine code the computer can run.",
        "answers": [
            ("A compiler translates high level source code into machine code the computer can run.", 5),
            ("It converts source code in a high level language into machine code.", 5),
            ("A compiler turns your written code into something the machine executes.", 4),
            ("It changes code into machine code.", 3),
            ("It checks your code for errors.", 2),
            ("A compiler runs the program.", 1),
            ("It stores files on disk.", 0),
        ],
    },
    {
        "question": "What is the difference between RAM and a hard drive?",
        "reference": "RAM is fast volatile memory used while programs run, while a hard drive is slower nonvolatile storage that keeps data after power off.",
        "answers": [
            ("RAM is fast volatile memory used while programs run; a hard drive is slower nonvolatile storage that keeps data after power off.", 5),
            ("RAM is temporary fast memory; a hard drive stores data permanently even without power.", 5),
            ("RAM is used while running programs and is fast, the hard drive stores things long term.", 4),
            ("RAM is memory and a hard drive stores files.", 3),
            ("One is faster than the other.", 1),
            ("They are both the CPU.", 0),
            ("RAM holds running data, disk keeps it after shutdown.", 4),
        ],
    },
]


def jitter(score: int) -> float:
    """Add mild noise so human scores aren't a perfect function of quality."""
    val = score + random.choice([-0.5, 0, 0, 0, 0.5])
    return max(0, min(5, round(val * 2) / 2))  # keep on a .5 grid, 0-5


rows = []
rid = 1
for item in ITEMS:
    for answer_text, quality in item["answers"]:
        rows.append({
            "id": f"S{rid:03d}",
            "question": item["question"],
            "reference_answer": item["reference"],
            "student_answer": answer_text,
            "human_score": jitter(quality),
        })
        rid += 1

df = pd.DataFrame(rows)
df.to_csv("data/synthetic_sample.csv", index=False)
print(f"Wrote data/synthetic_sample.csv with {len(df)} rows.")
