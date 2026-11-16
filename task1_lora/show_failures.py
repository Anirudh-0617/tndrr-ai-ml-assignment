"""List fine-tuned model mistakes so you can pick 3 to analyse in the report.

    python show_failures.py

Also prints the most common confusions (gold -> predicted), which usually
point at near-duplicate labels.
"""
import json
from collections import Counter

rows = [json.loads(l) for l in open("outputs/preds_finetuned.jsonl")]
wrong = [r for r in rows if not r["correct"]]
print(f"{len(wrong)} wrong out of {len(rows)}\n")

pairs = Counter((r["gold"], r["pred"]) for r in wrong)
print("Most common confusions:")
for (g, pr), c in pairs.most_common(10):
    print(f"  {c}x  gold={g}  pred={pr}")

print("\nAll mistakes:")
for r in wrong:
    print(f"- TEXT: {r['text']}\n  GOLD: {r['gold']}\n  PRED: {r['pred']}  RAW: {r['raw']!r}\n")
