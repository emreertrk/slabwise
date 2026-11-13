import csv
import statistics
from pathlib import Path

GRADE_TO_COND = {
    10: "NM", 9: "LP", 8: "LP", 7: "MP", 6: "MP",
    5: "HP", 4: "HP", 3: "DMG", 2: "DMG", 1: "DMG",
}

def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]

by_cond = {}
with open("outputs/condition_ratios.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        by_cond.setdefault(r["condition"], []).append(float(r["ratio"]))

rows = []
for grade in range(10, 0, -1):
    cond = GRADE_TO_COND[grade]
    if cond == "NM":
        rows.append({"grade": grade, "condition": cond, "multiplier": 1.0, "p25": 1.0, "p75": 1.0, "n": 0})
        continue
    vals = by_cond[cond]
    rows.append({
        "grade": grade,
        "condition": cond,
        "multiplier": round(statistics.median(vals), 3),
        "p25": round(pct(vals, 0.25), 3),
        "p75": round(pct(vals, 0.75), 3),
        "n": len(vals),
    })

Path("outputs").mkdir(exist_ok=True)
with open("outputs/multiplier_table.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(r)
