import csv
import statistics
from pathlib import Path

BANDS = [(0, 2, "T1_<2"), (2, 5, "T2_2-5"), (5, 15, "T3_5-15"), (15, 50, "T4_15-50"), (50, float("inf"), "T5_50+")]

def tier(nm):
    for lo, hi, name in BANDS:
        if lo <= nm < hi:
            return name
    return None

groups = {}
with open("outputs/condition_ratios.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        t = tier(float(r["nm_price"]))
        if t is None:
            continue
        groups.setdefault((t, r["condition"]), []).append(float(r["ratio"]))

rows = []
for lo, hi, name in BANDS:
    for cond in ["LP", "MP", "HP", "DMG"]:
        vals = groups.get((name, cond), [])
        if not vals:
            continue
        s = sorted(vals)
        rows.append({
            "tier": name,
            "condition": cond,
            "multiplier": round(statistics.median(vals), 3),
            "p25": round(s[len(s) // 4], 3),
            "p75": round(s[(3 * len(s)) // 4], 3),
            "n": len(vals),
        })

Path("outputs").mkdir(exist_ok=True)
with open("outputs/tiered_multiplier.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(r)
