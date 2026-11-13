import csv
import statistics

by_cond = {}
with open("outputs/condition_ratios.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        by_cond.setdefault(("all", r["condition"]), []).append(float(r["ratio"]))
        if float(r["nm_price"]) >= 5:
            by_cond.setdefault(("nm>=5", r["condition"]), []).append(float(r["ratio"]))

for tier in ["all", "nm>=5"]:
    for cond in ["LP", "MP", "HP", "DMG"]:
        vals = by_cond.get((tier, cond), [])
        if vals:
            print(tier, cond, "n=", len(vals), "medyan=", round(statistics.median(vals), 3))
