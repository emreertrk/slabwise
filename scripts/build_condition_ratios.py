import csv
import glob
import json
import statistics
from pathlib import Path

CONDS = {
    "Near Mint": "NM",
    "Lightly Played": "LP",
    "Moderately Played": "MP",
    "Heavily Played": "HP",
    "Damaged": "DMG",
}

rows = []
for f in glob.glob("data/raw/*.json"):
    doc = json.load(open(f, encoding="utf-8"))
    cards = doc["data"] if isinstance(doc, dict) else doc
    for c in cards:
        prices = {}
        for v in c.get("variants", []):
            cond = CONDS.get(v.get("condition"))
            if cond is None or v.get("priceStatus") != "current" or v.get("language") != "English":
                continue
            if not v.get("price"):
                continue
            prices[(v.get("printing"), cond)] = float(v["price"])
        for (printing, cond), p in prices.items():
            nm = prices.get((printing, "NM"))
            if cond == "NM" or not nm:
                continue
            rows.append({
                "card_uuid": c.get("uuid"),
                "tcgplayer_id": c.get("tcgplayerId"),
                "printing": printing,
                "condition": cond,
                "nm_price": nm,
                "cond_price": p,
                "ratio": round(p / nm, 4),
            })

Path("outputs").mkdir(exist_ok=True)
with open("outputs/condition_ratios.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("satir:", len(rows))
for cond in ["LP", "MP", "HP", "DMG"]:
    vals = sorted(r["ratio"] for r in rows if r["condition"] == cond)
    if vals:
        print(cond, "n=", len(vals), "medyan=", round(statistics.median(vals), 3),
              "p10=", round(vals[len(vals) // 10], 3), "p90=", round(vals[len(vals) * 9 // 10], 3))
