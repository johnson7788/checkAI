#!/usr/bin/env python3
"""把 v3 JSONL 数据集转成 Laya 微调用的 CSV（text,label）。

用法：
  python export_laya_csv.py --input data/zh/train.jsonl --out train.csv --max-rows 100000
"""
import argparse
import csv
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=100000, dest="max_rows")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    if a.max_rows and len(rows) > a.max_rows:
        rng = random.Random(a.seed)
        by = defaultdict(list)
        for r in rows:
            by[(round(r.get("coverage", 0), 2), r["label"])].append(r)
        out = []
        for k in sorted(by):
            v = by[k]
            rng.shuffle(v)
            out += v[: max(1, int(len(v) * a.max_rows / len(rows)))]
        rng.shuffle(out)
        rows = out[: a.max_rows]

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for r in rows:
            w.writerow([r["text"], r["label"]])
    print(f"{a.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
