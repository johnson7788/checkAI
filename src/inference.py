#!/usr/bin/env python3
"""AI-writing-detector-laya-zh 推理脚本。

用法：
  python inference.py --input paper.txt --thr 0.9
  python inference.py --text "本文提出一种方法。实验表明……"
  python inference.py --input paper.txt --out result.json

依赖：pip install laya huggingface_hub
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

MODEL_ID = os.environ.get("MODEL_ID", "johnson/AI-writing-detector-laya-zh")
BOUNDARY = re.compile(r"[。！？!?…；;]+[”’\"』」）)\]]*|\n{2,}")


def split_sentences(text: str, min_len: int = 6):
    spans, start = [], 0
    for m in BOUNDARY.finditer(text):
        seg = text[start:m.end()].strip()
        if seg:
            spans.append(seg)
        start = m.end()
    if text[start:].strip():
        spans.append(text[start:].strip())
    merged = []
    for s in spans:
        if merged and len(s) < min_len:
            merged[-1] += s
        else:
            merged.append(s)
    return merged


def smooth(xs, w: int = 3):
    if len(xs) < w:
        return xs
    out = []
    half = w // 2
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def segments(scores, thr: float, min_len: int = 2):
    out, start = [], None
    for i, s in enumerate(scores):
        if s >= thr and start is None:
            start = i
        elif s < thr and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(scores) - start >= min_len:
        out.append((start, len(scores) - 1))
    return out


def load_agent(model_id: str):
    import os
    from huggingface_hub import snapshot_download
    import laya

    local = model_id if os.path.isdir(model_id) else snapshot_download(model_id)
    return laya.load(local), local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--thr", type=float, default=0.9, help="AI 判定阈值（0.9 ≈ 1% FPR）")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--min-seg", type=int, default=2, dest="min_seg")
    args = ap.parse_args()

    text = args.text or (open(args.input, encoding="utf-8").read() if args.input else "")
    if not text.strip():
        sys.exit("请提供 --input 或 --text")

    agent, local = load_agent(args.model)
    questions = json.load(open(f"{local}/questions.json"))

    sents = split_sentences(text)
    rows = []
    for i, s in enumerate(sents):
        try:
            ans = agent.predict(s, questions)["answers"]["label"]
            probs = ans.get("probabilities") or {}
            if isinstance(probs, list):
                probs = {x["value"]: x["probability"] for x in probs}
            p_ai = float(probs.get("deepseek_light", 0.0) + probs.get("deepseek_heavy", 0.0))
        except Exception as e:
            probs, p_ai = {}, 0.0
        rows.append({"i": i, "text": s, "p_ai": round(p_ai, 4),
                     "p_light": round(float(probs.get("deepseek_light", 0.0)), 4),
                     "p_heavy": round(float(probs.get("deepseek_heavy", 0.0)), 4)})

    p = [r["p_ai"] for r in rows]
    ps = smooth(p, 3)
    segs = segments(ps, args.thr, args.min_seg)
    ratio = sum(x >= args.thr for x in p) / max(1, len(p))

    result = {
        "model": args.model,
        "threshold": args.thr,
        "n_sentences": len(rows),
        "ai_ratio": round(ratio, 4),
        "segments": [{"start": a, "end": b,
                      "mean_p": round(sum(ps[a:b + 1]) / (b - a + 1), 4)} for a, b in segs],
        "sentences": rows,
    }

    print(f"句数={len(rows)}  AI句占比={ratio:.1%}  区间数={len(segs)}  阈值={args.thr}")
    for a, b in segs:
        print(f"  [{a}-{b}] p={sum(ps[a:b + 1]) / (b - a + 1):.2f}  {rows[a]['text'][:60]}")

    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"已写入 {args.out}")
    else:
        print(json.dumps(result, ensure_ascii=False)[:300] + " ...")


if __name__ == "__main__":
    main()
