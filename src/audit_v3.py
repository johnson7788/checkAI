#!/usr/bin/env python3
"""v3 标注数据质量审计。

检查项：
1. 标签一致性：AI 句必须与 v0 不同，human 句必须与 v0 完全相同
2. 指令泄漏：改写结果里是否混入 prompt 文字
3. 长度异常：改写长度比是否超出 [0.4, 2.2]
4. 语言漂移：CJK 占比变化是否过大
5. 重复：完全重复句、跨论文重复
6. 语义保真：AI 句与原句的字符 3-gram Jaccard
7. 覆盖率：实际 AI 占比 vs 目标

用法：
  python3 scripts/audit_v3.py --src ../data/l3/zh ../data/l3/zh_med ../data/l3/en --out ../data/quality_report.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict

LEAK = re.compile(r"操作类型|硬性要求|目标句|Operation:|Constraints:|Output:|Target sentence|改写后的|只输出|不要解释")
CJK = re.compile(r"[\u4e00-\u9fff]")


def grams(s: str, n: int = 3):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def audit(src: str, sample_every: int = 20):
    st = {
        "papers": 0, "sentences": 0, "ai": 0, "human": 0,
        "label_mismatch": 0, "leak": 0, "ratio_out": 0, "cjk_drift": 0,
        "dup_sent": 0, "cross_paper_dup": 0, "dup_same_version": 0,
        "similarity": [], "near_copy": 0, "low_sim": 0, "by_op": defaultdict(lambda: {"n": 0, "sim": [], "ratio": []}),
        "coverage": defaultdict(lambda: {"target": 0.0, "ai": 0, "total": 0}),
        "examples": [],
    }
    seen = {}
    for fp in glob.glob(f"{src}/*.json"):
        rec = json.load(open(fp, encoding="utf-8"))
        if not isinstance(rec, dict) or not rec.get("paper_id") or not rec.get("versions"):
            continue
        if not isinstance(rec["versions"][0], dict):
            continue
        st["papers"] += 1
        v0 = rec["versions"][0]["sentences"]
        origs = {s["i"]: s["text"] for s in v0}
        for ver in rec["versions"]:
            local = set()
            for s in ver["sentences"]:
                orig = origs.get(s["i"])
                if orig is None:
                    continue
                st["sentences"] += 1
                is_ai = s["label"] != "human"
                st["ai" if is_ai else "human"] += 1
                if (s["text"] == orig) == is_ai:
                    st["label_mismatch"] += 1
                    if len(st["examples"]) < 20:
                        st["examples"].append({"type": "label_mismatch", "paper": rec["paper_id"],
                                               "v": ver["v"], "label": s["label"],
                                               "orig": orig[:80], "text": s["text"][:80]})
                if LEAK.search(s["text"]):
                    st["leak"] += 1
                    if len(st["examples"]) < 20:
                        st["examples"].append({"type": "leak", "paper": rec["paper_id"],
                                               "text": s["text"][:120]})
                st["coverage"][ver["v"]]["target"] = ver["coverage"]
                st["coverage"][ver["v"]]["total"] += 1
                if is_ai:
                    st["coverage"][ver["v"]]["ai"] += 1
                    ratio = len(s["text"]) / max(1, len(orig))
                    cj = len(CJK.findall(s["text"])) / max(1, len(s["text"]))
                    cjo = len(CJK.findall(orig)) / max(1, len(orig))
                    op = s.get("op", "?")
                    st["by_op"][op]["n"] += 1
                    st["by_op"][op]["ratio"].append(ratio)
                    if not (0.4 <= ratio <= 2.2):
                        st["ratio_out"] += 1
                    if abs(cj - cjo) > 0.3:
                        st["cjk_drift"] += 1
                    if len(st["similarity"]) < 200000 and hash(s["text"]) % sample_every == 0:
                        sim = jaccard(grams(orig), grams(s["text"]))
                        st["similarity"].append(sim)
                        if sim > 0.9: st["near_copy"] += 1
                        if sim < 0.15: st["low_sim"] += 1
                        st["by_op"][op]["sim"].append(sim)
                h = hashlib.md5(s["text"].strip().encode()).hexdigest()
                if h in local:
                    st["dup_same_version"] += 1
                local.add(h)
                if h in seen:
                    st["dup_sent"] += 1
                    if seen[h] != rec["paper_id"]:
                        st["cross_paper_dup"] += 1
                else:
                    seen[h] = rec["paper_id"]
    return st


def summarize(st):
    n = max(1, st["sentences"])
    sims = sorted(st["similarity"])
    def q(p): return sims[int(len(sims) * p)] if sims else 0
    out = {
        "papers": st["papers"], "sentences": st["sentences"],
        "ai_ratio": round(st["ai"] / n, 4),
        "label_mismatch_rate": round(st["label_mismatch"] / n, 5),
        "leak_rate": round(st["leak"] / n, 5),
        "ratio_out_rate": round(st["ratio_out"] / max(1, st["ai"]), 4),
        "cjk_drift_rate": round(st["cjk_drift"] / max(1, st["ai"]), 4),
        "dup_rate": round(st["dup_sent"] / n, 4),
        "dup_same_version_rate": round(st["dup_same_version"] / n, 5),
        "cross_paper_dup_rate": round(st["cross_paper_dup"] / n, 4),
        "near_copy_rate": round(st["near_copy"] / max(1, len(sims)), 4),
        "low_sim_rate": round(st["low_sim"] / max(1, len(sims)), 4),
        "sim_p10": round(q(0.1), 3), "sim_p50": round(q(0.5), 3), "sim_p90": round(q(0.9), 3),
        "coverage": {k: {"target": v["target"], "actual": round(v["ai"] / max(1, v["total"]), 3)}
                     for k, v in sorted(st["coverage"].items())},
        "by_op": {k: {"n": v["n"],
                      "sim_p50": round(sorted(v["sim"])[len(v["sim"]) // 2], 3) if v["sim"] else None,
                      "ratio_p50": round(sorted(v["ratio"])[len(v["ratio"]) // 2], 3)}
                  for k, v in st["by_op"].items()},
        "examples": st["examples"][:10],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {}
    for s in a.src:
        print("auditing", s, flush=True)
        report[s.split("/")[-1]] = summarize(audit(s))
    json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
