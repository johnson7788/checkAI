#!/usr/bin/env python3
"""把“近拷贝”AI 句降级为 human：sim>0.9 说明改写没有实质发生，保留为正样本只会教坏模型。"""
import argparse, glob, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_v3 import SCHEDULE, LABEL_MAP, char_sim, cjk_ratio, plan_assignments

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--threshold", type=float, default=0.9)
ap.add_argument("--export", action="store_true")
a = ap.parse_args()

files = [f for f in glob.glob(f"{a.dir}/*.json") if not f.endswith("stats.json")]
flipped_total = kept_total = 0
recs = []
for fp in files:
    rec = json.load(open(fp, encoding="utf-8"))
    if not isinstance(rec, dict) or not rec.get("versions"):
        continue
    v0 = [s["text"] for s in rec["versions"][0]["sentences"]]
    final = [s["text"] for s in rec["versions"][-1]["sentences"]]
    n = len(v0)
    seed = int(hashlib.md5(rec["paper_id"].encode()).hexdigest()[:8], 16)
    plan = {si: (vi, op) for si, vi, op in plan_assignments(n, seed)}
    edits = {}
    flipped = 0
    for i in range(len(v0)):
        if final[i] == v0[i]:
            edits[i] = (final[i], "none", 0, False)
            continue
        sim = char_sim(final[i], v0[i])
        if sim > a.threshold:
            edits[i] = (v0[i], "none", 0, False)
            flipped += 1
        else:
            vi, op = plan.get(i, (8, "paraphrase"))
            edits[i] = (final[i], op, vi, True)
    flipped_total += flipped
    is_zh = cjk_ratio("".join(v0)) > 0.3
    sep = "" if is_zh else " "
    current = list(v0)
    versions = []
    for k, (v, op_all, cov) in enumerate(SCHEDULE):
        for i, (txt, op, vi, ok) in edits.items():
            if ok and vi <= k:
                current[i] = txt
        text, rows, pos = "", [], 0
        for i, x in enumerate(current):
            if i:
                text += sep; pos += len(sep)
            txt, op, vi, ok = edits[i]
            applied = ok and vi <= k
            start = pos; text += x; pos += len(x)
            rows.append({"i": i, "text": x, "start": start, "end": pos,
                         "label": LABEL_MAP.get(op, "deepseek_heavy") if applied else "human",
                         "op": op if applied else "none",
                         "edited_v": SCHEDULE[vi][0] if applied else "v0"})
        versions.append({"v": v, "coverage": cov, "text": text, "sentences": rows})
    rec["versions"] = versions
    rec["failed_edits"] = sum(1 for i in edits if not edits[i][3])
    Path(fp).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    recs.append(rec)
    kept_total += sum(1 for i in edits if edits[i][3])
print(json.dumps({"papers": len(recs), "flipped_sentences": flipped_total, "kept_ai": kept_total},
                 ensure_ascii=False))
if a.export:
    from build_v3 import export_dataset
    counts = export_dataset(recs, Path(a.dir) / "dataset", keep_versions=["v0","v2","v4","v6","v8"])
    print(json.dumps({"counts": counts}, ensure_ascii=False))
