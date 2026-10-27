#!/usr/bin/env python3
"""v3 构造：OpAI-Bench 式渐进人→AI 修订轨迹（中文适配版）。

协议来源：VILA-Lab/OpAI-Bench（Apache-2.0，见 vendor/opai_bench/NOTICE.md）
- v0..v8 共 9 个版本，AI 覆盖率 0 → 1.0
- 5 种操作：polish / paraphrase / style / compress / expand
- 每句只被改写一次，后续版本累积继承

与原文的差异（中文适配）：
- 用本项目切句器（识别半角 "."）替代 NLTK
- 长度约束按“字符”而非“词”
- 中文 prompt + OpenAI 兼容网关
- 输出 JSONL，带字符区间与逐句 provenance

用法：
  GEN_API_KEY=... GEN_BASE_URL=... GEN_MODEL=deepseek-chat \
  python3 scripts/build_v3_progressive.py --papers /tmp/zh_papers.jsonl \
      --out ../data/l3/zh --max-papers 30 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import ChatClient
from sentences import split_sentences

SCHEDULE = [
    ("v0", "none", 0.00), ("v1", "polish", 0.15), ("v2", "paraphrase", 0.25),
    ("v3", "style", 0.40), ("v4", "compress", 0.50), ("v5", "expand", 0.60),
    ("v6", "style", 0.75), ("v7", "paraphrase", 0.90), ("v8", "polish", 1.00),
]

LABEL_MAP = {"polish": "deepseek_light", "paraphrase": "deepseek_heavy",
             "style": "deepseek_heavy", "compress": "deepseek_heavy",
             "expand": "deepseek_heavy"}

GUIDE = {
    "polish": "轻度润色：修正语法、搭配与标点，提升流畅度；句式基本不变。",
    "paraphrase": "改写措辞：用明显不同的表达重写，保持原意与信息量。",
    "style": "风格转换：改为更正式、书面的表达，保持原意。",
    "compress": "压缩：更短、更精炼，保留全部关键信息，不丢事实。",
    "expand": "扩写：补充少量解释性措辞，只使用原文已有或可直接 implied 的信息。",
}

EXCLUDE_SECTIONS = {"references", "ack", "appendix", "other"}


def cjk_ratio(s: str) -> float:
    return len(re.findall(r"[\u4e00-\u9fff]", s)) / max(1, len(s))


def build_prompt(sent: str, op: str, context: str) -> str:
    n = len(sent)
    if op in ("polish", "paraphrase", "style"):
        lo, hi = int(n * 0.85), int(n * 1.15)
    elif op == "compress":
        lo, hi = int(n * 0.6), int(n * 0.8)
    elif op == "expand":
        lo, hi = int(n * 1.2), int(n * 1.5)
    else:
        lo, hi = n, n
    return f"""你在做一句话的改写实验，操作类型：{op}。
{GUIDE[op]}

硬性要求：
- 只改写“目标句”，不要动上下文。
- 不新增原文没有的事实、例子、数字、人名。
- 保留原有数字、专有名词、引用标记。
- 保持中文（若原文是英文则保持英文）。
- 长度约 {lo}–{hi} 个字符。
- 输出必须且只能是一句话，不要解释、不要引号、不要编号。

上下文（仅用于保持连贯，不要改写）：
{context}

目标句：
{sent}
"""


def char_sim(a: str, b: str, n: int = 3) -> float:
    def grams(x):
        x = re.sub(r"\s+", "", x)
        return {x[i:i + n] for i in range(max(0, len(x) - n + 1))}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def clean_output(s: str) -> str:
    s = s.strip()
    s = re.split(r"\n+", s)[0].strip()
    s = re.sub(r"^(改写|输出|结果|重写)[:：]\s*", "", s)
    s = s.strip('"“”\'')
    parts = split_sentences(s)
    if len(parts) > 1 and len(parts[0]["text"]) >= 0.6 * len(s):
        s = parts[0]["text"]
    return s.strip()


def valid_rewrite(new: str, old: str, max_sim: float = 0.88) -> bool:
    new = clean_output(new)
    if not new or len(new) < 6:
        return False
    if new == old:
        return False
    if char_sim(new, old) > max_sim:
        return False
    if len(split_sentences(new)) != 1:
        return False
    if any(k in new for k in ("操作类型", "硬性要求", "目标句", "上下文")):
        return False
    r = len(new) / max(1, len(old))
    if not (0.4 <= r <= 2.2):
        return False
    if abs(cjk_ratio(new) - cjk_ratio(old)) > 0.3:
        return False
    return True


def doc_sentences(paper: dict) -> list[str]:
    parts = []
    for s in paper["sections"]:
        if s["key"] in EXCLUDE_SECTIONS:
            continue
        t = s["text"].strip()
        if t:
            parts.append(t)
    text = "\n\n".join(parts)
    return [s["text"] for s in split_sentences(text)]


def client_from_env() -> ChatClient:
    key = os.environ.get("GEN_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 GEN_API_KEY")
    return ChatClient(base_url=os.environ.get("GEN_BASE_URL", "https://api.deepseek.com/v1"),
                      api_key=key, model=os.environ.get("GEN_MODEL", "deepseek-chat"),
                      temperature=float(os.environ.get("GEN_TEMPERATURE", "0.7")),
                      max_tokens=600, cache_dir=".cache/llm", json_mode=False)


def plan_assignments(n: int, seed: int) -> list[tuple[int, int, str]]:
    """返回 [(rank, version_idx, op)]，每句恰好一次。"""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    rank = {sent_i: r for r, sent_i in enumerate(order)}
    assign = []
    for vi, (v, op, cov) in enumerate(SCHEDULE):
        if vi == 0:
            continue
        k = int(round(cov * n))
        assign.append((vi, v, op, k))
    out = {}
    prev_k = 0
    for vi, v, op, k in assign:
        for r in range(prev_k, min(k, n)):
            si = order[r]
            out[si] = (vi, v, op)
        prev_k = max(prev_k, k)
    return [(si, vi, op) for si, (vi, v, op) in out.items()]


def build_paper(paper: dict, client: ChatClient, workers: int) -> dict:
    sents = doc_sentences(paper)
    n = len(sents)
    if n < 5:
        return {}
    seed = int(hashlib.md5(paper["paper_id"].encode()).hexdigest()[:8], 16)
    tasks = plan_assignments(n, seed)
    version_of_op = {v: vi for vi, (v, _, _) in enumerate(SCHEDULE)}

    def do(si_op):
        si, vi, op = si_op
        sent = sents[si]
        ctx = " ".join(sents[max(0, si - 2):si] + sents[si + 1:si + 3])[:320]
        strict = ("实质性改写：不得照抄原句主干，不得只改标点或个别词；"
                  "改写后与原句的字符重合度必须明显下降。只输出一句话。")
        prompts = [build_prompt(sent, op, ctx),
                   f"{strict}\n操作：{op}\n原句：{sent}\n改写："]
        for prompt in prompts:
            try:
                out = client.chat([{"role": "user", "content": prompt}])
                out = clean_output(out)
                if valid_rewrite(out, sent):
                    return si, out, op, vi, True
            except Exception:
                pass
        return si, sent, op, vi, False

    with ThreadPoolExecutor(max_workers=min(workers, 8)) as ex:
        results = list(ex.map(do, tasks))

    edited = {si: (txt, op, vi, ok) for si, txt, op, vi, ok in results}
    is_zh = cjk_ratio("".join(sents)) > 0.3
    sep = "" if is_zh else " "
    current = list(sents)
    versions = []
    for vi, (v, op_all, cov) in enumerate(SCHEDULE):
        for si, (txt, op, evi, ok) in edited.items():
            if evi == vi and ok:
                current[si] = txt
        text, rows, pos = "", [], 0
        for i, s in enumerate(current):
            if i > 0:
                text += sep
                pos += len(sep)
            info = edited.get(i)
            applied = bool(info and info[3] and info[2] <= vi)
            label = LABEL_MAP[info[1]] if applied else "human"
            start = pos
            text += s
            pos += len(s)
            rows.append({"i": i, "text": s, "start": start, "end": pos, "label": label,
                         "op": info[1] if applied else "none",
                         "edited_v": SCHEDULE[info[2]][0] if applied else "v0"})
        versions.append({"v": v, "coverage": cov, "text": text, "sentences": rows})

    return {
        "paper_id": paper["paper_id"], "n_sentences": n,
        "lang": "zh" if is_zh else "en",
        "failed_edits": sum(1 for r in results if not r[4]),
        "versions": versions,
    }


def repair_record(rec: dict, client: ChatClient, workers: int) -> dict:
    """只重做“失败/近拷贝”的句子，其余保持原样。"""
    sents0 = [x["text"] for x in rec["versions"][0]["sentences"]]
    n = len(sents0)
    if n < 5:
        return rec
    seed = int(hashlib.md5(rec["paper_id"].encode()).hexdigest()[:8], 16)
    plan = {si: (vi, op) for si, vi, op in plan_assignments(n, seed)}
    final = [x["text"] for x in rec["versions"][-1]["sentences"]]
    todo = []
    for si, (vi, op) in plan.items():
        if final[si] == sents0[si] or char_sim(final[si], sents0[si]) > 0.88:
            todo.append((si, vi, op))
    if not todo:
        return rec

    strict = ("实质性改写：不得照抄原句主干，不得只改标点或个别词；"
              "改写后与原句的字符重合度必须明显下降。只输出一句话。")

    def do(t):
        si, vi, op = t
        sent = sents0[si]
        ctx = " ".join(sents0[max(0, si - 2):si] + sents0[si + 1:si + 3])[:320]
        ops = [op] + (["paraphrase"] if op != "paraphrase" else ["style"])
        pairs = [(build_prompt(sent, o, ctx), o) for o in ops]
        pairs.append((f"{strict}\n操作：paraphrase\n原句：{sent}\n改写：", "paraphrase"))
        pairs.append((("Rewrite the sentence from scratch with different structure and vocabulary. "
                       "Do not reuse any four consecutive words from the original. "
                       "Keep the meaning and all facts. Output exactly one sentence, nothing else.\n"
                       f"Sentence: {sent}"), "paraphrase"))
        for prompt, lab_op in pairs:
            try:
                out = clean_output(client.chat([{"role": "user", "content": prompt}]))
                if valid_rewrite(out, sent, 0.87):
                    return si, out, lab_op, vi, True
            except Exception:
                pass
        return si, final[si], op, vi, final[si] != sents0[si]

    with ThreadPoolExecutor(max_workers=min(workers, 8)) as ex:
        results = list(ex.map(do, todo))

    edits = {}
    for si, (vi, op) in plan.items():
        edits[si] = (final[si], op, vi, final[si] != sents0[si])
    for si, txt, op, vi, ok in results:
        edits[si] = (txt, op, vi, ok)

    is_zh = cjk_ratio("".join(sents0)) > 0.3
    sep = "" if is_zh else " "
    current = list(sents0)
    versions = []
    for k, (v, op_all, cov) in enumerate(SCHEDULE):
        for si, (txt, op, vi, ok) in edits.items():
            if ok and vi <= k:
                current[si] = txt
        text, rows, pos = "", [], 0
        for i, x in enumerate(current):
            if i > 0:
                text += sep
                pos += len(sep)
            txt, op, vi, ok = edits[i]
            applied = ok and vi <= k
            start = pos
            text += x
            pos += len(x)
            rows.append({"i": i, "text": x, "start": start, "end": pos,
                         "label": LABEL_MAP[op] if applied else "human",
                         "op": op if applied else "none",
                         "edited_v": SCHEDULE[vi][0] if applied else "v0"})
        versions.append({"v": v, "coverage": cov, "text": text, "sentences": rows})
    rec["versions"] = versions
    rec["failed_edits"] = sum(1 for si in plan if not edits[si][3])
    return rec


def repair_dir(out_dir: Path, client: ChatClient, workers: int, paper_workers: int = 12,
               shard_index: int = 0, shard_count: int = 1):
    files = sorted(out_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("stats")]
    if shard_count > 1:
        files = files[shard_index::shard_count]
    def run(fp):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(rec, dict) or not rec.get("versions"):
            return None
        new = repair_record(rec, client, workers)
        fp.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
        return new
    with ThreadPoolExecutor(max_workers=paper_workers) as ex:
        recs = [r for r in ex.map(run, files) if r]
    print(f"repaired {len(recs)} papers", file=sys.stderr)


def export_dataset(records: list[dict], out_dir: Path, keep_versions=None):
    """每个版本一篇文档，按论文切分 train/val/test。"""
    docs = []
    seen_text = {}
    for rec in records:
        for ver in rec["versions"]:
            if keep_versions and ver["v"] not in keep_versions:
                continue
            docs.append({"doc_id": f"{rec['paper_id']}::{ver['v']}",
                         "coverage": ver["coverage"],
                         "sentences": ver["sentences"]})

    papers = sorted({d["doc_id"].split("::")[0] for d in docs})
    rng = random.Random(13)
    rng.shuffle(papers)
    n_test = max(1, len(papers) // 7)
    n_val = max(1, len(papers) // 7)
    split = {p: ("test" if i < n_test else ("val" if i < n_test + n_val else "train"))
             for i, p in enumerate(papers)}

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    files = {k: open(out_dir / f"{k}.jsonl", "w", encoding="utf-8") for k in counts}
    for d in docs:
        sp = split[d["doc_id"].split("::")[0]]
        for r in d["sentences"]:
            t = r["text"].strip()
            owner = seen_text.get(t)
            if owner is not None and owner != d["doc_id"].split("::")[0]:
                continue
            seen_text[t] = d["doc_id"].split("::")[0]
            files[sp].write(json.dumps({
                "id": f"{d['doc_id']}:{r['i']}", "doc_id": d["doc_id"],
                "text": r["text"], "label": r["label"], "soft": {r["label"]: 1.0},
                "agreement": 1.0, "split": sp, "coverage": d["coverage"],
                "op": r["op"]}, ensure_ascii=False) + "\n")
            counts[sp] += 1
    for f in files.values():
        f.close()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-papers", type=int, default=30, dest="max_papers")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--paper-workers", type=int, default=1, dest="paper_workers")
    ap.add_argument("--limit-sentences", type=int, default=120, dest="limit_sentences")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--no-export", action="store_true", dest="no_export")
    ap.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    ap.add_argument("--shard-count", type=int, default=1, dest="shard_count")
    ap.add_argument("--export-versions", default="v0,v2,v4,v6,v8", dest="export_versions",
                    help="导出哪些版本（减少重复）")
    args = ap.parse_args()

    if args.repair:
        client = client_from_env()
        repair_dir(Path(args.out), client, args.workers, args.paper_workers,
                   args.shard_index, args.shard_count)
        recs = [json.loads(f.read_text(encoding="utf-8")) for f in Path(args.out).glob("*.json")
                if not f.name.startswith("stats")]
        if args.no_export:
            print(json.dumps({"repaired": len(recs), "exported": False}, ensure_ascii=False))
            return
        keep = [x.strip() for x in args.export_versions.split(",") if x.strip()]
        counts = export_dataset(recs, Path(args.out) / "dataset", keep_versions=keep)
        print(json.dumps({"repaired": len(recs), "counts": counts}, ensure_ascii=False))
        return

    papers = [json.loads(l) for l in open(args.papers, encoding="utf-8") if l.strip()]
    random.Random(7).shuffle(papers)
    papers = papers[: args.max_papers]
    client = client_from_env()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(p):
        dst = out_dir / f"{p['paper_id']}.json"
        if dst.exists():
            return json.loads(dst.read_text(encoding="utf-8"))
        rec = build_paper(p, client, args.workers)
        if rec:
            dst.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            print(f"{p['paper_id']} n={rec['n_sentences']} failed={rec['failed_edits']}",
                  file=sys.stderr, flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.paper_workers) as ex:
        records = [r for r in ex.map(run_one, papers) if r]

    keep = [x.strip() for x in args.export_versions.split(",") if x.strip()]
    counts = export_dataset(records, out_dir / "dataset", keep_versions=keep)
    (out_dir / "stats.json").write_text(json.dumps(
        {"papers": len(records), "counts": counts,
         "versions": [v for v, _, _ in SCHEDULE]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"papers": len(records), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
