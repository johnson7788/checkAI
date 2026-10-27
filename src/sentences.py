"""句子切分：保留字符偏移，服务"高亮定位"与训练标注对齐。

设计要点
- 在**原文**上切分，start/end 是原文字符偏移，前端可直接用它做高亮。
- 额外给出 norm 字段（NFKC + 去零宽），只用于送模型；不影响偏移。
- 过短片段并入前一句，避免"好。"这种句子单独判定。
"""
from __future__ import annotations

import re

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)

# 句末标点（含中文省略号、分号）+ 其后的右引号/右括号
_BOUNDARY = re.compile(
    r"[。．！？!?…；;]+[”’\"』」）)\]]*"          # 中文/通用句末标点（含全角 ．）
    r"|\n{2,}"                                   # 只把空行当边界，PDF 的软换行不算
    r"|(?<=[A-Za-z0-9\)\]\"'])\.(?=\s+[A-Z0-9(\"'])"   # 英文句号
    # 中文论文大量使用半角 "." 作句号：前一个字符是汉字时切分；数字小数点不受影响
    r"|(?<=[\u4e00-\u9fff])\.(?=[\s\u4e00-\u9fff“”\"’'）)]|$)"
)

_ABBR = re.compile(r"(?i)(?:\bet\s+al|\be\.g|\bi\.e|\bfig|\beq|\bvs|\bno|\bdr|\bmr|\bms|\bcf)\.$")


def normalize_for_model(text: str) -> str:
    """送模型前的轻量清洗：只去零宽、压缩空白。

    注意：**不要做 NFKC**。NFKC 会把中文全角标点（，。：）转成半角，
    既破坏文体信号，也会让生成式标注产出带标点伪影的样本。
    """
    text = text.translate(_ZERO_WIDTH)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sentences(text: str, min_len: int = 6) -> list[dict]:
    """返回 [{i, text, norm, start, end}, ...]。

    text 为原文切片（首尾空白已 trim），start/end 为原文偏移。
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        if text[start:m.end()].strip():
            spans.append((start, m.end()))
        start = m.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))

    items: list[dict] = []
    for s, e in spans:
        seg = text[s:e]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        items.append({"text": seg.strip(), "start": s + lead, "end": e - trail})

    # 缩写结尾（et al. / Fig. / e.g.）不单独成句，并入下一句
    abbr: list[dict] = []
    for it in items:
        if abbr and _ABBR.search(abbr[-1]["text"]):
            abbr[-1]["end"] = it["end"]
            abbr[-1]["text"] = text[abbr[-1]["start"]:it["end"]].strip()
        else:
            abbr.append(it)
    items = abbr

    # 合并过短句（并入前一句）
    merged: list[dict] = []
    for it in items:
        if merged and len(it["text"]) < min_len:
            merged[-1]["end"] = it["end"]
            merged[-1]["text"] = text[merged[-1]["start"]:merged[-1]["end"]].strip()
        else:
            merged.append(it)

    for i, it in enumerate(merged):
        it["i"] = i
        it["norm"] = normalize_for_model(it["text"])
    return merged


def attrs_by_overlap(start: int, end: int, spans: list[dict]):
    """返回重叠最多的 span（dict），用于同时取 label / section / mode。"""
    best, best_len = None, 0
    for sp in spans:
        ov = min(end, sp["end"]) - max(start, sp["start"])
        if ov > best_len:
            best, best_len = sp, ov
    return best


def label_by_overlap(start: int, end: int, spans: list[dict], default: str = "unknown") -> str:
    """按字符区间最大重叠，给预测句分配金标（用于真值对齐）。"""
    best, best_len = default, 0
    for sp in spans:
        ov = min(end, sp["end"]) - max(start, sp["start"])
        if ov > best_len:
            best, best_len = sp["label"], ov
    return best if best_len > 0 else default
