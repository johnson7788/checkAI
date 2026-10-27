# 数据集：AI-writing-detector-dataset

**开源地址**：https://huggingface.co/datasets/johnson/AI-writing-detector-dataset

句级数据集：人类原文 vs DeepSeek 润色/改写，标签由构造过程确定，无需人工标注。

| 子集 | 来源 | 文档数 | train / val / test |
|---|---|---|---|
| `zh/` | 中文期刊论文 | 118 | 186,640 / 36,189 / 29,932 |
| `zh_med/` | 中文医学摘要 | 8,000 | 325,300 / 65,420 / 65,850 |
| `en/` | arXiv 2007–2017 | 808 | 783,372 / 150,857 / 145,470 |

## 许可

Apache-2.0。

## 构造协议（v3 渐进式）

```
v0 ─ v1(15%) ─ v2(25%) ─ v3(40%) ─ v4(50%) ─ v5(60%) ─ v6(75%) ─ v7(90%) ─ v8(100%)
      polish    paraphrase   style     compress   expand      style     paraphrase  polish
```

1. 人类论文切句；
2. 每句按固定随机序分配到某个版本，只改写一次，后续版本累积继承；
3. 5 种操作：润色/改写/风格/压缩/扩写；
4. 句级一对一改写，标签与文本同时产生；
5. 导出 v0/v2/v4/v6/v8。

## 质量

| 指标 | zh | zh_med | en |
|---|---|---|---|
| 标签不一致 | 0 | 0 | 0 |
| 近拷贝率 | 0.38% | 1.1% | 0.0% |
| v8 覆盖率 | 96.4% | 98.2% | 96.0% |

## 字段

`id` / `doc_id` / `text` / `label` / `soft` / `agreement` / `split` / `coverage` / `op`

## 复现

```bash
# 构造（需要 LLM API）
python src/build_v3.py --papers papers.jsonl --out data/l3/zh --max-papers 118 --workers 8

# 质量审计
python src/audit_v3.py --src data/l3/zh --out audit.json

# 近拷贝降级
python src/clean_nearcopy.py --dir data/l3/zh --threshold 0.9 --export
```
