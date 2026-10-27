# AI Writing Detector (Laya)

中文论文**句级 AI 写作检测**：给一篇文章，标出哪些句子/区间是 AI 生成或改写的，供作者针对性修改。

- 模型：**Laya（421M 多语 System One 决策模型）** 微调
- 数据：人类论文 vs DeepSeek 润色/改写，构造式标签（零人工标注）
- 效果：中文学术 **TPR@1%FPR = 0.895**（对照 RoBERTa 102M：0.831）

## 开源地址

| 资产 | 地址 |
|---|---|
| **模型** | https://huggingface.co/johnson/AI-writing-detector-laya-zh |
| **数据集** | https://huggingface.co/datasets/johnson/AI-writing-detector-dataset |

## 快速开始

```bash
pip install -r requirements.txt

# 1. 推理（自动下载模型）
python src/inference.py --input paper.txt --thr 0.9

# 2. 下载数据
hf download johnson/AI-writing-detector-dataset --repo-type dataset \
  --local-dir data --include "zh/*"

# 3. 导出 Laya 训练 CSV
python src/export_laya_csv.py --input data/zh/train.jsonl --out data/zh_train.csv --max-rows 100000

# 4. 微调 + 评测
bash scripts/finetune_laya.sh zh 1
```

## 输出

```json
{
  "n_sentences": 42,
  "ai_ratio": 0.214,
  "segments": [{"start": 17, "end": 20, "mean_p": 0.96}],
  "sentences": [{"i": 0, "text": "...", "p_ai": 0.0006}]
}
```

`p_ai = p_light + p_heavy`；默认阈值 0.9（≈1% FPR），高精确场景用 0.95。

## 仓库结构

```
src/
  inference.py          # 推理：切句 → LayA 判分 → 平滑 → 区间
  eval_laya.py          # 评测：AUC / TPR@1%FPR
  export_laya_csv.py    # JSONL → Laya CSV
  build_v3.py           # v3 渐进式数据构造
  audit_v3.py           # 数据质量审计
  clean_nearcopy.py     # 近拷贝降级
  sentences.py, llm.py  # 切句 / LLM 客户端
scripts/finetune_laya.sh
docs/{data,model,quality}.md
samples/                # 每域 200 条样例
examples/sample.txt
```

## 数据

三域共 **1,789,030 行 / 约 35 万独立句**：

| 子集 | 文档 | train / val / test |
|---|---|---|
| zh | 118 | 186,640 / 36,189 / 29,932 |
| zh_med | 8,000 | 325,300 / 65,420 / 65,850 |
| en | 808 | 783,372 / 150,857 / 145,470 |

详见 [docs/data.md](docs/data.md)。

## 模型

| 指标 | 数值 |
|---|---|
| TPR@1%FPR | 0.8954 |
| AUC | 0.9935 |
| FPR@1% | 1.03% |

训练：10 万句、1 epoch、micro-batch 8 × grad-accum 8、max_len 128，A800 约 30 分钟。
详见 [docs/model.md](docs/model.md)。

## 局限

- 只验证了中文**学术**；医学/英文同配置下不如小模型（0.587 / 0.548）；
- 只覆盖 DeepSeek 系改写；单句、无上下文；
- 存在误报，不得用于学术不端判定。

## 许可

- 代码：Apache-2.0
- 数据：Apache-2.0
- 模型：Apache-2.0（继承 Laya）

## 引用

```bibtex
@misc{ai-writing-detector-laya,
  title  = {AI Writing Detector: Sentence-level AI Involvement Detection with Laya},
  author = {johnson7788},
  year   = {2026},
  url    = {https://huggingface.co/johnson/AI-writing-detector-laya-zh}
}
```
