# 模型：AI-writing-detector-laya-zh

**开源地址**：https://huggingface.co/johnson/AI-writing-detector-laya-zh

基于 Laya（421M 多语 System One 决策模型）微调的中文论文句级 AI 写作检测器。

## 指标

| 指标 | 数值 |
|---|---|
| TPR@1%FPR | **0.8954** |
| AUC | 0.9935 |
| FPR@1% | 1.03% |
| 对照（RoBERTa 102M） | 0.831 |

## 用法

```bash
pip install laya huggingface_hub
python src/inference.py --input paper.txt --thr 0.9
```

`p_ai = p_light + p_heavy`，默认阈值 0.9（对应 1% FPR）。

## 训练

```bash
bash scripts/finetune_laya.sh zh 1
```

- 数据：`zh/` 子集 10 万句
- 1 epoch，micro-batch 8，grad-accum 8，max_len 128
- A800 约 30 分钟

## 局限

- 只覆盖中文**学术**；医学/英文同配置下不如小模型
- 仅 DeepSeek 系改写；单句无上下文
- 不适用于学术不端判定
