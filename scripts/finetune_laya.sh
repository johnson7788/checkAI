#!/bin/bash
# 端到端：下载数据集 → 导出 CSV → Laya 微调 → 评测
set -e
DOMAIN=${1:-zh}          # zh | zh_med | en
EPOCHS=${2:-1}
OUT=${3:-runs/laya_$DOMAIN}

hf download johnson/AI-writing-detector-dataset --repo-type dataset \
  --local-dir data --include "$DOMAIN/*"

python src/export_laya_csv.py --input data/$DOMAIN/train.jsonl --out data/${DOMAIN}_train.csv --max-rows 100000

laya-train --data data/${DOMAIN}_train.csv \
  --base convaiinnovations/laya-multilingual \
  --text-column text --label-column label \
  --instructions "判断这句话是人工撰写、AI 轻度润色，还是 AI 改写。human=人工，deepseek_light=轻度润色，deepseek_heavy=改写。" \
  --out "$OUT" --epochs "$EPOCHS" --micro-batch 8 --grad-accum 8 \
  --max-len 128 --device cuda

python src/eval_laya.py "$OUT" data/$DOMAIN/test.jsonl 3000
