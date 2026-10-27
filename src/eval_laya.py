import json, random, sys
import numpy as np
import laya

CKPT = sys.argv[1]
DATA = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

agent = laya.load(CKPT)
q = json.load(open(CKPT + "/questions.json"))
rows = [json.loads(l) for l in open(DATA)]
random.seed(9); random.shuffle(rows)
rows = rows[:N]

probs, ys = [], []
for r in rows:
    txt = r.get("cur_text", r["text"])
    a = agent.predict(txt, q)["answers"]["label"]
    p = a.get("probabilities")
    if isinstance(p, dict):
        d = p
    else:
        d = {x["value"]: x["probability"] for x in p}
    probs.append(d.get("deepseek_light", 0) + d.get("deepseek_heavy", 0))
    ys.append(0 if r["label"] == "human" else 1)

y = np.array(ys); s = np.array(probs)
o = np.argsort(s); rank = np.empty_like(o, float); rank[o] = np.arange(1, len(s) + 1)
pos = y == 1
auc = (rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
neg = s[y == 0]; posv = s[y == 1]
thr = np.quantile(neg, 0.99)
print(json.dumps({"n": len(y), "auc": round(float(auc), 4),
                  "tpr@1fpr": round(float((posv >= thr).mean()), 4),
                  "fpr@1fpr": round(float((neg >= thr).mean()), 4),
                  "mean_human": round(float(neg.mean()), 3),
                  "mean_ai": round(float(posv.mean()), 3)}))
