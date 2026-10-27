"""OpenAI 兼容的极简客户端（stdlib，无第三方依赖）。

支持任何 /chat/completions 端点：
- OpenAI:    JUDGE_BASE_URL=https://api.openai.com/v1
- DeepSeek:  GEN_BASE_URL=https://api.deepseek.com/v1
- 本地 vLLM/Ollama(openai 兼容): http://127.0.0.1:8000/v1

特性：磁盘缓存（同样的 messages 不重复付费）、指数退避、并发由上层控制。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _extract_json(text: str):
    """从模型输出里抠出第一个完整 JSON（对象或数组）。"""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 去掉 ```json ``` 围栏
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            text = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    raise LLMError(f"无法解析 JSON：{text[:300]}")


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.0, max_tokens: int = 1200,
                 timeout: int = 120, cache_dir: str | None = ".cache/llm",
                 json_mode: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_mode = json_mode
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ---------- cache ----------
    def _cache_path(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key[:2], key + ".json")

    def _cache_get(self, key: str):
        p = self._cache_path(key)
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)["content"]
        return None

    def _cache_put(self, key: str, content: str):
        p = self._cache_path(key)
        if not p:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f, ensure_ascii=False)
        os.replace(tmp, p)

    # ---------- call ----------
    def chat(self, messages: list[dict], **overrides) -> str:
        params = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        params.update(overrides)
        raw_key = json.dumps([params, messages], ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(raw_key.encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        body = {"model": params["model"], "messages": messages,
                "temperature": params["temperature"], "max_tokens": params["max_tokens"]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body, ensure_ascii=False).encode()
        last = ""
        for attempt in range(4):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode())
                content = payload["choices"][0]["message"]["content"]
                self._cache_put(key, content)
                return content
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code in (400, 401, 403, 404):
                    raise LLMError(last) from e
                time.sleep(2 ** attempt)
            except Exception as e:  # 网络/解析
                last = str(e)
                time.sleep(2 ** attempt)
        raise LLMError(f"调用失败（重试后）：{last}")

    def chat_json(self, messages: list[dict], **overrides):
        return _extract_json(self.chat(messages, **overrides))


class MockClient:
    """离线演示/单测用：基于关键词的确定性"假模型"，不产生任何费用。

    仅用于验证流水线跑通，**不能**用来评估真实 judge 质量。
    """

    CUE = ("首先", "其次", "此外", "综上", "值得注意", "不仅", "而且", "因此",
           "总之", "总而言之", "进一步", "从而", "在当今", "随着")

    def __init__(self, noise: float = 0.0, label: str = "deepseek"):
        self.noise = noise
        self.label = label
        self.calls = 0

    def _items(self, prompt: str):
        return re.findall(r"^\[(\d+)\]\s*(.+)$", prompt, re.M)

    def chat(self, messages: list[dict], **overrides) -> str:
        self.calls += 1
        prompt = messages[-1]["content"]
        items = self._items(prompt)

        # 生成侧：整节代写
        if "撰写论文的「" in prompt:
            sec = prompt.split("撰写论文的「", 1)[1].split("」", 1)[0]
            return json.dumps({"text": (
                f"本节围绕{sec}展开讨论。首先，本文梳理了该问题的研究背景与基本假设，"
                "并给出了形式化描述。其次，结合已有工作说明了本文方法的适用边界与约束条件。"
                "综上所述，本节为后续实验与分析奠定了概念基础。"
            )}, ensure_ascii=False)

        # 生成侧：论文段落级操作（润色/降重/扩写/翻译）
        if any(k in prompt for k in ("润色", "改写降重", "扩写约", "学术翻译")):
            body = prompt.split("原文：")[-1].split("硬性要求")[0].strip()
            if not body:
                body = "本文围绕该问题展开分析，并从多个角度进行了讨论。"
            text = f"此外，{body.rstrip('。')}。综上所述，这一点值得进一步关注。"
            return json.dumps({"text": text}, ensure_ascii=False)

        # 生成侧：逐句改写
        if "逐句改写" in prompt:
            out = []
            for i, sent in items:
                s = sent.strip().rstrip("。！？!?")
                out.append({"i": int(i), "text": f"此外，{s}，这一点值得进一步关注。"})
            return json.dumps({"items": out}, ensure_ascii=False)

        # 生成侧：整篇改写
        if "改写整篇" in prompt or "整体改写" in prompt:
            body = prompt.split("原文：", 1)[-1].strip()
            return json.dumps({"text": body.replace("。", "。此外，这一点值得关注。")},
                              ensure_ascii=False)

        # 判定侧
        out = []
        for i, sent in items:
            s = sent.strip()
            has_cue = any(c in s for c in self.CUE)
            if has_cue:
                lab, conf = self.label, 0.82
            else:
                lab, conf = "human", 0.74
            if self.noise and (int(i) % 7 == 0):
                lab, conf = "uncertain", 0.35
            out.append({"i": int(i), "label": lab, "confidence": conf,
                        "cues": ["连接词密集"] if has_cue else ["口语/具体细节"]})
        return json.dumps({"items": out}, ensure_ascii=False)

    def chat_json(self, messages, **overrides):
        return _extract_json(self.chat(messages, **overrides))
