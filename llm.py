import os, re, json, time, random
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError

load_dotenv()
BASE_URL = "https://api.inference.crusoecloud.com/v1"
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
FALLBACK = "openai/gpt-oss-120b"
PRICE = {SUPER: (0.30, 2.40), NANO: (0.05, 0.20), FALLBACK: (0.05, 0.25)}  # $ per 1M tokens (in, out)


class LLM:
    def __init__(self, budget_usd=35.0, log=print):
        self.client = OpenAI(api_key=os.environ["CRUSOE_API_KEY"], base_url=BASE_URL,
                             max_retries=0, timeout=180)
        self.spent, self.calls, self.budget = 0.0, 0, budget_usd
        self.kwargs_ok = True
        self.log = log

    def chat(self, model, system, user, max_tokens=2000, temperature=0.6, thinking=False):
        if self.spent > self.budget:
            raise RuntimeError(f"Budget stop: spent ${self.spent:.2f}")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        delay = 2.0
        for attempt in range(8):
            kw = dict(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
            if self.kwargs_ok:
                kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
            try:
                r = self.client.chat.completions.create(**kw)
                if r.usage:
                    pin, pout = PRICE.get(model, (1.0, 3.0))
                    self.spent += (r.usage.prompt_tokens * pin + r.usage.completion_tokens * pout) / 1e6
                self.calls += 1
                text = r.choices[0].message.content or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            except APIStatusError as e:
                if e.status_code == 400 and self.kwargs_ok:
                    self.kwargs_ok = False
                    continue
                if e.status_code in (429, 500, 502, 503, 504):
                    if attempt >= 4 and model != FALLBACK:
                        self.log(f"[fallback] {model} busy -> {FALLBACK}")
                        model = FALLBACK
                    time.sleep(min(delay, 30) + random.random())
                    delay *= 2
                    continue
                raise
            except (APIConnectionError, APITimeoutError):
                time.sleep(min(delay, 30) + random.random())
                delay *= 2
        raise RuntimeError("LLM call failed after retries")


def extract_json(text):
    for pat in (r"```json\s*(.*?)```", r"```\s*(\{.*?\})\s*```", r"(\{.*\})"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


if __name__ == "__main__":
    llm = LLM()
    out = llm.chat(SUPER, "Reply only with JSON.", 'Return {"ok": true, "word": "unfoolable"}')
    print("JSON:", extract_json(out))
    out = llm.chat(NANO, "You write Python.", "Write a function add(a, b) in a python code block.")
    print("CODE:", extract_code(out))
    print(f"COST: ${llm.spent:.5f} over {llm.calls} calls")