import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["CRUSOE_API_KEY"], base_url="https://api.inference.crusoecloud.com/v1")
print("MODELS:", [m.id for m in client.models.list().data])
raw = client.chat.completions.with_raw_response.create(
    model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    messages=[{"role": "user", "content": "Say hi in five words."}], max_tokens=300)
print("RATE LIMIT:", raw.headers.get("x-ratelimit-limit-requests"))
print("REPLY:", raw.parse().choices[0].message.content)