import requests
from config import QWEN3_AGENT_URL, BGE_EMBED_URL, QWEN_VL_URL, QWEN_CODER_URL

def call_agent(messages, tools=None):
    resp = requests.post(f"{QWEN3_AGENT_URL}/v1/chat", json={
        "messages": messages, "tools": tools or []
    }, timeout=30)
    return resp.json()

def call_embed(texts):
    resp = requests.post(f"{BGE_EMBED_URL}/embed", json={"texts": texts}, timeout=30)
    return resp.json()

def call_vision(image_base64, prompt):
    resp = requests.post(f"{QWEN_VL_URL}/vision/analyze", json={
        "image_base64": image_base64, "prompt": prompt
    }, timeout=30)
    return resp.json()

def call_coder(prompt, context=""):
    resp = requests.post(f"{QWEN_CODER_URL}/code/generate", json={
        "prompt": prompt, "context": context
    }, timeout=30)
    return resp.json()