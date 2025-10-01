# src/ai_agent_new.py
import os
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
API_KEY = os.getenv("OPENROUTER_API_KEY")

def query_ai_agent(prompt):
    """
    Send a prompt to the AI agent (via OpenRouter API).
    The AI will interpret QSAR descriptors.
    """
    if not API_KEY:
        return "No API key found. Please set OPENROUTER_API_KEY in src/.env"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a chemoinformatics assistant. "
                           "Your role is to analyze QSAR descriptors of molecules "
                           "and suggest how these features may relate to activity or drug-likeness."
            },
            {"role": "user", "content": prompt}
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI agent request failed: {e}"
