import os
from openai import OpenAI

def get_grok_client() -> OpenAI:
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise EnvironmentError("GROK_API_KEY is not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

def grok_chat(messages: list[dict], temperature: float = 0.0) -> str:
    client = get_grok_client()
    try:
        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Grok API error: {e}")