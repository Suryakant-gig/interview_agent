import os
from google import genai
from google.genai import types


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def gemini_chat(messages: list[dict], temperature: float = 0.0) -> str:
    client = get_gemini_client()
    try:
        # Gemini has no "system"/"user"/"assistant" roles like OpenAI;
        # collapse the OpenAI-style message list into a single prompt.
        prompt = "\n\n".join(m["content"] for m in messages)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature
            )
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")
