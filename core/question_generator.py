from core.grok_client import grok_chat


def build_question_prompt(context, difficulty="medium"):
    return f"""You are an expert AI technical interviewer.

Your task is to generate ONE high-quality technical interview question from the provided reference context.

-----------------------------------
REFERENCE CONTEXT
-----------------------------------

{context}

-----------------------------------
INSTRUCTIONS
-----------------------------------

1. Generate ONLY one question.
2. The question must be technical.
3. The question should test understanding.
4. Avoid trivial definitions.
5. Keep the difficulty level: {difficulty}
6. Do not include answers.
7. Be concise and interview-style.

-----------------------------------
OUTPUT FORMAT
-----------------------------------

Return ONLY the interview question."""


def generate_question(context, difficulty="medium"):
    prompt = build_question_prompt(context=context, difficulty=difficulty)
    messages = [{"role": "user", "content": prompt}]
    question = grok_chat(messages, temperature=0.7)
    return question