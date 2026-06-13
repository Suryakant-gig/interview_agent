from langchain_ollama import ChatOllama


# ---------------------------------
# Load LLM
# ---------------------------------
llm = ChatOllama(
    model="llama2",
    temperature=0.7
)


# ---------------------------------
# Prompt Builder
# ---------------------------------
def build_question_prompt(
    context,
    difficulty="medium"
):
    """
    Build interview question generation prompt.
    """

    prompt = f"""
You are an expert AI technical interviewer.

Your task is to generate ONE high-quality
technical interview question from the
provided reference context.

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

Return ONLY the interview question.
"""

    return prompt


# ---------------------------------
# Main Generator
# ---------------------------------
def generate_question(
    context,
    difficulty="medium"
):
    """
    Generate interview question
    from retrieved context.
    """

    # ---------------------------------
    # Build prompt
    # ---------------------------------
    prompt = build_question_prompt(
        context=context,
        difficulty=difficulty
    )

    # ---------------------------------
    # Call LLM
    # ---------------------------------
    response = llm.invoke(prompt)

    question = response.content.strip()

    return question