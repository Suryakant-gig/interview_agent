import json
import re
from core.grok_client import grok_chat


def build_evaluation_prompt(question, candidate_answer, rubric, context):
    rubric_text = "\n".join([f"- {r}" for r in rubric])
    return f"""You are an expert AI technical interviewer.

Your task is to evaluate a candidate answer using:
1. Ideal reference answer
2. Evaluation rubric
3. Retrieved technical context

-----------------------------------
QUESTION
-----------------------------------
{question}

-----------------------------------
CANDIDATE ANSWER
-----------------------------------
{candidate_answer}
-----------------------------------
RUBRIC
-----------------------------------
{rubric_text}

-----------------------------------
REFERENCE CONTEXT
-----------------------------------
{context}

-----------------------------------
INSTRUCTIONS
-----------------------------------

1. Evaluate the answer carefully.
2. Compare candidate answer with ideal answer.
3. Use the rubric strictly.
4. Detect missing concepts.
5. Be concise and factual.
6. Avoid hallucinations.

-----------------------------------
OUTPUT FORMAT
-----------------------------------

Return ONLY valid JSON.

{{
  "overall_score": 8,
  "rubric_scores": {{
      "concept clarity": 8,
      "key points covered": 7,
      "example or explanation": 6
  }},
  "strengths": [
      "Good explanation of boosting"
  ],
  "weaknesses": [
      "Did not explain corrective loop clearly"
  ],
  "missing_concepts": [
      "Difference between bagging and boosting"
  ],
  "improvement": "Add clearer explanation of iterative error correction."
}}"""


def evaluate_answer(question, candidate_answer, rubric, context):
    prompt = build_evaluation_prompt(question, candidate_answer, rubric, context)
    messages = [{"role": "user", "content": prompt}]

    try:
        raw_output = grok_chat(messages, temperature=0.0)
    except RuntimeError as e:
        return {
            "overall_score": 0,
            "rubric_scores": {},
            "strengths": [],
            "weaknesses": [],
            "missing_concepts": [],
            "improvement": f"Grok API call failed: {e}",
            "raw_output": ""
        }

    print("\nRAW MODEL OUTPUT:\n", raw_output)

    cleaned = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        try:
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("No JSON found")
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return {
                "overall_score": 0,
                "rubric_scores": {},
                "strengths": [],
                "weaknesses": [],
                "missing_concepts": [],
                "improvement": "Failed to parse model output",
                "raw_output": raw_output
            }