from fastapi import APIRouter, HTTPException
from app.schemas import AnswerRequest
from core.orchestrator import evaluate_answer

router = APIRouter(prefix="/api", tags=["Evaluation"])


@router.post("/evaluate")
def evaluate(req: AnswerRequest):
    try:
        result = evaluate_answer(req.question, req.answer)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))