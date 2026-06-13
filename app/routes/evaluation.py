from fastapi import APIRouter
from pydantic import BaseModel

from core.retriever import (
    get_retriever
)

from core.reranker import (
    rerank_documents
)

from core.context_builder import (
    build_context
)

from core.evaluator import (
    evaluate_answer
)


# ---------------------------------
# Router
# ---------------------------------
router = APIRouter()


# ---------------------------------
# Request schema
# ---------------------------------
class EvaluationRequest(BaseModel):

    question: str

    candidate_answer: str

    file_name: str

rubric = [
    "technical accuracy",
    "clarity",
    "depth of explanation"
]
from core.model_manager import (
    embedding_model,
    vectorstore
)

# ---------------------------------
# Evaluation Route
# ---------------------------------
@router.post("/evaluate-answer")
def evaluate_candidate_answer(
    request: EvaluationRequest
):

    # ---------------------------------
    # 1. Create retriever
    # ---------------------------------
    retriever = get_retriever(

        vectorstore,

        request.question,

        embedding_model,

        request.file_name
    )
    # ---------------------------------
    # 2. Retrieve documents
    # ---------------------------------
    retrieved_docs = retriever.invoke(
        request.question
    )

    # ---------------------------------
    # 3. Rerank documents
    # ---------------------------------
    reranked_docs = rerank_documents(
        request.question,
        retrieved_docs
    )

    # ---------------------------------
    # 4. Build context
    # ---------------------------------
    context = build_context(
        reranked_docs
    )

    # ---------------------------------
    # 5. Evaluate answer
    # ---------------------------------
    result = evaluate_answer(
        question=request.question,

        candidate_answer=(
            request.candidate_answer
        ),

        rubric=rubric,

        context=context
    )
    # ---------------------------------
    # 6. Return response
    # ---------------------------------
    return {

        "status": "success",

        "question": request.question,

        "evaluation": result
    }