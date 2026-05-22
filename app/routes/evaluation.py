from fastapi import APIRouter
from pydantic import BaseModel

from core.embeddings import (
    get_embedding_model
)

from core.vector_store import (
    get_vector_store
)

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

    ideal_answer: str

    rubric: list[str]


# ---------------------------------
# Load embedding model once
# ---------------------------------
embedding_model = (
    get_embedding_model()
)


# ---------------------------------
# Connect vector DB once
# ---------------------------------
vectorstore = get_vector_store(
    embedding_model
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

        embedding_model
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

        ideal_answer=(
            request.ideal_answer
        ),

        rubric=request.rubric,

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