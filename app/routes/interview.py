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

from core.question_generator import (
    generate_question
)


# ---------------------------------
# Router
# ---------------------------------
router = APIRouter()


# ---------------------------------
# Load models once
# ---------------------------------
embedding_model = (
    get_embedding_model()
)

vectorstore = get_vector_store(
    embedding_model
)


# ---------------------------------
# Request schema
# ---------------------------------
class InterviewRequest(BaseModel):

    topic_query: str

    difficulty: str = "medium"


# ---------------------------------
# Generate Interview Question
# ---------------------------------
@router.post("/generate-question")
def generate_interview_question(
    request: InterviewRequest
):

    # ---------------------------------
    # Create retriever
    # ---------------------------------
    retriever = get_retriever(

        vectorstore,

        request.topic_query,

        embedding_model
    )

    # ---------------------------------
    # Retrieve documents
    # ---------------------------------
    retrieved_docs = retriever.invoke(
        request.topic_query
    )

    # ---------------------------------
    # Safety check
    # ---------------------------------
    if not retrieved_docs:

        return {
            "error":
            "No relevant documents found"
        }

    # ---------------------------------
    # Rerank docs
    # ---------------------------------
    reranked_docs = rerank_documents(

        request.topic_query,

        retrieved_docs
    )

    # ---------------------------------
    # Build context
    # ---------------------------------
    context = build_context(
        reranked_docs
    )

    # ---------------------------------
    # Generate question
    # ---------------------------------
    question = generate_question(

        context=context,

        difficulty=request.difficulty
    )

    # ---------------------------------
    # Return response
    # ---------------------------------
    return {

        "generated_question":
            question
    }