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

from core.question_generator import (
    generate_question
)


# ---------------------------------
# Router
# ---------------------------------
router = APIRouter()

from core.model_manager import (
    embedding_model,
    vectorstore
)

# ---------------------------------
# Request schema
# ---------------------------------
class InterviewRequest(BaseModel):

    topic_query: str

    file_name: str

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

        embedding_model,

        request.file_name
    )

    # ---------------------------------
    # Retrieve documents
    # ---------------------------------
    retrieved_docs = retriever.invoke(
        request.topic_query
    )
    for doc in retrieved_docs[:3]:
        print("\n====================")
        print(doc.metadata)
        print(doc.page_content[:300])

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