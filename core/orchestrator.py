from core.loader import (
    load_documents
)

from core.chunker import (
    split_documents
)

from core.embeddings import (
    get_embedding_model
)

from core.vector_store import (
    get_vector_store,
    add_documents
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
# 1. Load documents
# ---------------------------------
documents = load_documents()

print(
    f"Loaded docs: {len(documents)}"
)


# ---------------------------------
# 2. Split into chunks
# ---------------------------------
chunks = split_documents(
    documents
)

print(
    f"Created chunks: {len(chunks)}"
)


# ---------------------------------
# 3. Load embedding model
# ---------------------------------
embedding_model = (
    get_embedding_model()
)

print(
    "Embedding model loaded"
)


# ---------------------------------
# 4. Connect vector store
# ---------------------------------
vectorstore = get_vector_store(
    embedding_model
)

print(
    "Vector store connected"
)


# ---------------------------------
# 5. Upload chunks
# ---------------------------------
add_documents(
    vectorstore,
    chunks
)

print(
    "Chunks uploaded"
)


# ---------------------------------
# 6. User question
# ---------------------------------
question = (
    "Explain CNN architecture"
)


candidate_answer = (
    "CNN is a neural network "
    "used mainly for images."
)


ideal_answer = (
    "CNN is a deep learning "
    "architecture using "
    "convolution layers for "
    "spatial feature extraction."
)


rubric = [

    "concept clarity",

    "key points covered",

    "example or explanation"

]


# ---------------------------------
# 7. Create retriever
# ---------------------------------
retriever = get_retriever(
    vectorstore,
    question
)

print(
    "Retriever ready"
)


# ---------------------------------
# 8. Retrieve candidate docs
# ---------------------------------
retrieved_docs = retriever.invoke(
    question
)

print(
    f"Retrieved docs: "
    f"{len(retrieved_docs)}"
)


# ---------------------------------
# 9. Rerank docs
# ---------------------------------
reranked_docs = rerank_documents(

    query=question,

    docs=retrieved_docs,

    top_k=5
)

print(
    f"Reranked docs: "
    f"{len(reranked_docs)}"
)


# ---------------------------------
# 10. Build context
# ---------------------------------
context = build_context(
    reranked_docs
)

print("\nContext Built\n")

print(context[:1000])


# ---------------------------------
# 11. Evaluate answer
# ---------------------------------
result = evaluate_answer(

    question=question,

    candidate_answer=
    candidate_answer,

    ideal_answer=
    ideal_answer,

    rubric=rubric,

    context=context
)


# ---------------------------------
# 12. Final result
# ---------------------------------
print("\nEvaluation Result\n")

print(result)