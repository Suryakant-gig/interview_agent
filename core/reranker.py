from sentence_transformers import (
    CrossEncoder
)


# ---------------------------------
# Load reranker model
# ---------------------------------
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------
# Rerank documents
# ---------------------------------
def rerank_documents(
    query,
    docs,
    top_k=5,
    score_threshold=0.3
):
    """
    Rerank retrieved documents
    using CrossEncoder.
    """

    # ---------------------------------
    # Empty guard
    # ---------------------------------
    if not docs:
        return []

    # ---------------------------------
    # Query-document pairs
    # ---------------------------------
    pairs = [

        (query, d.page_content)

        for d in docs
    ]

    # ---------------------------------
    # Predict relevance scores
    # ---------------------------------
    scores = reranker_model.predict(
        pairs
    )

    # ---------------------------------
    # Combine scores + docs
    # ---------------------------------
    ranked = sorted(

        zip(scores, docs),

        key=lambda x: x[0],

        reverse=True
    )

    reranked_docs = []

    # ---------------------------------
    # Keep strong documents only
    # ---------------------------------
    for score, doc in ranked:

        if score < score_threshold:
            continue

        doc.metadata[
            "rerank_score"
        ] = float(score)

        reranked_docs.append(doc)

        if len(reranked_docs) >= top_k:
            break

    return reranked_docs