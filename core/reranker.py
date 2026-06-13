from rank_bm25 import BM25Okapi


def rerank_documents(query, docs, top_k=5, score_threshold=0.0):
    if not docs:
        return []

    tokenized_corpus = [
        doc.page_content.lower().split() for doc in docs
    ]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_docs = []
    for score, doc in ranked:
        if score < score_threshold:
            continue
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)
        if len(reranked_docs) >= top_k:
            break

    return reranked_docs