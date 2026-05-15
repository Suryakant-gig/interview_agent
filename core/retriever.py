def infer_filters(query: str):
    q = query.lower()
    flt = {}

    if "deep learning" in q or "cnn" in q or "rnn" in q:
        flt["topic"] = "DL"
    elif "machine learning" in q or "regression" in q or "svm" in q:
        flt["topic"] = "ML"

    return flt or None

def get_retriever(vectorstore, query: str, k: int = 20):
    flt = infer_filters(query)

    if flt:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": flt}
        )
    else:
        return vectorstore.as_retriever(
            search_kwargs={"k": k}
        )