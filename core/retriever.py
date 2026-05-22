from sklearn.metrics.pairwise import (
    cosine_similarity
)


# ---------------------------------
# Topic prototypes
# ---------------------------------
TOPIC_PROTOTYPES = {

    "DL":
        "deep learning neural networks CNN RNN transformers attention",

    "ML":
        "machine learning regression svm clustering decision trees random forest"
}


# ---------------------------------
# Semantic topic routing
# ---------------------------------
def infer_topic(
    query,
    embedding_model
):

    # ---------------------------------
    # Embed query
    # ---------------------------------
    query_embedding = (
        embedding_model.embed_query(query)
    )

    best_topic = None

    best_score = -1

    # ---------------------------------
    # Compare against prototypes
    # ---------------------------------
    for topic, prototype_text in (
        TOPIC_PROTOTYPES.items()
    ):

        prototype_embedding = (
            embedding_model.embed_query(
                prototype_text
            )
        )

        score = cosine_similarity(

            [query_embedding],
            [prototype_embedding]

        )[0][0]

        if score > best_score:

            best_score = score

            best_topic = topic

    print(
        f"Predicted Topic: {best_topic}"
    )

    print(
        f"Similarity Score: {best_score}"
    )

    return best_topic


# ---------------------------------
# Retriever builder
# ---------------------------------
def get_retriever(
    vectorstore,
    query,
    embedding_model,
    k: int = 20
):

    # ---------------------------------
    # Infer topic
    # ---------------------------------
    topic = infer_topic(
        query,
        embedding_model
    )

    # ---------------------------------
    # Build retriever
    # ---------------------------------
    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k": k,

            "filter": {
                "topic": topic
            }
        }
    )

    return retriever