from sklearn.metrics.pairwise import (
    cosine_similarity
)


TOPIC_PROTOTYPES = {

    "DL":
        "deep learning neural networks CNN RNN transformers attention backpropagation",

    "ML":
        "machine learning regression svm clustering decision trees random forest"
}


# ---------------------------------
# Infer Topic
# ---------------------------------
def infer_topic(
    query,
    embedding_model
):

    query_embedding = (
        embedding_model.embed_query(query)
    )

    best_topic = None

    best_score = -1

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

    print(f"Predicted Topic: {best_topic}")
    print(f"Similarity Score: {best_score}")

    return best_topic


# ---------------------------------
# Hybrid Retriever
# ---------------------------------
def get_retriever(

            vectorstore,
            query,
            embedding_model,
            file_name,
            k: int = 20
    ):

    # ---------------------------------
    # Infer topic
    # ---------------------------------
    topic = infer_topic(
        query,
        embedding_model
    )

    print(f"Using Topic: {topic}")

    # ---------------------------------
    # Retrieve only from uploaded file
    # ---------------------------------
    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={

            "k": k,

            "filter": {

                "file": file_name,

                "topic": topic
            }
        }
    )

    return retriever