import os
from langchain_community.embeddings import FastEmbedEmbeddings


def get_embedding_model():
    embedding_model = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return embedding_model