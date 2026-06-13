from core.embeddings import get_embedding_model
from core.vector_store import get_vector_store

embedding_model = get_embedding_model()
vectorstore = get_vector_store(embedding_model)