import os

from dotenv import load_dotenv

from pinecone import Pinecone

from langchain_pinecone import (
    PineconeVectorStore
)


# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()


# ---------------------------------
# Pinecone settings
# ---------------------------------
INDEX_NAME = "interview-agent"


# ---------------------------------
# Initialize Pinecone client
# ---------------------------------
pc = Pinecone(
    api_key=os.getenv(
        "PINECONE_API_KEY"
    )
)


# ---------------------------------
# Connect to Pinecone vector DB
# ---------------------------------
def get_vector_store(
    embedding_model
):
    """
    Connect to Pinecone index.
    """

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_model
    )

    print("Connected to Pinecone")

    return vectorstore


# ---------------------------------
# Add documents to Pinecone
# ---------------------------------
def add_documents(
    vectorstore,
    chunks
):
    """
    Upload chunks + embeddings
    to Pinecone cloud.
    """

    # ---------------------------------
    # Get existing IDs
    # ---------------------------------
    existing_ids = set()

    try:

        stats = vectorstore.index.describe_index_stats()

        # namespaces may be empty initially
        namespaces = stats.get(
            "namespaces",
            {}
        )

        # optional handling
        print("Index stats loaded")

    except Exception:
        pass

    # ---------------------------------
    # Prepare unique chunks
    # ---------------------------------
    new_chunks = []

    ids = []

    for chunk in chunks:

        chunk_id = chunk.metadata.get(
            "chunk_id"
        )

        if chunk_id:

            new_chunks.append(chunk)

            ids.append(chunk_id)

    # ---------------------------------
    # Upload to Pinecone
    # ---------------------------------
    if len(new_chunks) > 0:

        vectorstore.add_documents(
            documents=new_chunks,
            ids=ids
        )

        print(
            f"Uploaded {len(new_chunks)} chunks"
        )

    else:

        print(
            "No chunks to upload"
        )


# ---------------------------------
# Get Pinecone statistics
# ---------------------------------
def get_vectorstore_info(
    vectorstore
):
    """
    Show Pinecone statistics.
    """

    stats = (
        vectorstore.index
        .describe_index_stats()
    )

    print("\nPinecone Index Info")
    print("-------------------")

    print(stats)

    return stats