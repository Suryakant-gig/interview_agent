from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

import os


# ---------------------------------
# Topic mapping
# ---------------------------------
TOPIC_MAP = {
    "deep learning": "DL",
    "machine learning": "ML",
    "nlp": "NLP",
    "computer vision": "CV",
    "transformer": "NLP"
}


# ---------------------------------
# Main loader
# ---------------------------------
def load_documents():

    try:

        loader = DirectoryLoader(
            path="knowledge_pdfs/",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        docs = loader.load()

    except Exception as e:

        print(f"Error loading PDFs: {e}")

        return []

    # ---------------------------------
    # Metadata enrichment
    # ---------------------------------
    for d in docs:

        src = d.metadata.get("source", "")

        fname = os.path.basename(src)

        fname_lower = fname.lower()

        # ---------------------------------
        # Dynamic topic detection
        # ---------------------------------
        topic = "GENERAL"

        for key, value in TOPIC_MAP.items():

            if key in fname_lower:
                topic = value
                break

        # ---------------------------------
        # Page number
        # ---------------------------------
        page = d.metadata.get("page", -1)

        # ---------------------------------
        # Unique document ID
        # ---------------------------------
        doc_id = f"{fname}_page_{page}"

        # ---------------------------------
        # Update metadata
        # ---------------------------------
        d.metadata.update({
            "topic": topic,
            "file": fname,
            "page": page,
            "doc_id": doc_id
        })

    return docs