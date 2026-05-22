from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from pathlib import Path

import shutil

from langchain_community.document_loaders import (
    PyPDFLoader
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


# ---------------------------------
# Router
# ---------------------------------
router = APIRouter()


# ---------------------------------
# Upload directory
# ---------------------------------
UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------
# Load embedding model once
# ---------------------------------
embedding_model = (
    get_embedding_model()
)


# ---------------------------------
# Connect vector store once
# ---------------------------------
vectorstore = get_vector_store(
    embedding_model
)

@router.post("/upload-pdf")
def upload_pdf(
    file: UploadFile = File(...)
):

    save_path = (
        UPLOAD_DIR / file.filename
    )

    with open(save_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # ---------------------------------
    # Load PDF
    # ---------------------------------
    loader = PyPDFLoader(
        str(save_path)
    )

    documents = loader.load()

    # ---------------------------------
    # Add metadata
    # ---------------------------------
    for i, doc in enumerate(documents):

        doc.metadata.update({

            "file": file.filename,

            "page": doc.metadata.get(
                "page",
                -1
            ),

            "topic": "custom_upload",

            "doc_id": (
                f"{file.filename}_page_{i}"
            )
        })

    # ---------------------------------
    # Split documents
    # ---------------------------------
    chunks = split_documents(
        documents
    )

    # ---------------------------------
    # Upload chunks to vector DB
    # ---------------------------------
    add_documents(
        vectorstore,
        chunks
    )

    # ---------------------------------
    # Return success response
    # ---------------------------------
    return {

        "status": "success",

        "filename": file.filename,

        "pages_loaded": len(documents),

        "chunks_created": len(chunks)
    }