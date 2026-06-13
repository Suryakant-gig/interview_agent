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

from core.vector_store import (
    add_documents
)

from core.retriever import (
    infer_topic
)

from core.model_manager import (
    embedding_model,
    vectorstore
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
# Upload PDF Route
# ---------------------------------
@router.post("/upload-pdf")
def upload_pdf(
    file: UploadFile = File(...)
):

    try:

        # ---------------------------------
        # Save uploaded file
        # ---------------------------------
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
        # Safety check
        # ---------------------------------
        if not documents:

            raise HTTPException(

                status_code=400,

                detail="No text could be extracted from PDF"
            )

        # ---------------------------------
        # Split documents into chunks
        # ---------------------------------
        chunks = split_documents(
            documents
        )

        # ---------------------------------
        # Add semantic metadata
        # ---------------------------------
        for i, chunk in enumerate(chunks):

            # ---------------------------------
            # Infer semantic topic
            # ---------------------------------
            content = chunk.page_content.strip()

            if not content:

                topic = "UNKNOWN"

            else:

                topic = infer_topic(
                    content,
                    embedding_model
                )
            # ---------------------------------
            # Update metadata
            chunk.metadata.update({

                "file": file.filename,

                "page": chunk.metadata.get(
                    "page",
                    -1
                ),

                "topic": topic,

                "doc_id": (
                    f"{file.filename}_chunk_{i}"
                )
            })
        # ---------------------------------
        # Upload to Pinecone
        # ---------------------------------
        add_documents(

            vectorstore,

            chunks
        )

        # ---------------------------------
        # Success response
        # ---------------------------------
        return {

            "status": "success",

            "filename": file.filename,

            "pages_loaded": len(documents),

            "chunks_created": len(chunks)
        }

    except Exception as e:

        print(f"Upload Error: {e}")

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )