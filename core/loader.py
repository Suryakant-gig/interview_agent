from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os

def load_documents():
    loader = DirectoryLoader(
        path="knowledge_pdfs/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

    for d in docs:
        src = d.metadata.get("source", "")
        fname = os.path.basename(src)

        # topic tagging (extend as needed)
        if "Deep Learning" in fname:
            topic = "DL"
        else:
            topic = "ML"

        d.metadata.update({
            "topic": topic,
            "file": fname,
            "page": d.metadata.get("page", -1)
        })
    return docs