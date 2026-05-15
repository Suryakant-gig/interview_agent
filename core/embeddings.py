from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="knowledge_pdfs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

for doc in documents:
    if "Deep Learning" in doc.metadata["source"]:
        doc.metadata["topic"] = "DL"
    else:
        doc.metadata["topic"] = "ML"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=70,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_text(documents)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.embed_documents(chunks)