from langchain_text_splitters import TokenTextSplitter


def split_documents(
    documents,
    chunk_size: int = 200,
    chunk_overlap: int = 40
):
    """
    Token-based document chunking.
    """

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks