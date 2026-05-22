import tiktoken
from typing import List


# -----------------------------
# Token counter (important)
# -----------------------------
def count_tokens(text: str, model: str = "gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# -----------------------------
# Main Context Builder
# -----------------------------
def build_context(
    docs: List,
    max_tokens: int = 1200,
    max_chunks: int = 10,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Converts retrieved docs into structured, token-limited context.

    Steps:
    1. Deduplicate
    2. Filter weak chunks
    3. Limit chunks
    4. Token-aware trimming
    5. Structured formatting
    """

    # -----------------------------
    # 1. Deduplicate (file + page)
    # -----------------------------
    seen = set()
    unique_docs = []

    for d in docs:
        key = (
            d.metadata.get("file"),
            d.metadata.get("page")
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)


    docs = sorted(
        docs,
        key=lambda d:
        d.metadata.get(
            "rerank_score",
            0
        ),
        reverse=True
    )

    # -----------------------------
    # 2. Filter weak chunks
    # -----------------------------
    filtered_docs = []
    for d in unique_docs:
        text = d.page_content.strip()

        # remove very small chunks
        if len(text) < 50:
            continue

        filtered_docs.append(d)

    # -----------------------------
    # 3. Limit number of chunks
    # -----------------------------
    docs = filtered_docs[:max_chunks]

    # -----------------------------
    # 4. Build context with token control
    # -----------------------------
    context_parts = []
    total_tokens = 0

    for d in docs:
        file = d.metadata.get("file", "unknown")
        page = d.metadata.get("page", -1)
        topic = d.metadata.get("topic", "")

        chunk_text = d.page_content.strip()

        # 1. format first
        formatted = (
            f"[{topic} | {file} | Page {page}]\n"
            f"{chunk_text}"
        )

        # 2. count tokens
        tokens = count_tokens(formatted, model=model)

        remaining = max_tokens - total_tokens
        if remaining <= 0:
            break

        # 3. trim if needed
        if tokens > remaining:
            ratio = remaining / tokens
            chunk_text = chunk_text[: int(len(chunk_text) * ratio)]

            formatted = (
                f"[{topic} | {file} | Page {page}]\n"
                f"{chunk_text}"
            )

            tokens = count_tokens(formatted, model=model)

        # 4. final safety check
        if total_tokens + tokens > max_tokens:
            break

        # 5. add to context
        context_parts.append(formatted)
        total_tokens += tokens
    # -----------------------------
    # 5. Final context
    # -----------------------------
    context = "\n\n".join(context_parts)

    return context