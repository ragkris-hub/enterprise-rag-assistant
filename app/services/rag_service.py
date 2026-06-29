from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config
from app.llm.factory import get_llm_provider


SYSTEM_PROMPT = """
You are an enterprise IT support assistant.

Answer only using the provided context.
If the answer is not present in the context, say:
"I could not find this in the available knowledge base."

Always include concise citations using the source file name.
"""


def ask_question(question: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    vector_store = Chroma(
        persist_directory=config.VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    docs = vector_store.similarity_search(
        question,
        k=config.TOP_K
    )

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )

    final_prompt = f"""
{SYSTEM_PROMPT}

Question:
{question}

Context:
{context}
"""

    llm_provider = get_llm_provider()
    answer = llm_provider.generate(final_prompt)

    citations = list(set(doc.metadata.get("source", "unknown") for doc in docs))

    return {
    "question": question,
    "answer": answer,
    "citations": citations,
    "retrieved_chunks": [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]
}