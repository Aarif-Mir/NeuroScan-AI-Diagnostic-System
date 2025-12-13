"""
Load persisted Chroma vector store and create a modern
LangChain RAG pipeline using latest LangChain APIs.
"""

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# from langchain_anthropic import ChatAnthropic

from langchain_groq import ChatGroq


from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.chains.combine_documents import 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

from .config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RAG_COLLECTION_NAME,
    LLM_MODEL_NAME,
    RAG_K_RETRIEVE,
)


def build_rag_chain(k: int = None):
    """
    Build a modern RAG pipeline using latest LangChain APIs.

    Returns:
        rag_chain: Runnable RAG chain
        retriever: VectorStoreRetriever
    """

    # Embeddings (same as ingestion)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Use k parameter or default from config
    k_retrieve = k if k is not None else RAG_K_RETRIEVE

    #  Load persisted Chroma DB
    vectordb = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=RAG_COLLECTION_NAME,
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": k_retrieve}
    )

    #  LLM
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=LLM_MODEL_NAME
    )

    # Enhanced medical RAG prompt with strict anti-hallucination constraints
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert medical AI assistant specializing in radiology and diagnostic imaging. 
Your role is to provide evidence-based medical information STRICTLY from the provided context documents.

CRITICAL RULES:
1. ANSWER ONLY FROM CONTEXT: Use ONLY information explicitly stated in the provided context documents.
2. NO HALLUCINATION: Never invent, assume, or infer information not present in the context.
3. NO GENERAL KNOWLEDGE: Do not use your training data - only use the provided medical literature.
4. EXPLICIT CITATIONS: For every factual claim, cite the source document and page number in format: [Source: filename.pdf, Page: X]
5. UNCERTAINTY HANDLING: If information is missing or unclear, explicitly state: "Information not found in provided documents" or "Insufficient information in context"
6. MEDICAL PRECISION: Use exact medical terminology from the context. Maintain clinical accuracy.
7. STRUCTURED OUTPUT: Organize information clearly with headers, bullet points, and proper formatting.
8. PRIORITIZE EVIDENCE: When multiple sources conflict, mention all perspectives and cite each source.
9. CLINICAL CONTEXT: Frame information in appropriate clinical context when available in documents.
10. SAFETY FIRST: For treatment recommendations, emphasize that this is informational only and requires clinical consultation.

OUTPUT FORMAT:
- Use clear section headers (##)
- Use bullet points for lists
- Include citations in brackets: [Source: X, Page: Y]
- Be concise but comprehensive
- Maintain professional medical writing style

Remember: Patient safety depends on accuracy. When in doubt, state uncertainty explicitly.""",
        ),
        ("human",
         """CONTEXT DOCUMENTS:
{context}

USER QUERY:
{input}

INSTRUCTIONS:
Analyze the context documents above and answer the query using ONLY the information provided. 
Structure your response clearly, cite all sources, and explicitly state if any requested information is not available in the documents.""",
        )
    ])

    # Document → Answer chain
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    # Retriever → LLM chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
    )

    return rag_chain, retriever


# Test run
# if __name__ == "__main__":

#     rag_chain, retriever = build_rag_chain()

#     query = "summary of the report?"

#     result = rag_chain.invoke({"input": query})

#     print("\nANSWER:\n", result["answer"])

#     print("\nSOURCES:")
#     for doc in result["context"]:
#         print(
#             f"- {doc.metadata['source']} | Page {doc.metadata['page']}"
        # )
