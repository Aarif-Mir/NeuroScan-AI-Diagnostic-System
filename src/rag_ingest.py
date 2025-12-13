"""
Ingest medical PDFs into a persistent Chroma vector store
(using HuggingFace embeddings – no TensorFlow).

PDFs location:
    data/rag_pdfs/

Vector DB:
    chroma_db/   (local only, NOT committed to GitHub)
"""


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

def ingest_pdfs(
    pdf_dir: str,
    persist_dir: str = "chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 300,
):
    """
    Ingest PDFs and persist embeddings in Chroma.

    Args:
        pdf_dir (str): Directory containing PDFs
        persist_dir (str): Chroma persistence directory
        chunk_size (int): Characters per chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        retriever: LangChain retriever
    """

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    # LangChain-compatible embedding model 
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = []

    # Load PDFs
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, fname)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Metadata
        for i, doc in enumerate(docs):
            doc.metadata["source"] = fname
            doc.metadata["page"] = i

        documents.extend(docs)

    if not documents:
        raise ValueError("No PDF text extracted.")

    #  Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(documents)

    # Persistent Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="rag_collection"
    )


    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Example usage

if __name__ == "__main__":
    from .config import PDF_DIR
    
    retriever = ingest_pdfs(pdf_dir=str(PDF_DIR))
    
    query = "What are CT findings for brain hemorrhage?"
    docs = retriever.invoke(query)
    
    for d in docs:
        print(f"\nSOURCE: {d.metadata['source']} | Page: {d.metadata['page']}")
        print(d.page_content[:200], "...")




# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import Chroma

# def ingest_pdfs_inmemory(pdf_dir, embedding_model, chunk_size=1000, chunk_overlap=300):
#     """
#     Ingest PDFs from a folder and create an in-memory Chroma retriever.

#     Args:
#         pdf_dir (str): Directory containing PDF files
#         embedding_model: LangChain embedding model or SentenceTransformer
#         chunk_size (int): Number of characters per chunk
#         chunk_overlap (int): Overlap characters between chunks

#     Returns:
#         retriever: LangChain retriever ready for RAG queries
#     """

#     if not os.path.exists(pdf_dir):
#         raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

#     documents = []

#     # Loop through all PDFs in the folder
#     for fname in os.listdir(pdf_dir):
#         if not fname.lower().endswith(".pdf"):
#             continue

#         pdf_path = os.path.join(pdf_dir, fname)
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()

#         # Add metadata for source file and chunk tracking
#         for i, doc in enumerate(docs):
#             doc.metadata["source"] = fname
#             doc.metadata["chunk_index"] = i

#         documents.extend(docs)

#     if not documents:
#         raise ValueError("No PDF text extracted from the folder.")

#     # Split large documents into smaller chunks
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     splits = splitter.split_documents(documents)

#     # Create in-memory Chroma vector store
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embedding_model,
#         persist_directory="chroma_db",
#         collection_name="rag_collection"
#     )

#     # Return retriever for queries
#     retriever = vectorstore.as_retriever()
#     return retriever


# # Load embedding model
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 
# from .config import PDF_DIR
# pdf_dir = str(PDF_DIR)
# 
# # Create retriever
# retriever = ingest_pdfs_inmemory(pdf_dir, embedding_model)

# # Example query
# query = "What are CT findings for brain hemorrhage?"
# docs = retriever.get_relevant_documents(query)
# for d in docs:
#     print(d.metadata["source"], d.page_content[:200], "...")





