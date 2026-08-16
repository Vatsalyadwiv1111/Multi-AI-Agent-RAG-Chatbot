import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.pdf_processor import load_pdf
from tools.url_retriever import retrieve_url_content
from config.settings import settings

vector_store = None
current_files_hash = ""

def get_retriever(files, urls):
    """
    Creates or updates a vector store from the provided files and URLs.
    Thread-safe vector retrieval factory.
    """
    global vector_store, current_files_hash

    new_hash = str(sorted(files or [])) + str(sorted(urls or []))

    if vector_store is not None and new_hash == current_files_hash:
        return vector_store.as_retriever(search_kwargs={"k": 3})

    documents = []

    if files:
        for file_path in files:
            try:
                docs = load_pdf(file_path)
                for doc in docs:
                    doc.metadata["source"] = f"PDF - {os.path.basename(file_path)}"
                documents.extend(docs)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if urls:
        for url in urls:
            try:
                docs = retrieve_url_content(url)
                for doc in docs:
                    doc.metadata["source"] = f"URL - {url}"
                documents.extend(docs)
            except Exception as e:
                print(f"Error reading {url}: {e}")

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(splits, embeddings)
    current_files_hash = new_hash

    return vector_store.as_retriever(search_kwargs={"k": 3})
