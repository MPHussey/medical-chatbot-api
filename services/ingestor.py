import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma  
from core.config import EMBED_MODEL, CHROMA_PATH, COLLECTION_NAME

def ingest_document(file_path: str):
    # Load
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Wipe existing collection and re-create (replace logic)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Store
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    return len(chunks)