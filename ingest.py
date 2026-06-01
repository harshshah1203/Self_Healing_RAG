import os
from pathlib import Path 
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space" : "cosine"}
)

def load_and_split_documents(docs_folder: str) -> list:
    """Load all .txt files from a folder and split into chunks."""
    all_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap = 50,
        length_function=len,
    )
    
    docs_path = Path(docs_folder)
    for txt_file in docs_path.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {txt_file.name}")
    
    return all_chunks

def ingest_to_chroma(chunks: list):
    """Store documetnts chunks in ChromaDB."""
    
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadata = [
        {
            "source": chunk.metadata.get("source", "unknown")
        }
        for chunk in chunks
    ]
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadata
    )
    print(f" Stored {len(documents)} chunks in ChromaDB.")


if __name__ == "__main__":
    print("Starting document ingestion...")
    chunks = load_and_split_documents("docs")
    
    if not chunks:
        print("No .txt files found in docs/ folder")
    else:
        ingest_to_chroma(chunks)
        print(f"\nDone! Total chunks ingested: {len(chunks)}")
        print("Your knowledge base is Ready.")