import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from collections import defaultdict
from langchain_core.documents import Document

load_dotenv()

def load_docs(docs_path):

    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        show_progress = True,
    )

    documents = loader.load()

    # for i, doc in enumerate(documents):  
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")

    return documents

def split_docs(docs, chunk_size=1000, chunk_overlap=0):

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap,
        )
    
    # Better intuition program
    # docs_chunks = defaultdict(list)
    # docs_chunks = []
    # for doc in docs:
    #     chunks_for_doc = text_splitter.split_documents([doc])
    #     for i, chunk in enumerate(chunks_for_doc):
    #         chunk.metadata["source"] = doc.metadata.get("source")
    #         chunk.metadata.update({"source": doc.metadata.get("source"), "chunk_id": i})
    #         docs_chunks.append(chunk)
    #     print(f"Source: {doc.metadata.get('source')} -> {len(chunks_for_doc)} chunks")

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

def create_vector_store(chunks, directory="db/chroma_db"):

    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    # docs = [Document(page_content=text) for text in docs]
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=directory,
        collection_metadata={"hnsw:space": "cosine"} # ?
    )

    return vector_store

def main():
    documents = load_docs(docs_path="docs")
    chunks = split_docs(documents)
    vector = create_vector_store(chunks)

if __name__ == "__main__":
    main()