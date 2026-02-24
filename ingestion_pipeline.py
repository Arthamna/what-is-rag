import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from collections import defaultdict

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

def main():
    documents = load_docs(docs_path="docs")
    chunks = split_docs(documents)
    return documents

def split_docs(docs, chunk_size=1000, chunk_overlap=0):

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap,
        )
    
    # chunks = text_splitter.split_documents(docs)
    # for chunk in chunks:
    #     docs_chunks[chunk.metadata.get("source")].append(chunk)

    # for name, chunk_items in docs_chunks.items():
    #     print(f"  Source: {name}")
    #     print(f"total chunks is {len(chunk_items)}")
    
    # Better intuition program
    docs_chunks = defaultdict(list)

    for doc in docs:
        chunks_for_doc = text_splitter.split_documents([doc])
        docs_chunks[doc.metadata.get("source")] = chunks_for_doc
        print(f"Source: {doc.metadata.get('source')} -> {len(chunks_for_doc)} chunks")

    return docs_chunks

def create_vector_store(chunks, directory="db/chroma_db"):
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=directory,
        collection_metadata={"hnsw:space": "cosine"} # ?
    )

    return vector_store

if __name__ == "__main__":
    main()