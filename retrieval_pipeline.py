import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from collections import defaultdict

load_dotenv()
db_dir = "db/chroma_db"

embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=db_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# EXAMPLE QUERY :
query = "Which island does SpaceX lease for its launches in the Pacific?"
# query = "In what year did Tesla begin production of the Roadster?"

retriever = db.as_retriever(
    search_kwargs={"k":5}
)

relevant_docs = retriever.invoke(query)

print(f"user query: {query}")
print("Results")
for i, doc in enumerate(relevant_docs):
    print(f"Source: {doc.metadata['source']}")
    print(f"Document {i+1}:\n{doc.page_content}\n")