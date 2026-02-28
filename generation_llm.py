import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from huggingface_hub import InferenceClient

load_dotenv()

def retrieval(dir, query): 

    db_dir = "db/chroma_db"
    db_dir = dir

    embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    query = query

    retriever = vector_db.as_retriever(
        search_kwargs={"k":5}
    )

    relevant_docs = retriever.invoke(query)

    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

    return relevant_docs

def generation(docs, query, model):

    INPUT = f"""
    Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."

    """

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=INPUT),
    ]

    result = model.text_generation(INPUT)
    
    print("\n -- Content Only -- \n")
    print(result.content)

def main():
    # client = InferenceClient(token=os.environ.get("HF_TOKEN"))
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    docs = retrieval(dir = "db/chroma_db", query = "Which island does SpaceX lease for its launches in the Pacific?")
    generation(docs, "Which island does SpaceX lease for its launches in the Pacific?", model)

if __name__ == "__main__":
    main()
