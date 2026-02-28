from collections import defaultdict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# retrieval (store) :
persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# need to research about this pydantic further
class QueryVariations(BaseModel):
    queries: List[str]

# retrieval (make) :
def multi_query(original_query):

    # store docs
    retriever = db.as_retriever(search_kwargs={"k": 5})  
    all_retrieval_results = []  

    # first, make query variations
    llm_with_tools = llm.with_structured_output(QueryVariations)

    prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
    Original query: {original_query}
    Return 3 alternative queries that rephrase or approach the same question from different angles."""

    response = llm_with_tools.invoke(prompt)
    query_variations = response.queries

    print("Generated Query Variations:")
    for i, variation in enumerate(query_variations, 1):
        print(f"{i}. {variation}")

        docs = retriever.invoke(variation)
        all_retrieval_results.append(docs)  # Store for RRF calculation

        print(f"Retrieved {len(docs)} documents:\n")
        
        for j, doc in enumerate(docs, 1):
            print(f"Document {j}:")
            print(f"{doc.page_content[:150]}...\n")

    return all_retrieval_results


def rrf(chunk_lists, k=60):
    
    rrf_scores = defaultdict(float)  # {chunk_content: rrf_score}
    all_unique_chunks = {}  # {chunk_content: actual_chunk_object}

    for chunk_list in chunk_lists:

        for position, chunk in enumerate(chunk_list, 1):
            chunk_content = chunk.page_content
            if chunk_content not in all_unique_chunks:
                all_unique_chunks[chunk_content] = chunk
                position_score = 1 / (k + position)
                rrf_scores[chunk_content] = position_score
            else:
                position_score = 1 / (k + position)
                rrf_scores[chunk_content] += position_score

    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )

    return sorted_chunks

def main():
    original_query = "What is the best way to learn Python?"
    all_retrieval_results = multi_query(original_query)
    sorted_chunks = rrf(all_retrieval_results)
    print(sorted_chunks)
