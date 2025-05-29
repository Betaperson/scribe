from openai import OpenAI
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import file_upload

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name=f"{file_upload.hash}",
    embedding_function = embedding_model,
    persist_directory="./"
)


def sendToLLM(results, query):
    prompt_template=f"""
        You are an assistant tasked with answering questions using the provided notes. Retrieve and synthesize relevant information to form a complete, accurate, and clear response. Focus only on the information found in the notes. If the notes do not contain enough information, state that explicitly.
            Notes: {results[0][0].page_content}
            Question: {query}
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
    #meta-llama/llama-4-scout-17b-16e-instruct
    completion = client.chat.completions.create(
        model = "meta-llama/llama-4-maverick-17b-128e-instruct",
        messages = [
            {
                "role":"user",
                "content":[
                    {
                        "type": "text",
                        "text": prompt_template
                    }
                ]
            }
        ]
    )

    return completion

def search(query, numResults):
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=embedding_model.embed_query(query), k=numResults
    )
    return results

