from openai import OpenAI
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import file_upload
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

_embedding_model = get_embedding_model()

def sendToLLM(results, query):
    prompt_template=f"""
        You are an assistant tasked with answering questions using the provided notes. Retrieve and synthesize relevant information to form a complete, accurate, and clear response. Focus only on the information found in the notes. If the notes do not contain enough information, state that explicitly.
            Notes: {results[0][0].page_content}
            Question: {query}
    """
    client = file_upload.get_openai_client()
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

def search(query, numResults, hash):
    vector_store = Chroma(
        collection_name=f"{hash[:7]}",
        embedding_function = _embedding_model,
        persist_directory="./"
    )   
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=_embedding_model.embed_query(query), k=numResults
    )
    return results

