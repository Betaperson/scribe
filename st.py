import streamlit as st
from openai import OpenAI
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#import file_upload
import search
import nest_asyncio
import asyncio

nest_asyncio.apply()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="test",
    embedding_function = embedding_model,
    persist_directory="/Users/prathamwankhede/Documents/scribe/"
)   

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type to get started."):
    results = search.search(prompt, 5)
    prompt_template=f"""
        You are an assistant tasked with answering questions using the provided notes. Retrieve and synthesize relevant information to form a complete, accurate, and clear response. Focus only on the information found in the notes. 
        If the notes do not contain enough information, state that explicitly.
        Use as many of the notes as much as possible to synthesize a coherent answer. Do not put the document ids when responding.
            Notes: {results}
            Question: {prompt}
    """
    st.session_state.messages.append({"role": "user", "content": prompt_template})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        stream = client.chat.completions.create(
            model = "meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})