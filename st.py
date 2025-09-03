import streamlit as st
from openai import OpenAI
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import search
import nest_asyncio
import asyncio
import hashlib
import file_upload
import json
import dotenv

nest_asyncio.apply()

@st.cache_data(show_spinner="Processing PDF...")

def process_pdf_cached(file_bytes):
    file_hash = hashlib.md5(file_bytes).hexdigest()[:7]
    with open(f"{file_hash}.pdf", "wb") as f:
        f.write(file_bytes)
    st.toast(f"File Saved as: {file_hash}.pdf")
    totalMD = asyncio.run(file_upload.pdfLoader(f"{file_hash}.pdf"))
    file_upload.createStore(totalMD, file_hash)
    return totalMD, file_hash

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#global hash
dotenv.load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

uploaded_file = st.file_uploader("Choose a file", type="pdf")
if uploaded_file is not None:
    bytesdata = uploaded_file.getvalue()
    st.toast("File Uploaded")
    totalMD, hash = process_pdf_cached(bytesdata)
    #with open(f"{hash[:7]}.pdf", "wb") as f:
    #    f.write(bytesdata)
    #totalMD = asyncio.run(file_upload.pdfLoader(f"{hash[:7]}.pdf"))
    #file_upload.createStore(totalMD, hash[:7])


    #try:
    #    with open("files.json", "r") as f:
    #        data = json.load(f)
    #except FileNotFoundError:
    #    data = {}

    #if hash not in data:
    #    data[hash] = {"processed": True}
    #    with open("files.json", "w") as f:
    #        json.dump(data, f)]


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type to get started."):
    results = search.search(prompt, 5, hash)
    prompt_template=f"""
        You are an assistant tasked with answering questions using the provided document. Retrieve and synthesize relevant information to form a complete, accurate, and clear response. Focus only on the information found in the notes. 
        If the notes do not contain enough information, state that explicitly.
        Use as many of the notes as much as possible to synthesize a coherent answer. Do not put the document ids when responding.
            Notes: {results}
            Question: {prompt}
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        api_messages = st.session_state.messages[:-1]
        api_messages.append({"role": "user", "content": prompt_template})
        stream = client.chat.completions.create(
            model = "meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=api_messages,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})