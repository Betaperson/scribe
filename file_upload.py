from openai import OpenAI
import os
import dotenv
import base64
import io
import fitz
from PIL import Image
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

@lru_cache(maxsize=1)
def get_openai_client():
    dotenv.load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

hash = "NaN"
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def imgToMDRouter(img):
    client = get_openai_client()

    completion = client.chat.completions.create(
        model="google/gemma-3-27b-it:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                            You are an AI that extracts text from an image of a PDF page and converts it into Markdown while preserving its structure. Pay close attention to lists, headings, and logical formatting. Do *not* introduce information not present in the original text.  Do *not* include code blocks.

Here are some examples of how to convert PDF text to Markdown, paying attention to structure.  Includes examples of both regular text *and* truth tables:

**Example 1 (Standard Text):**

**Input (PDF Text):**
Validity:
Rules out the possibility of the premises and conclusions not being true at once -
It is possible for the conclusion to be false but the argument valid-

**Output (Markdown):**
**Validity:**
Rules out the possibility of the premises and conclusions not being true at once -
It is possible for the conclusion to be false but the argument valid-

**Example 2 (List):**
**Input (PDF Text):**
Practice Questions:
A:
Valid1.Valid2.
Invalid3.Valid4.
Invalid5.Invalid6.

**Output (Markdown):**
Practice Questions:

A:
1. Valid
2. Valid
3. Invalid
4. Valid
5. Invalid
6. Invalid

**Example 3 (Truth Table):**

**Input (PDF Text):**
Column Index(i)3210
2iT & F8 T & F4 T & F2 T & F1 T & F

**Output (Markdown):**

| Column Index (i) | 3 | 2 | 1 | 0 |
|---|---|---|---|---|
| 2i | T | & | F | 8 |
| T | & | F | 4 | T |
| & | F | 2 | T | F |
| 1 | T | & | F |  |

**Example 4 (Combined Text & Logical Notation):**
**Input (PDF Text):**
A If A, then C 
∴C

**Output (Markdown):**
A If A, then C
∴C

Now, convert the following PDF into Markdown, adhering to the examples above. Pay special attention to accurately converting any truth tables that might be present.

                            """
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{img}"
                            }
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content

#can be reused for incorrect conversions
def pdf_to_base_64(pdf_path: str, page_number: int):
    pdf_doocument = fitz.open(pdf_path)
    page = pdf_doocument.load_page(page_number)
    image = page.get_pixmap()
    image = Image.frombytes("RGB", [image.width, image.height], image.samples)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def pdf_to_base64_batch(pdf_path: str, pages: list) -> list:
    pdf_document = fitz.open(pdf_path)
    results  = []

    try:
        for page in pages:
            curPage = pdf_document.load_page(page)
            image = curPage.get_pixmap()
            pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append((page, base64_img))
    finally:
        pdf_document.close()

    return results

async def pdfLoader(path: str) -> str:
    loader = PyPDFLoader(file_path=path)
    pages = []
    first_page = None
    async for page in loader.alazy_load():
        first_page = page
        break
    totalpages = first_page.metadata["total_pages"]
    all_images = pdf_to_base64_batch(path, list(range(totalpages)))

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(imgToMDRouter, img)
                  for _, img in all_images]
        results = [future.result() for future in futures]

    return "\n\n".join(results)

def createStore(totalMD, filehash):
    vector_store = Chroma(
        collection_name=f"{filehash}",
        embedding_function = embedding_model,
        persist_directory="./"
    )
    hash = filehash
    texts = text_splitter.split_text(totalMD)
    docs=[]
    for i in range(len(texts)):
        docs.append(Document(page_content=texts[i]))
    vector_store.add_documents(documents = docs)