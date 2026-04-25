# app.py
# AI Document Assistant using RAG

import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.base import Embeddings
from typing import List

# -------------------------------
# CUSTOM EMBEDDINGS USING google-genai
# -------------------------------
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from google import genai as genai_client
        client = genai_client.Client(api_key=self.api_key)
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# -------------------------------
# SET PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("📄 AI Document Assistant using RAG")
st.write("Upload a PDF and ask questions from it.")

# -------------------------------
# API KEY INPUT
# -------------------------------
google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# -------------------------------
# READ PDF TEXT
# -------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# -------------------------------
# SPLIT TEXT INTO CHUNKS
# -------------------------------
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# -------------------------------
# CREATE VECTOR STORE
# -------------------------------
def create_vector_store(text_chunks, api_key):
    embeddings = GeminiEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# -------------------------------
# ASK QUESTION
# -------------------------------
def user_input(question, api_key):
    embeddings = GeminiEmbeddings(api_key=api_key)
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I couldn't find that information in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})

    st.subheader("🤖 Answer")
    st.write(response)

# -------------------------------
# SIDEBAR FILE UPLOAD
# -------------------------------
with st.sidebar:
    st.header("Upload PDF Files")
    pdf_docs = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=["pdf"]
    )
    if st.button("Process Documents"):
        if not google_api_key:
            st.error("Please enter your Gemini API Key first.")
        elif not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_chunks(raw_text)
                create_vector_store(chunks, google_api_key)
                st.success("Documents Processed Successfully!")

# -------------------------------
# MAIN QUESTION INPUT
# -------------------------------
question = st.text_input("Ask a question from your PDF")
if question:
    if not google_api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not os.path.exists("faiss_index"):
        st.warning("Please upload and process a PDF first.")
    else:
        user_input(question, google_api_key)
