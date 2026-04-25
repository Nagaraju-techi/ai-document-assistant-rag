# app.py
# AI Document Assistant using RAG
# Beginner Friendly Cognizant Ready Project

import streamlit as st
import os

from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.question_answering import load_qa_chain

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
            text += page.extract_text()
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
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# -------------------------------
# ASK QUESTION
# -------------------------------
def user_input(question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=question)

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
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_chunks(raw_text)
            create_vector_store(chunks)
            st.success("Documents Processed Successfully!")

# -------------------------------
# MAIN QUESTION INPUT
# -------------------------------
question = st.text_input("Ask a question from your PDF")

if question:
    user_input(question)
