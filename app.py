# app.py
# AI Document Assistant using RAG

import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
    """Create and save FAISS vector store from text chunks"""
    # Use the CORRECT embedding model (gemini-embedding-001)
    # Note: This model outputs 3072 dimensions by default[citation:6][citation:9]
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # ✅ Correct model name
        google_api_key=api_key,
        task_type="retrieval_document"
    )
    
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )
    
    vector_store.save_local("faiss_index")
    return vector_store

# -------------------------------
# ASK QUESTION
# -------------------------------
def user_input(question, api_key):
    """Retrieve answer using RAG"""
    # Use the CORRECT embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # ✅ Correct model name
        google_api_key=api_key,
        task_type="retrieval_query"
    )
    
    # Load the vector store
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Search for relevant documents
    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt template
    prompt = PromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.

If the answer is not available in the context, say:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # You can also use "gemini-1.5-flash"
        temperature=0.3,
        google_api_key=api_key
    )
    
    # Create and run chain
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        st.subheader("🤖 Answer")
        st.write(response)
        
        # Show sources
        with st.expander("📚 Sources"):
            for i, doc in enumerate(docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:500] + "...")
                st.divider()
                
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            st.error("❌ API quota exceeded. Please try again later or use a different API key.")
        elif "api key" in error_msg.lower() or "invalid" in error_msg.lower():
            st.error("❌ Invalid or expired API key. Please check your Gemini API key.")
        elif "404" in error_msg or "not found" in error_msg.lower():
            st.error("❌ Model not found. Make sure you're using the correct API key and model names.")
        else:
            st.error(f"❌ Error: {error_msg}")

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
            st.error("❌ Please enter your Gemini API Key first.")
        elif not pdf_docs:
            st.warning("⚠️ Please upload at least one PDF.")
        else:
            try:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("❌ No text could be extracted from the PDF. Make sure it's not scanned or image-based.")
                    else:
                        chunks = get_chunks(raw_text)
                        st.info(f"✅ Extracted {len(chunks)} text chunks from PDF")
                        create_vector_store(chunks, google_api_key)
                        st.success("✅ Documents Processed Successfully!")
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower():
                    st.error("❌ API quota exceeded. Please try again later.")
                elif "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                    st.error("❌ Invalid API key. Please check your Gemini API key.")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    st.error("❌ Model not found. Make sure you're using the correct API key and model names.")
                else:
                    st.error(f"❌ Error processing documents: {error_msg}")

# -------------------------------
# MAIN QUESTION INPUT
# -------------------------------
question = st.text_input("Ask a question from your PDF")
if question:
    if not google_api_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar.")
    elif not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        user_input(question, google_api_key)
