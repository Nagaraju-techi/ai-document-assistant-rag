# app.py
# AI Document Assistant using RAG

import streamlit as st
import os
import time
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

# -------------------------------
# SET PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("📄 AI Document Assistant using RAG")
st.write("Upload a PDF and ask questions from it.")

# -------------------------------
# INITIALIZE SESSION STATE
# -------------------------------
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = None
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0
if 'vector_store_created' not in st.session_state:
    st.session_state.vector_store_created = False

# -------------------------------
# API KEY INPUT
# -------------------------------
google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
@st.cache_data
def get_pdf_text(pdf_files):
    """Extract text from uploaded PDF files"""
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            continue
    return text

# -------------------------------
# TEXT SPLITTING
# -------------------------------
@st.cache_data
def get_chunks(text):
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# -------------------------------
# EMBEDDINGS AND VECTOR STORE
# -------------------------------

# Ordered list of model names to try — newest first, fallbacks after
EMBEDDING_MODELS = [
    "models/text-embedding-004",
    "models/embedding-001",
    "text-embedding-004",
    "embedding-001",
]

def get_embeddings(api_key):
    """
    Try each known embedding model name in order and return the first one
    that initialises without error.  The object is NOT cached with
    @st.cache_resource so we can pass different api_key values safely.
    """
    last_error = None
    for model_name in EMBEDDING_MODELS:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            # Do a tiny smoke-test so we catch 404/403 here rather than later
            embeddings.embed_query("test")
            # Cache the winning model name so we don't retry on every call
            st.session_state["embedding_model"] = model_name
            return embeddings
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"None of the embedding models worked. Last error: {last_error}\n"
        "Check that your API key is valid and has the Generative Language API enabled."
    )


def get_cached_embeddings(api_key):
    """
    Return the embeddings object, reusing the previously discovered model name
    if we already found a working one this session.
    """
    model_name = st.session_state.get("embedding_model")
    if model_name:
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )
    return get_embeddings(api_key)


def create_vector_store(text_chunks, api_key):
    """Create and save FAISS vector store"""
    embeddings = get_embeddings(api_key)  # auto-detects working model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# -------------------------------
# RAG RESPONSE WITH RETRY LOGIC
# -------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_llm_response(context, question, api_key):
    """Get response from LLM with retry logic"""
    prompt = PromptTemplate.from_template("""
You are a helpful AI assistant specialized in answering questions about documents.

Answer the user's question using ONLY the context provided below.
If the answer is not available in the context, say:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
""")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=1024
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# -------------------------------
# RATE LIMITING
# -------------------------------
def check_rate_limit():
    """Check and enforce rate limiting for API calls"""
    current_time = datetime.now()
    
    if st.session_state.last_request_time:
        time_diff = (current_time - st.session_state.last_request_time).total_seconds()
        
        if time_diff < 3:
            return False, f"⏳ Please wait {3 - int(time_diff)} seconds before asking another question."
        
        if time_diff > 60:
            st.session_state.request_count = 0
    
    if st.session_state.request_count >= 15:
        return False, "⚠️ Request limit reached. Please wait a minute before asking more questions."
    
    st.session_state.request_count += 1
    st.session_state.last_request_time = current_time
    return True, ""

# -------------------------------
# MAIN QUESTION ANSWERING
# -------------------------------
def user_input(question, api_key):
    """Process user question and return answer"""
    
    can_proceed, message = check_rate_limit()
    if not can_proceed:
        st.warning(message)
        return
    
    try:
        embeddings = get_cached_embeddings(api_key)
        
        if not os.path.exists("faiss_index"):
            st.error("❌ Vector store not found. Please process documents first.")
            return
        
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        with st.spinner("🔍 Searching document..."):
            docs = db.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        if not context.strip():
            st.warning("⚠️ No relevant information found in the document.")
            return
        
        with st.spinner("🤔 Generating answer..."):
            response = get_llm_response(context, question, api_key)
        
        st.subheader("🤖 Answer")
        st.write(response)
        
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:**")
                st.text_area(
                    "Relevant text",
                    value=doc.page_content[:500] + "...",
                    height=150,
                    key=f"source_{i}"
                )
                st.divider()
                
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            st.warning("⚠️ Rate limit reached. Please wait and try again.")
        elif "api_key" in error_msg.lower() or "API_KEY" in error_msg:
            st.error("❌ Invalid API key. Please check your Gemini API key.")
        elif "404" in error_msg or "NOT_FOUND" in error_msg:
            st.error(
                "❌ Model not found. Make sure the **Generative Language API** is enabled "
                "in your Google Cloud project and that your API key has access to it."
            )
        else:
            st.error(f"❌ Error: {error_msg}")

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("📁 Upload PDF Files")
    pdf_docs = st.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True,
        type=["pdf"],
        help="Upload one or more PDF files to analyze"
    )
    
    if st.button("🔨 Process Documents", type="primary"):
        if not google_api_key:
            st.error("❌ Please enter your Gemini API Key first.")
        elif not pdf_docs:
            st.warning("⚠️ Please upload at least one PDF file.")
        else:
            try:
                with st.spinner("📄 Extracting text..."):
                    raw_text = get_pdf_text(pdf_docs)
                    
                if not raw_text.strip():
                    st.error("❌ No text could be extracted. The PDF might be scanned or image-based.")
                else:
                    with st.spinner("🔨 Processing..."):
                        chunks = get_chunks(raw_text)
                        st.info(f"📊 Created {len(chunks)} text chunks")
                        create_vector_store(chunks, google_api_key)
                        st.session_state.vector_store_created = True

                    model_used = st.session_state.get("embedding_model", "unknown")
                    st.success(f"✅ Documents processed! (using `{model_used}`)")
                        
            except RuntimeError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
    
    if st.session_state.vector_store_created:
        st.sidebar.success("✅ Ready for questions")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Ask specific questions
    - Try "Summarize this document"
    - Ask about specific topics
    """)
    
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("API Key:", "Set" if google_api_key else "Not Set")
        st.sidebar.write("Vector Store:", "Exists" if os.path.exists("faiss_index") else "Not Found")
        st.sidebar.write("Embedding model:", st.session_state.get("embedding_model", "not detected yet"))
        st.sidebar.write("Python Version:", os.sys.version)

# -------------------------------
# MAIN INTERFACE
# -------------------------------
st.markdown("---")
question = st.text_input(
    "💭 Ask a question about your document:",
    placeholder="e.g., What is the main topic of this document?"
)

if question:
    if not google_api_key:
        st.error("❌ Please enter your Gemini API Key.")
    elif not st.session_state.vector_store_created:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        user_input(question, google_api_key)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and Google Gemini")
