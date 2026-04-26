# app.py
# AI Document Assistant using RAG

import streamlit as st
import os
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

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
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = None

# -------------------------------
# API KEY INPUT
# -------------------------------
google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# -------------------------------
# MODEL DISCOVERY (single source of truth)
# -------------------------------
def discover_models(api_key):
    """
    Calls ListModels once and returns two lists:
      - embedding_models : support embedContent
      - chat_models      : support generateContent  (prefer flash/pro, skip vision/legacy)
    """
    genai.configure(api_key=api_key)
    embedding_models = []
    chat_models = []

    try:
        for m in genai.list_models():
            methods = m.supported_generation_methods
            if "embedContent" in methods:
                embedding_models.append(m.name)
            if "generateContent" in methods:
                # Prefer gemini flash/pro, skip embedding-only or vision-only models
                name = m.name.lower()
                if "gemini" in name and "vision" not in name:
                    chat_models.append(m.name)
    except Exception as e:
        raise RuntimeError(f"Could not call ListModels: {e}")

    return embedding_models, chat_models


def get_embedding_model(api_key):
    """Return a working GoogleGenerativeAIEmbeddings instance."""
    if st.session_state.embedding_model:
        return GoogleGenerativeAIEmbeddings(
            model=st.session_state.embedding_model,
            google_api_key=api_key,
        )

    embedding_models, _ = discover_models(api_key)

    if not embedding_models:
        raise RuntimeError(
            "No embedding models available for your API key.\n"
            "Enable the Generative Language API at:\n"
            "https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com"
        )

    last_error = None
    for model_name in embedding_models:
        try:
            emb = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
            emb.embed_query("test")
            st.session_state.embedding_model = model_name
            return emb
        except Exception as e:
            last_error = e

    raise RuntimeError(f"No embedding model worked. Last error: {last_error}")


def get_chat_model_name(api_key):
    """Return the name of a working Gemini chat model."""
    if st.session_state.chat_model:
        return st.session_state.chat_model

    _, chat_models = discover_models(api_key)

    if not chat_models:
        raise RuntimeError(
            "No Gemini chat models available for your API key.\n"
            "Enable the Generative Language API in Google Cloud Console."
        )

    # Prefer models with 'flash' in name for speed, then 'pro', then whatever is available
    def priority(name):
        n = name.lower()
        if "flash" in n:
            return 0
        if "pro" in n and "vision" not in n:
            return 1
        return 2

    chat_models.sort(key=priority)

    last_error = None
    for model_name in chat_models:
        try:
            # Strip "models/" prefix if present — ChatGoogleGenerativeAI expects bare name
            bare_name = model_name.replace("models/", "")
            llm = ChatGoogleGenerativeAI(
                model=bare_name,
                temperature=0.1,
                google_api_key=api_key,
                max_output_tokens=64,
            )
            llm.invoke("hi")  # smoke test
            st.session_state.chat_model = bare_name
            return bare_name
        except Exception as e:
            last_error = e

    raise RuntimeError(f"No chat model worked. Last error: {last_error}")

# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------
@st.cache_data
def get_pdf_text(pdf_files):
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
    return text

# -------------------------------
# TEXT SPLITTING
# -------------------------------
@st.cache_data
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# -------------------------------
# VECTOR STORE
# -------------------------------
def create_vector_store(text_chunks, api_key):
    embeddings = get_embedding_model(api_key)
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
    model_name = get_chat_model_name(api_key)

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
        model=model_name,
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=1024,
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# -------------------------------
# RATE LIMITING
# -------------------------------
def check_rate_limit():
    current_time = datetime.now()
    if st.session_state.last_request_time:
        time_diff = (current_time - st.session_state.last_request_time).total_seconds()
        if time_diff < 3:
            return False, f"⏳ Please wait {3 - int(time_diff)} seconds before asking another question."
        if time_diff > 60:
            st.session_state.request_count = 0
    if st.session_state.request_count >= 15:
        return False, "⚠️ Request limit reached. Please wait a minute."
    st.session_state.request_count += 1
    st.session_state.last_request_time = current_time
    return True, ""

# -------------------------------
# MAIN QUESTION ANSWERING
# -------------------------------
def user_input(question, api_key):
    can_proceed, message = check_rate_limit()
    if not can_proceed:
        st.warning(message)
        return

    try:
        embeddings = get_embedding_model(api_key)

        if not os.path.exists("faiss_index"):
            st.error("❌ Vector store not found. Please process documents first.")
            return

        db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
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
                    key=f"source_{i}",
                )
                st.divider()

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("📁 Upload PDF Files")

    if google_api_key:
        if st.button("🔍 Check Available Models"):
            with st.spinner("Calling ListModels API..."):
                try:
                    emb_models, chat_models = discover_models(google_api_key)
                    st.success("✅ Embedding models:")
                    for m in emb_models:
                        st.code(m)
                    st.success("✅ Chat models:")
                    for m in chat_models:
                        st.code(m)
                except Exception as e:
                    st.error(f"❌ {e}")

    pdf_docs = st.file_uploader(
        "Choose PDF files",
        accept_multiple_files=True,
        type=["pdf"],
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
                    st.error("❌ No text could be extracted.")
                else:
                    with st.spinner("🔨 Processing..."):
                        chunks = get_chunks(raw_text)
                        st.info(f"📊 Created {len(chunks)} text chunks")
                        create_vector_store(chunks, google_api_key)
                        st.session_state.vector_store_created = True

                    emb = st.session_state.get("embedding_model", "unknown")
                    st.success(f"✅ Done! Embedding model: `{emb}`")

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
        st.sidebar.write("Chat model:", st.session_state.get("chat_model", "not detected yet"))
        st.sidebar.write("Python Version:", os.sys.version)

# -------------------------------
# MAIN INTERFACE
# -------------------------------
st.markdown("---")
question = st.text_input(
    "💭 Ask a question about your document:",
    placeholder="e.g., What is the main topic of this document?",
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
