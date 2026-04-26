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
if 'chunks_processed' not in st.session_state:
    st.session_state.chunks_processed = 0

# -------------------------------
# API KEY INPUT
# -------------------------------
google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# -------------------------------
# CACHED FUNCTIONS
# -------------------------------

@st.cache_data
def get_pdf_text(pdf_docs):
    """Extract text from PDF files - Cached for performance"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

@st.cache_data
def get_chunks(text):
    """Split text into chunks - Cached for performance"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

@st.cache_resource
def get_embeddings(api_key):
    """Get embeddings model - Cached as resource"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def create_vector_store(text_chunks, api_key):
    """Create and save FAISS vector store from text chunks"""
    embeddings = get_embeddings(api_key)
    
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )
    
    vector_store.save_local("faiss_index")
    return vector_store

@st.cache_data(ttl=3600)
def get_cached_response(_context, question, api_key):
    """Cache LLM responses based on context and question"""
    prompt = PromptTemplate.from_template("""
You are a helpful AI assistant specialized in answering questions about documents.

Answer the user's question using ONLY the context provided below.
If the answer is not available in the context, say:
"I couldn't find that information in the uploaded document."

Be concise and accurate in your response.

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
        max_output_tokens=1024,
        top_p=0.95,
        top_k=40
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# -------------------------------
# MAIN FUNCTIONALITY
# -------------------------------

def check_rate_limit():
    """Check and enforce rate limiting"""
    current_time = datetime.now()
    
    if st.session_state.last_request_time:
        time_diff = (current_time - st.session_state.last_request_time).total_seconds()
        
        # Enforce 3-second delay between requests
        if time_diff < 3:
            return False, f"⏳ Please wait {3 - int(time_diff)} seconds before asking another question."
        
        # Reset counter every 60 seconds
        if time_diff > 60:
            st.session_state.request_count = 0
    
    # Check request limit (15 requests per minute for free tier)
    if st.session_state.request_count >= 15:
        time_since_first = (current_time - st.session_state.last_request_time).total_seconds()
        if time_since_first < 60:
            return False, "⚠️ Request limit reached. Please wait a minute before asking more questions."
    
    st.session_state.request_count += 1
    st.session_state.last_request_time = current_time
    return True, ""

def user_input(question, api_key):
    """Retrieve answer using RAG with better error handling"""
    
    # Check rate limiting
    can_proceed, message = check_rate_limit()
    if not can_proceed:
        st.warning(message)
        return
    
    try:
        # Load embeddings
        embeddings = get_embeddings(api_key)
        
        # Load vector store
        if not os.path.exists("faiss_index"):
            st.error("❌ Vector store not found. Please process documents first.")
            return
        
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Search for relevant documents
        with st.spinner("🔍 Searching document for relevant information..."):
            docs = db.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
        
        # Check if we have meaningful context
        if not context.strip():
            st.warning("⚠️ No relevant information found in the document.")
            return
        
        # Get response from LLM with caching
        with st.spinner("🤔 Generating answer..."):
            try:
                response = get_cached_response(context, question, api_key)
                
                # Display answer
                st.subheader("🤖 Answer")
                st.write(response)
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Source {i+1}:**")
                        # Show relevance score if available
                        st.text_area(
                            f"Content from page (approx.)",
                            value=doc.page_content[:500] + "...",
                            height=150,
                            key=f"source_{i}"
                        )
                        st.divider()
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    st.warning("⚠️ API rate limit reached. Waiting 10 seconds before retrying...")
                    time.sleep(10)
                    try:
                        # Clear cache for this specific query to force new call
                        st.cache_data.clear()
                        response = get_cached_response(context, question, api_key)
                        st.subheader("🤖 Answer")
                        st.write(response)
                        
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.text_area(
                                    f"Content from page (approx.)",
                                    value=doc.page_content[:500] + "...",
                                    height=150,
                                    key=f"source_retry_{i}"
                                )
                                st.divider()
                    except Exception as retry_e:
                        st.error(f"❌ Still experiencing issues. Please try again in a minute. Error: {retry_e}")
                else:
                    raise e
                
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "invalid" in error_msg.lower():
            st.error("❌ Invalid API key. Please check your Gemini API key.")
        elif "not found" in error_msg.lower() or "404" in error_msg:
            st.error("❌ Resource not found. The vector store might be corrupted. Try processing the document again.")
        else:
            st.error(f"❌ An error occurred: {error_msg}")

# -------------------------------
# SIDEBAR FILE UPLOAD
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
                with st.spinner("📄 Extracting text from PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    
                if not raw_text.strip():
                    st.error("❌ No text could be extracted from the PDF. The file might be scanned or image-based.")
                else:
                    with st.spinner("🔨 Splitting text into chunks..."):
                        chunks = get_chunks(raw_text)
                        st.session_state.chunks_processed = len(chunks)
                        st.info(f"📊 Created {len(chunks)} text chunks from the document(s)")
                    
                    with st.spinner("🧠 Creating vector store..."):
                        create_vector_store(chunks, google_api_key)
                        st.session_state.vector_store_created = True
                        st.success(f"✅ Successfully processed {len(pdf_docs)} document(s)!")
                        
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    st.error("❌ API quota exceeded during processing. Please try again later.")
                elif "api_key" in error_msg.lower() or "invalid" in error_msg.lower():
                    st.error("❌ Invalid API key. Please check your Gemini API key.")
                else:
                    st.error(f"❌ Error processing documents: {error_msg}")
    
    # Show status in sidebar
    if st.session_state.vector_store_created:
        st.sidebar.success("✅ Documents are ready for questions")
        st.sidebar.info(f"📊 {st.session_state.chunks_processed} chunks indexed")
    
    # Add helpful tips
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("""
    - Ask specific questions about your document
    - Questions like "Summarize this document" work well
    - You can ask about specific topics in the document
    - The system will show you source paragraphs
    """)
    
    # Clear cache button
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.vector_store_created = False
        st.session_state.chunks_processed = 0
        st.success("Cache cleared successfully!")

# -------------------------------
# MAIN QUESTION INPUT
# -------------------------------
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input(
        "💭 Ask a question about your document:",
        placeholder="e.g., What is the main topic of this document?",
        key="question_input"
    )
with col2:
    if st.session_state.vector_store_created:
        st.metric("📄 Documents Ready", "✅")
    else:
        st.metric("📄 Documents Ready", "❌")

if question:
    if not google_api_key:
        st.error("❌ Please enter your Gemini API Key in the sidebar first.")
    elif not st.session_state.vector_store_created:
        st.warning("⚠️ Please upload and process a PDF first using the sidebar.")
    else:
        user_input(question, google_api_key)
        # Clear the input after processing
        # Comment this out if you want to keep the question visible
        # st.session_state.question_input = ""

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and Google Gemini")
