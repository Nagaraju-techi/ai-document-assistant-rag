# 📄 AI Document Assistant using RAG

An AI-powered document question answering system built using **Retrieval-Augmented Generation (RAG)**. Upload PDF files, retrieve relevant content using embeddings + vector search, and generate contextual answers using **Google Gemini LLM** through an interactive **Streamlit** interface.

---

## 🚀 Features

✅ Upload one or multiple PDF documents  
✅ Extract text from PDFs automatically  
✅ Smart text chunking for better retrieval  
✅ Semantic search using embeddings  
✅ FAISS vector database for fast similarity search  
✅ Context-aware answers using Gemini AI  
✅ Clean and interactive Streamlit UI  
✅ Beginner-friendly GenAI project

---

## 🧠 How It Works

1. User uploads PDF files  
2. Text is extracted from documents  
3. Content is split into smaller chunks  
4. Embeddings are created for each chunk  
5. FAISS stores vectors for semantic retrieval  
6. User asks a question  
7. Relevant chunks are retrieved  
8. Gemini LLM generates an accurate answer

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Language:** Python  
- **LLM:** Google Gemini API  
- **Framework:** LangChain  
- **Vector Database:** FAISS  
- **Embeddings:** HuggingFace Sentence Transformers  
- **PDF Parsing:** PyPDF2  

---

## 📁 Project Structure

```text
ai-document-assistant-rag/
│── app.py
│── requirements.txt
│── README.md
