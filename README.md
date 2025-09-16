# 📚 PDF to Q&A Generator

## This project is an End-to-End Agentic RAG Chatbot System built with Django, LangChain, CrewAI, FAISS, and OpenAI/HuggingFace Embeddings.

### It allows users to upload documents (PDF/TXT), process them into embeddings, and query the system. The chatbot retrieves relevant information from the knowledge base or performs external web searches via Tavily when needed, and returns clear, synthesized answers.
---

## 🌟 Features
- **Document Upload**: Upload PDF or TXT files for knowledge base creation.
- **Text Extraction & Chunking**: Extracts text using PyPDFLoader / TextLoader and splits content with RecursiveCharacterTextSplitter.
- **Semantic Retrieval**: Embeds documents with HuggingFace Sentence-Transformers and stores them in FAISS for fast similarity search.
- **Agentic Workflow**: Uses CrewAI agents (Planner, Retriever, Web Searcher, Summarizer) to decide whether to answer from internal knowledge or perform external web searches.
- **External Knowledge Search**: Integrated with TavilySearchTool for up-to-date web results.
- **Conversational LLM**: Generates final user-facing answers via OpenAI Chat Models.
- **REST API + Web Interface**: Django backend provides endpoints and HTML frontend for interaction.
- **Customizable Reasoning Levels**: Choose between low, medium, high reasoning for answer depth and creativity

 
## 🛠️ Tech Stack
- **Backend**: Django (REST APIs & web views)  
- **Agents**: CrewAI (multi-agent reasoning & orchestration) 
- **AI/ML**: LangChain, HuggingFace Sentence-Transformers, OpenAI LLMs 
- **Vector Database**: FAISS (semantic search & retrieval)  
- **Search Tool**: Tavily Web Search API
- **Frontend**: Django templates (HTML/JS)
- **Environment**: Python 3.11+


## **Setup Guide (Windows Command Prompt)*

### **Step 1:** Clone the Repository
```bash
git clone https://github.com/hannan70/end-to-end-agentic-rag-chatbot

cd end-to-end-agentic-rag-chatbot
```

### **Step 1:** Create Virtual Environment
```bash
conda create -p venv python==3.11 --y (version: Python 3.11.13)
```

### **Step 2:** Activate Environment
```bash
conda activate venv\
```

### **Step 3:** Create .env File
```bash
Create a .env file in the project root and add:

TAVILY_API_KEY="******************"
OPENAI_API_KEY="******************"
```

### **Step 4:** Install Packages
```bash
pip install -r requirements.txt
```
 
### **Step 5:** Run Django Backend
```bash
python manage.py runserver
API will run at: http://127.0.0.1:8000/
```

## 📄 API Documentation
![Frontend View ](ui-image.png)


## 🏗️ System Architecture

### **1. Text Extraction**
- **Primary Tool**:  PyPDFLoader, TextLoader
- **Workflow**: PDF/TXT → Loader → Extracted Text
- **Strengths**: Handles both PDF and TXT files, ensuring clean extraction of content for downstream processing.

### **2. Chunking & Context Preservation**
- **Tool**: LangChain's RecursiveCharacterTextSplitter
- **Approach**: Splits text into overlapping chunks (chunk_size=500, chunk_overlap=100)
- **Benefits**: Preserves context while ensuring text is small enough for embeddings and retrieval.

### **3. Embedding & Retrieval**
- **Embedding Model**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Similarity Metric**: Cosine similarity for semantic matching
- **Advantages**: Lightweight, efficient embeddings with fast and scalable retrieval for RAG-based Q&A.

### **3. Agentic Decision Making**
- **Planner Agent**: Decides whether to query the internal knowledge base or perform an external web search.
- **Retriever Agent**: Searches FAISS for relevant document chunks.
- **External Agent**: Uses TavilySearchTool for live web information.
- **Summarizer Agent**: Synthesizes retrieved/internal knowledge into a polished final answer.

---

## 📝 License

This project is **open-source** and built with ❤️ using **CrewAI** **LangChain**, **OpenAI**, **FAISS**, and modern **NLP techniques**.  
Feel free to use, modify, and distribute under the terms of the MIT License.