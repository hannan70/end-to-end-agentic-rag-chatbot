from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
from crewai import Agent, Task, Crew, Process
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from django.http import JsonResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from typing import Any
from crewai.tools import BaseTool
from crewai_tools import TavilySearchTool
from langchain_openai import ChatOpenAI

# load Tavily Search env
tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY") 
os.environ["LITELLM_PROVIDER"] = "openai" 

vector_store = None

def process_file(uploaded_files):
    global vector_store
    loaders = []
    all_docs = []
    for upload_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(upload_file.read())
            temp_path = temp_file.name

        if upload_file.name.endswith(".pdf"):
            loaders.append(PyPDFLoader(temp_path)) 
        elif upload_file.name.endswith(".txt"):
            loaders.append(TextLoader(temp_path))
        else:
            return JsonResponse({"message": "File upload problem"})

    for loader in loaders:
        loaded = loader.load() 
        if loaded:
            all_docs.extend(loaded)

    # splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_documents = text_splitter.split_documents(all_docs)

    # embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(final_documents, embedding=embeddings)
    
    return JsonResponse({"status":200, "message": "Files processed successfully"})
     

def llm_process(model_name, reasoning, user_question):
    global vector_store 
    
    if vector_store is None:
        return JsonResponse({"answer": "No documents uploaded yet!"})
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
 
    # LLM setup with reasoning
    if reasoning == "low":
        llm = ChatOpenAI(model=model_name, api_key=openai_api_key, temperature=0.2, max_tokens=512)
    elif reasoning == "high":
        llm = ChatOpenAI(model=model_name, api_key=openai_api_key, temperature=0.9, max_tokens=2048)
    else:  # medium
        llm = ChatOpenAI(model=model_name, api_key=openai_api_key, temperature=0.5, max_tokens=1024)


    # 🛠 Custom Retriever Tool
    class VectorStoreRetrieverTool(BaseTool):
        name: str = "Vector Store Retriever Tool"
        description: str = "Searches for information in the vector store."
        retriever: object

        def _run(self, query: str) -> str:
            if self.retriever:
                results = self.retriever.get_relevant_documents(query)
                return " ".join([doc.page_content for doc in results])
            return "No relevant information found in the vector store."
        

    # 🌐 Web Search Tool
    class TavilyWebSearchTool(BaseTool):
        name: str = "Tavily Search"
        description: str = "Search the web for up-to-date and accurate information."

        def _run(self, query: Any) -> str:
            tavily = TavilySearchTool(tavily_api_key=tavily_api_key)

            # CrewAI sometimes passes a dict like {"description": "...", "type": "str"}
            if isinstance(query, dict):
                query = query.get("description") or query.get("query") or str(query)

            # Call Tavily properly (not _run, but using run or invoke)
            return tavily.run(query)


    response_data = {"answer": "", "source": ""}

    # Agents
    planner = Agent(
        role="Planner",
        goal="Decide whether a query can be answered using the knowledge base or requires external search.",
        backstory="Planner that evaluates the query and routes it appropriately.",
        llm=llm
    )

    retriever_agent = Agent(
        role="Retriever",
        goal="Fetch the most relevant passages from the knowledge base.",
        backstory="Acts like a librarian who delivers information from documents.",
        tools=[VectorStoreRetrieverTool(retriever=retriever)],
        verbose=True,
        allow_delegation=False,
        llm=llm,
        on_complete=lambda output: response_data.update({"source": "vector_store", "answer": output})
    )

    external_agent = Agent(
        role="External Knowledge Seeker",
        goal="Retrieve accurate and up-to-date info from the web.",
        backstory="Expert researcher specialized in external sources.",
        tools=[TavilyWebSearchTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm,
        on_complete=lambda output: response_data.update({"source": "external_search", "answer": output})
    )

    summarizer = Agent(
        role="Answer Composer",
        goal="Synthesize all info into a clear final answer.",
        backstory="An expert communicator who gives polished responses.",
        llm=llm
    )

    # Tasks 
    task1 = Task(
        description="Analyze the query: {query}. Decide if it can be answered from the knowledge base or requires external search.",
        expected_output="Return either 'Internal knowledge is sufficient' or 'External search is necessary'.",
        agent=planner,
        context_variables=["query"]
    )

    task2 = Task(
        description="Using the query: {query}, search the knowledge base thoroughly and fetch the most relevant passages.",
        expected_output="A summary of relevant passages from the knowledge base.",
        agent=retriever_agent,
        context_variables=["query"]
    )

    task3 = Task(
        description="If internal knowledge is insufficient, perform a targeted external search for: {query}.",
        expected_output="A comprehensive answer from external sources.",
        agent=external_agent,
        context_variables=["query"]
    )

    task4 = Task(
        description="Synthesize all gathered info into a final answer for the query: {query}.",
        expected_output="A single, final answer that directly addresses the query.",
        agent=summarizer,
        context_variables=["query"]
    )

    # 🛠 Crew
    crew = Crew(
        agents=[planner, retriever_agent, external_agent, summarizer],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=True
    )

    # 🚀 Run
    response = crew.kickoff(inputs={"query": user_question})

    # Get final synthesized answer
    final_answer = response.tasks_output[-1].raw

    return final_answer