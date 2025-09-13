import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from django.http import JsonResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from predict.prompt import prompt



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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)

    # embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(final_documents, embedding=embeddings)
    
     


def llm_process(llm, user_question):
    
    if vector_store is None:
        return {"answer": "No documents uploaded yet!"}

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    response = rag_chain.invoke({"input": user_question})

    return response



    