from dotenv import load_dotenv
load_dotenv()
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from predict.helper import process_file, llm_process
from langchain_groq import ChatGroq
import json


groq_api_key = os.getenv("GROQ_API_KEY")
# setup llm
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key) 


# Create your views here.
def home_page(request):
    return render(request, "./pages/index.html")

def upload_document(request):
    if request.method == "POST":
        uploaded_files = request.FILES.getlist('documents')
        process_file(uploaded_files)
        return JsonResponse({"status": "Success", "message": "File uploaded Success"})
    
    return JsonResponse({"status": "Failed", "message": "Only Allow Post Method"})

def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_question = data.get("message")

        response = llm_process(llm, user_question)

        print(response)

        return JsonResponse({"status": "Success", "message": "File uploaded Success", "response": response['answer']})

    return JsonResponse({"status": "Failed", "message": "Only Allow Post Method"})
