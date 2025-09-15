from dotenv import load_dotenv
load_dotenv()
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from predict.helper import process_file, llm_process
import json 

# Create your views here.
def home_page(request):
    return render(request, "./pages/index.html")

def upload_document(request):
    if request.method == "POST":
        uploaded_files = request.FILES.getlist('documents')
        response = process_file(uploaded_files)
        return response
    
    return JsonResponse({"status": "Failed", "message": "Only Allow Post Method"})

def change_llm(request): 
    if request.method == "POST":
        model_name = request.POST.get("model_name")
    # set model name by session
    request.session['model_name'] = model_name
    print(model_name)
    return JsonResponse({"status": "Success", "message": model_name})


def set_reasoning(request):
    if request.method == "POST":
        body = json.loads(request.body)
        reasoning = body.get("reasoning", "medium")  # default medium
        print(reasoning)
        request.session['reasoning'] = reasoning
        return JsonResponse({"status": "Success", "reasoning": reasoning})
    return JsonResponse({"status": "Failed", "message": "Only Allow Post Method"})


def predict(request):
    default_model = "openai/gpt-3.5-turbo"
    model_name = request.session.get("model_name") or default_model
    reasoning = request.session.get("reasoning", "medium")
    
    if request.method == 'POST':
        data = json.loads(request.body)
        user_question = data.get("message")
        response = llm_process(model_name, reasoning, user_question)

        return JsonResponse({"status": "Success", "response": response})
    return JsonResponse({"status": "Failed", "message": "Only Allow Post Method"})


 

       
 