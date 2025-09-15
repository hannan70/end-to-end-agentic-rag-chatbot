from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_page, name="home_page"),
    path("upload_document/", views.upload_document, name="upload_document"),
    path("change-llm/", views.change_llm, name="change_llm"),
    path("set-reasoning/", views.set_reasoning, name="set_reasoning"),
    path("predict/", views.predict, name="predict")
]