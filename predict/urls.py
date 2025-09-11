from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_page, name="home_page"),
    path("upload_document/", views.upload_document, name="upload_document"),
    path("predict/", views.predict, name="predict")
]