from django.urls import path, include
from .views import helloAPI, download

urlpatterns = [
    path("hello/", helloAPI),
    path("download/", download)
]