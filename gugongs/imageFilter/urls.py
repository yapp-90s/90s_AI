from django.conf.urls import url, include
from .views import apply

urlpatterns = [
    url("apply/", apply)
]