# iceapp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("generate/", views.generate_image, name="generate_image"),
]
