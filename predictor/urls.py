from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_air_quality, name='predict_air_quality'),
]