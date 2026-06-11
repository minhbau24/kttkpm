from django.urls import path
from . import views

urlpatterns = [
    path('api/behavior/log/', views.log_event_api, name='api_log_event'),
]
