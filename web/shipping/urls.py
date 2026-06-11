from django.urls import path
from . import views

urlpatterns = [
    path('shipping/update/<int:order_id>/', views.update_shipping_status, name='update_shipping_status'),
    path('api/shipping/create/', views.shipping_create_api, name='api_shipping_create'),
    path('api/shipping/status/', views.shipping_status_api, name='api_shipping_status'),
]
