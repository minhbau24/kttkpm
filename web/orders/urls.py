from django.urls import path
from . import views

urlpatterns = [
    path('checkout/', views.order_checkout_view, name='checkout'),
    path('orders/', views.order_history_view, name='order_history'),
    path('orders/<int:pk>/', views.order_detail_view, name='order_detail'),
    path('api/orders/', views.order_create_api, name='api_order_create'),
    path('api/orders/list/', views.order_list_api, name='api_order_list'),
    path('api/orders/<int:pk>/', views.order_detail_api, name='api_order_detail'),
]
