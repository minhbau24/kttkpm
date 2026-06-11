from django.urls import path
from . import views

urlpatterns = [
    path('cart/', views.cart_view, name='cart'),
    path('cart/add/<int:product_id>/', views.cart_add_view, name='cart_add'),
    path('cart/remove/<int:product_id>/', views.cart_remove_view, name='cart_remove'),
    path('api/cart/', views.cart_detail_api, name='api_cart_detail'),
    path('api/cart/add/', views.cart_add_api, name='api_cart_add'),
    path('api/cart/remove/', views.cart_remove_api, name='api_cart_remove'),
]
