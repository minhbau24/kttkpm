from django.urls import path
from . import views
from . import admin_views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('store/', views.store_view, name='store'),
    path('products/<int:pk>/', views.product_detail_view, name='product_detail'),
    
    # Admin Dashboard URL patterns
    path('admin-dashboard/', admin_views.dashboard_view, name='admin_dashboard'),
    path('admin-dashboard/products/', admin_views.products_list_view, name='admin_products'),
    path('admin-dashboard/products/create/', admin_views.product_create_view, name='admin_product_create'),
    path('admin-dashboard/products/<int:pk>/edit/', admin_views.product_edit_view, name='admin_product_edit'),
    path('admin-dashboard/products/<int:pk>/delete/', admin_views.product_delete_view, name='admin_product_delete'),
    path('admin-dashboard/orders/', admin_views.orders_list_view, name='admin_orders'),
    path('admin-dashboard/shipments/', admin_views.shipments_list_view, name='admin_shipments'),

    # API Endpoints
    path('api/products/', views.product_list_api, name='api_product_list'),
    path('api/products/<int:pk>/', views.product_detail_api, name='api_product_detail'),
]
