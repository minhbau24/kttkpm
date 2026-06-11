from django.urls import path
from . import views

urlpatterns = [
    path('payment/pay/<int:order_id>/', views.order_payment_page, name='order_payment_page'),
    path('api/payment/pay/', views.payment_pay_api, name='api_payment_pay'),
    path('api/payment/status/', views.payment_status_api, name='api_payment_status'),
]
