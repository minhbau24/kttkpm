from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view_action, name='logout'),
    path('api/auth/register/', views.register_api, name='api_register'),
    path('api/auth/login/', views.login_api, name='api_login'),
    path('api/users/', views.users_list_api, name='api_users_list'),
]
