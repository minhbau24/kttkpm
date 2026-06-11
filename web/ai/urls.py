from django.urls import path
from . import views

urlpatterns = [
    # API Endpoints
    path('api/ai/predict-purchase/', views.predict_purchase_api, name='api_predict_purchase'),
    path('api/ai/recommendations/<int:user_id>/', views.recommendations_api, name='api_recommendations'),
    path('api/ai/similar-products/<int:product_id>/', views.similar_products_api, name='api_similar_products'),
    path('api/ai/chat/', views.chat_api, name='api_chat'),
    path('api/ai/chat/history/', views.chat_history_api, name='api_chat_history'),
    path('api/ai/profile/<int:user_id>/', views.user_profile_api, name='api_user_profile'),
    path('api/ai/model/status/', views.model_status_api, name='api_model_status'),
    
    # HTML Pages
    path('recommendations/', views.recommendations_view, name='recommendations'),
]
