from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("api/query/", views.query_rag_view, name="query_rag"),
    path("api/switch-backend/", views.switch_backend, name="switch_backend"),
    path("api/backend-info/", views.get_backend_info, name="backend_info"),
       path('api/upload-document/', views.upload_document_api, name='upload_document_api'),  # 
]