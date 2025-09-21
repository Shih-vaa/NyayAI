# nyayai/core/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from .rag_service import query_rag, set_backend, get_current_backend, get_available_backends

def home(request):
    return render(request, "core/chat.html")

@csrf_exempt
@require_POST
def query_rag_view(request):
    """Handle AI queries with JSON payload"""
    try:
        # Support both form data and JSON
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            user_input = data.get("user_input", "").strip()
        else:
            user_input = request.POST.get("user_input", "").strip()
        
        if not user_input:
            return JsonResponse({"error": "Empty input"}, status=400)

        # Call the rag_service
        result = query_rag(user_input)
        return JsonResponse({"response": result})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_POST
def switch_backend(request):
    """Switch between different AI backends"""
    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            backend_name = data.get("backend", "").strip()
        else:
            backend_name = request.POST.get("backend", "").strip()
        
        if not backend_name:
            return JsonResponse({"error": "Backend parameter is required"}, status=400)

        result = set_backend(backend_name)
        return JsonResponse({"message": result})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def get_backend_info(request):
    """Get information about current and available backends"""
    try:
        current = get_current_backend()
        available = get_available_backends()
        
        return JsonResponse({
            "current_backend": current,
            "available_backends": available
        })
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)