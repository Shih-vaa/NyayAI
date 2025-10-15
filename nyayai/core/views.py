# nyayai/core/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from .rag_service import query_rag, set_backend, get_current_backend, get_available_backends

# ADD THESE IMPORTS
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import PyPDF2
import docx

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

# ADD THESE NEW FUNCTIONS

@csrf_exempt
def upload_document_api(request):
    """Handle document upload and analysis"""
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            uploaded_file = request.FILES['file']
            
            # Validate file size (5MB max)
            if uploaded_file.size > 5 * 1024 * 1024:
                return JsonResponse({'success': False, 'error': 'File size should be less than 5MB.'})
            
            # Save the file temporarily
            file_path = default_storage.save(f'temp_{uploaded_file.name}', ContentFile(uploaded_file.read()))
            full_path = default_storage.path(file_path)
            
            # Extract text based on file type
            text_content = ""
            file_extension = uploaded_file.name.lower()
            
            if file_extension.endswith('.pdf'):
                text_content = extract_text_from_pdf(full_path)
            elif file_extension.endswith(('.doc', '.docx')):
                text_content = extract_text_from_docx(full_path)
            elif file_extension.endswith('.txt'):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                except:
                    with open(full_path, 'r', encoding='latin-1') as f:
                        text_content = f.read()
            else:
                return JsonResponse({'success': False, 'error': 'Unsupported file format.'})
            
            # Clean up temporary file
            default_storage.delete(file_path)
            
            # Check if we successfully extracted text
            if not text_content or len(text_content.strip()) < 10:
                return JsonResponse({
                    'success': False, 
                    'error': 'Could not extract readable text from the document. The file might be corrupted, scanned, or password protected.'
                })
            
            # Generate summary using your RAG service
            summary_prompt = f"""Please analyze this legal document and provide a comprehensive legal analysis focusing on:

1. Document Type and Purpose
2. Key Legal Issues Identified
3. Applicable Indian Laws and Sections
4. Rights and Obligations of Parties
5. Potential Legal Consequences
6. Recommended Next Steps

Document Content:
{text_content[:4000]}

Please provide a structured legal analysis in simple language."""
            
            summary = query_rag(summary_prompt)
            
            return JsonResponse({
                'success': True,
                'filename': uploaded_file.name,
                'summary': summary,
                'document_type': classify_document_type(uploaded_file.name, text_content),
                'extracted_length': len(text_content)
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Processing error: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'No file uploaded'})

def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    try:
        # Try PyPDF2 first
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if text.strip():
                return text
            
            # If PyPDF2 fails, try pdfplumber for better extraction
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text if text.strip() else "PDF appears to be scanned or image-based. OCR required."
            except ImportError:
                return "PDF text extraction limited. Consider using a text-based PDF."
                
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text if text.strip() else "No readable text content found in the document."
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}. Please ensure the file is not corrupted."

def classify_document_type(filename, content):
    """Classify the type of legal document"""
    content_lower = content.lower()
    if any(term in content_lower for term in ['rent', 'lease', 'tenant', 'landlord']):
        return 'Rental Agreement'
    elif any(term in content_lower for term in ['sale', 'purchase', 'property', 'deed']):
        return 'Property Document'
    elif any(term in content_lower for term in ['employment', 'salary', 'job', 'termination']):
        return 'Employment Contract'
    elif any(term in content_lower for term in ['notice', 'legal notice', 'advocate']):
        return 'Legal Notice'
    elif any(term in content_lower for term in ['affidavit', 'sworn', 'declaration']):
        return 'Affidavit'
    elif any(term in content_lower for term in ['fir', 'police', 'complaint']):
        return 'FIR/Police Document'
    elif any(term in content_lower for term in ['agreement', 'contract']):
        return 'Contract'
    else:
        return 'Legal Document'