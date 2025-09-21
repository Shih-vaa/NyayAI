# nyayai/core/rag_service.py
from enum import Enum
import os
import requests
from groq import Groq
import google.generativeai as genai
from collections import defaultdict
from datetime import datetime, timedelta

class LLMBackend(Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"

# Set GROQ as default backend
CURRENT_BACKEND = os.getenv("NYAYAI_BACKEND", LLMBackend.GROQ.value)

# Model configurations
MODEL_CONFIG = {
    LLMBackend.OLLAMA.value: {
        "model": "llama2",
        "api_url": "http://127.0.0.1:11434/api/generate",
        "requires_key": False
    },
    LLMBackend.GROQ.value: {
        "model": "llama-3.1-8b-instant",
        "requires_key": True
    },
    LLMBackend.GEMINI.value: {
        "model": "gemini-1.5-flash",
        "requires_key": True
    }
}

# Initialize clients only once (lazy loading)
_groq_client = None
_gemini_client = None

# Conversation memory storage
conversation_memory = defaultdict(list)
SESSION_TIMEOUT = timedelta(minutes=30)

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq backend")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini backend")
        
        genai.configure(api_key=api_key)
        model_name = MODEL_CONFIG[LLMBackend.GEMINI.value]["model"]
        _gemini_client = genai.GenerativeModel(model_name)
    
    return _gemini_client

def cleanup_old_sessions():
    """Remove old sessions to prevent memory leaks"""
    global conversation_memory
    if len(conversation_memory) > 100:
        # Keep only the 50 most recent sessions
        keys = list(conversation_memory.keys())[:50]
        conversation_memory = {k: conversation_memory[k] for k in keys}

def query_rag(user_query, top_k=1, session_id="default"):
    """Main function with conversation memory"""
    try:
        # Clean up old sessions
        cleanup_old_sessions()
        
        backend = CURRENT_BACKEND
        config = MODEL_CONFIG.get(backend, {})
        
        # Check if API key is required but not set
        if config.get("requires_key", True):
            api_key_env = {
                LLMBackend.GROQ.value: "GROQ_API_KEY",
                LLMBackend.GEMINI.value: "GEMINI_API_KEY"
            }.get(backend)
            
            if api_key_env and not os.getenv(api_key_env):
                return f"⚠️ Please set {api_key_env} environment variable for {backend}"
        
        # Get conversation history for this session
        conversation_history = conversation_memory[session_id]
        
        # Add new user message to history
        conversation_history.append({"role": "user", "content": user_query})
        
        # Prepare messages for API based on backend
        if backend == LLMBackend.GROQ.value:
            client = _get_groq_client()
            completion = client.chat.completions.create(
                model=config["model"],
                messages=conversation_history[-6:],  # Last 6 messages for context
                temperature=0.7,
                max_tokens=1024
            )
            response_text = completion.choices[0].message.content
            
        elif backend == LLMBackend.GEMINI.value:
            # Gemini needs different format
            model = _get_gemini_client()
            # Format conversation history for Gemini
            conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" 
                                            for msg in conversation_history[-4:]])
            full_query = f"Conversation context:\n{conversation_context}\n\nUser: {user_query}"
            response = model.generate_content(full_query)
            response_text = response.text
            
        elif backend == LLMBackend.OLLAMA.value:
            # Format for Ollama
            conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" 
                                            for msg in conversation_history[-4:]])
            full_query = f"{conversation_context}\nUser: {user_query}"
            response = requests.post(
                config["api_url"],
                json={"model": config["model"], "prompt": full_query, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "").strip()
        
        else:
            return "⚠️ No valid backend selected."
        
        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        # Keep only recent messages (avoid memory bloat)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        conversation_memory[session_id] = conversation_history
        
        return response_text
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {str(e)}"
    except Exception as e:
        return f"⚠️ Error with {backend} backend: {str(e)}"

# The rest of your utility functions remain the same
def set_backend(backend_name):
    """Dynamically change the backend at runtime"""
    global CURRENT_BACKEND
    valid_backends = [backend.value for backend in LLMBackend]
    
    if backend_name in valid_backends:
        CURRENT_BACKEND = backend_name
        return f"Backend switched to {backend_name}"
    else:
        return f"⚠️ Invalid backend. Available: {', '.join(valid_backends)}"

def get_backend_config(backend_name=None):
    """Get configuration for a specific backend or current backend"""
    if backend_name is None:
        backend_name = CURRENT_BACKEND
    return MODEL_CONFIG.get(backend_name, {})

def get_available_backends():
    """Return list of available backends with their models"""
    return {
        backend.value: {
            "model": MODEL_CONFIG[backend.value]["model"],
            "display_name": f"{backend.value.title()} ({MODEL_CONFIG[backend.value]['model']})",
            "requires_key": MODEL_CONFIG[backend.value]["requires_key"]
        }
        for backend in LLMBackend
    }

def get_current_backend():
    """Get the currently active backend with model info"""
    return {
        "backend": CURRENT_BACKEND,
        "model": MODEL_CONFIG[CURRENT_BACKEND]["model"],
        "display_name": f"{CURRENT_BACKEND.title()} ({MODEL_CONFIG[CURRENT_BACKEND]['model']})",
        "requires_key": MODEL_CONFIG[CURRENT_BACKEND]["requires_key"]
    }

def get_backend_status():
    """Check if current backend is properly configured"""
    backend = get_current_backend()
    
    if backend["requires_key"]:
        key_env = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY"
        }.get(backend["backend"])
        
        if key_env and not os.getenv(key_env):
            return {
                "status": "not_configured",
                "message": f"Please set {key_env} environment variable"
            }
    
    return {"status": "ready", "message": "Backend is configured"}