# nyayai/core/rag_service.py
from enum import Enum
import os
import requests
import json
from groq import Groq
import google.generativeai as genai
from collections import defaultdict
from datetime import timedelta

# Optional import for Redis (used if REDIS URL is provided)
try:
    import redis
except Exception:
    redis = None

class LLMBackend(Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"

# Set GROQ as default backend (string values are used across the file)
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

# In-memory conversation memory (fallback for dev/single-worker)
conversation_memory = defaultdict(list)
SESSION_TIMEOUT = timedelta(minutes=30)

# Redis configuration
REDIS_URL = os.getenv("NYAYAI_REDIS_URL") or os.getenv("REDIS_URL")
_use_redis = bool(REDIS_URL and redis is not None)
_redis_client = None
_MAX_MESSAGES = 10   # keep last 10 messages per session in store
_MAX_MESSAGES_FOR_SEND = 6  # how many messages to send to model for context

def _get_redis_client():
    global _redis_client
    if not _use_redis:
        return None
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

def _redis_key(session_id):
    return f"nyayai:conv:{session_id}"

def _load_conversation(session_id):
    """Load conversation list for session_id as list of {'role':.., 'content':..}"""
    if _use_redis:
        client = _get_redis_client()
        raw = client.lrange(_redis_key(session_id), 0, -1)  # left->right
        # stored as JSON strings
        conv = [json.loads(x) for x in raw] if raw else []
        return conv
    else:
        return list(conversation_memory.get(session_id, []))

def _append_message(session_id, role, content):
    """Append a message to session store (Redis or in-memory). Keeps length bounded."""
    msg = {"role": role, "content": content}
    if _use_redis:
        client = _get_redis_client()
        client.rpush(_redis_key(session_id), json.dumps(msg))
        # Keep only last _MAX_MESSAGES elements
        client.ltrim(_redis_key(session_id), -_MAX_MESSAGES, -1)
    else:
        conversation_memory[session_id].append(msg)
        if len(conversation_memory[session_id]) > _MAX_MESSAGES:
            conversation_memory[session_id] = conversation_memory[session_id][-_MAX_MESSAGES:]

def cleanup_old_sessions():
    """Only used for in-memory fallback. Fixes previous bug (kept defaultdict)."""
    global conversation_memory
    if _use_redis:
        return
    if len(conversation_memory) > 100:
        keys = list(conversation_memory.keys())[:50]
        conversation_memory = defaultdict(list, {k: conversation_memory[k] for k in keys})

# ---------------------------
# LLM clients (lazy)
# ---------------------------
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
        # Note: depending on google.generativeai version you might need different usage;
        # this keeps your existing usage style.
        _gemini_client = genai.GenerativeModel(model_name)
    return _gemini_client

# ---------------------------
# Main query function (uses session persistence)
# ---------------------------
def query_rag(user_query, top_k=1, session_id="default"):
    """
    Main function with conversation memory persisted in Redis (recommended) or in-memory (fallback).
    session_id must be stable across chat turns (client should send same session_id cookie/UUID).
    """
    try:
        # Clean up old sessions (only for in-memory)
        cleanup_old_sessions()

        backend = CURRENT_BACKEND
        config = MODEL_CONFIG.get(backend, {})

        # If API key required but missing -> helpful message
        if config.get("requires_key", True):
            api_key_env = {
                LLMBackend.GROQ.value: "GROQ_API_KEY",
                LLMBackend.GEMINI.value: "GEMINI_API_KEY"
            }.get(backend)
            if api_key_env and not os.getenv(api_key_env):
                return f"⚠️ Please set {api_key_env} environment variable for {backend}"

        # Load current conversation from store
        conversation_history = _load_conversation(session_id)

        # Append the new user message (persisted)
        _append_message(session_id, "user", user_query)
        # reload local history from store (so we include the persisted message)
        conversation_history = _load_conversation(session_id)

        # Build context slice to send to model
        messages_to_send = conversation_history[-_MAX_MESSAGES_FOR_SEND:]

        # Prepare messages for API based on backend (keep your original behaviour)
        if backend == LLMBackend.GROQ.value:
            client = _get_groq_client()
            # Groq client expects messages as list of dicts with role/content
            completion = client.chat.completions.create(
                model=config["model"],
                messages=messages_to_send,
                temperature=0.7,
                max_tokens=1024
            )
            response_text = completion.choices[0].message.content

        elif backend == LLMBackend.GEMINI.value:
            model = _get_gemini_client()
            # Convert messages to a readable conversation context for Gemini (string)
            conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
            full_query = f"Conversation context:\n{conversation_context}\n\nAssistant, answer the user's last message in plain language.\nUser: {user_query}"
            response = model.generate_content(full_query)
            # adapt to expected attribute names of the genai response object
            response_text = getattr(response, "text", None) or str(response)

        elif backend == LLMBackend.OLLAMA.value:
            conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
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

        # Persist assistant response
        _append_message(session_id, "assistant", response_text)

        # Return the answer text
        return response_text

    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {str(e)}"
    except Exception as e:
        return f"⚠️ Error with {backend} backend: {str(e)}"

# ---------------------------
# Utility functions you already had, unchanged (but updated to work with Redis)
# ---------------------------
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

    if _use_redis and not _get_redis_client():
        return {"status": "not_ready", "message": "Redis configured but client not available"}

    return {"status": "ready", "message": "Backend is configured"}
