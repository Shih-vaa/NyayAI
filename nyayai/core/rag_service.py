# nyayai/core/rag_service.py
from enum import Enum
import os
import requests
import json
from groq import Groq
import google.generativeai as genai
from collections import defaultdict
from datetime import timedelta

# ---------------------------
# Enhanced Legal System Prompt (More Natural)
# ---------------------------
LEGAL_SYSTEM_PROMPT = """
You are NyayAI, a friendly and knowledgeable legal assistant specialized in Indian law. 

**Your Role:**
- Primary focus: Indian laws, rights, legal procedures, and general legal guidance
- Secondary: General conversation while gently steering towards legal topics
- Tone: Friendly, conversational, empathetic, and professional

**Guidelines:**
1. **Legal Expertise**: Provide accurate information about Indian laws (IPC, CrPC, Constitution, Acts)
2. **Conversational Flow**: Maintain natural conversation, remember context, build on previous messages
3. **Legal Steering**: When non-legal topics come up, acknowledge them briefly and gently guide back to legal assistance
4. **Practical Help**: Offer step-by-step guidance, document checklists, and next steps
5. **Clarity**: Explain legal concepts in simple language anyone can understand
6. **Disclaimer**: Include a gentle disclaimer when providing specific legal information

**Response Style:**
- Be conversational and engaging
- Ask follow-up questions to better understand the situation
- Provide practical, actionable advice
- Use examples and analogies where helpful
- Show empathy for the user's situation

Remember: You're here to help people navigate legal challenges with confidence.
"""

# ---------------------------
# Enhanced Legal Keywords (More Inclusive)
# ---------------------------
LEGAL_KEYWORDS = [
    # Core legal terms
    "law", "legal", "rights", "case", "court", "judge", "lawyer", "advocate",
    "ipc", "crpc", "cpc", "constitution", "act", "section", "article",
    "fir", "complaint", "suit", "petition", "appeal", "bail", "arrest",
    
    # Legal domains
    "property", "rent", "tenant", "landlord", "eviction", "ownership",
    "contract", "agreement", "deal", "terms", "violation", "breach",
    "employment", "job", "salary", "termination", "harassment", "discrimination",
    "consumer", "product", "service", "refund", "warranty", "defective",
    "family", "marriage", "divorce", "maintenance", "custody", "inheritance",
    "criminal", "theft", "fraud", "assault", "threat", "cheating", "harassment",
    
    # Legal procedures
    "file", "register", "complaint", "notice", "summons", "hearing", "trial",
    "evidence", "witness", "document", "affidavit", "testimony",
    "police", "station", "investigation", "charge sheet",
    
    # Common legal issues
    "dispute", "problem", "issue", "conflict", "argument", "fight",
    "money", "payment", "due", "owed", "recover", "claim",
    "injury", "accident", "medical", "compensation", "damages",
    
    # Help-seeking terms
    "help with", "what to do", "how to", "procedure", "process", "steps",
    "can i", "should i", "rights for", "legal option"
]

def is_legal_query(query: str) -> bool:
    """More lenient legal detection - looks for legal intent"""
    if not query or len(query.strip()) < 2:
        return False
        
    query_lower = query.lower().strip()
    
    # Direct legal indicators
    if any(keyword in query_lower for keyword in LEGAL_KEYWORDS):
        return True
    
    # Question patterns that suggest legal need
    question_indicators = [
        "what are my rights", "how to file", "can i sue", "legal action",
        "what should i do", "is it illegal", "against the law", "legal help",
        "court procedure", "police complaint", "law against"
    ]
    
    if any(indicator in query_lower for indicator in question_indicators):
        return True
    
    return False

def should_redirect_to_legal(query: str) -> bool:
    """Check if this is clearly non-legal and should be redirected"""
    query_lower = query.lower().strip()
    
    # Clearly non-legal topics
    non_legal_topics = [
        "sports", "cricket", "football", "movie", "music", "entertainment",
        "cooking", "recipe", "food", "restaurant", "travel", "vacation",
        "weather", "joke", "funny", "game", "gaming", "celebrity"
    ]
    
    return any(topic in query_lower for topic in non_legal_topics)

# ---------------------------
# Enhanced Casual Message Handling
# ---------------------------
GREETINGS = ["hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening"]
CASUAL_RESPONSES = ["ok", "thanks", "thank you", "bye", "see you", "goodbye", "cool", "great"]

def handle_casual_message(query):
    """More natural casual conversation handling"""
    query_lower = query.lower().strip()
    
    # Greetings
    if any(greeting in query_lower for greeting in GREETINGS):
        return "Hello! I'm NyayAI, your friendly legal assistant. I'm here to help you with any legal questions about Indian laws, your rights, or legal procedures. What's on your mind today?"
    
    # Gratitude
    if any(thanks in query_lower for thanks in ["thanks", "thank you"]):
        return "You're welcome! I'm glad I could help. If you have any other legal questions, feel free to ask. Remember, I'm here to help you understand your legal options and rights."
    
    # Farewells
    if any(bye in query_lower for bye in ["bye", "goodbye", "see you"]):
        return "Goodbye! Feel free to come back if you have any legal questions. Remember, understanding your rights is the first step toward justice. Stay informed! 👋"
    
    # General acknowledgments
    if query_lower in ["ok", "okay", "got it", "understood"]:
        return "Great! Is there anything else you'd like to know about legal matters? I'm here to help with any questions about laws, rights, or legal procedures."
    
    return None

# ---------------------------
# Enhanced Conversation Flow Management
# ---------------------------
def enhance_legal_response(response_text, user_query, conversation_context):
    """Add conversational elements to legal responses"""
    
    # Add follow-up questions for better engagement
    follow_up_questions = [
        "Would you like me to explain any part in more detail?",
        "Do you have any specific situation you'd like to discuss?",
        "Would you like me to help you with the next steps?",
        "Is there anything else about this legal topic you're curious about?",
        "Would a step-by-step guide be helpful for your situation?"
    ]
    
    import random
    follow_up = random.choice(follow_up_questions)
    
    # Add gentle disclaimer (not too repetitive)
    disclaimer = " ⚠️ Remember, this is general legal information. For specific legal advice, please consult a qualified lawyer."
    
    # Only add follow-up if the response is substantial
    if len(response_text) > 100 and "?" not in response_text:
        response_text += f"\n\n{follow_up}"
    
    # Add disclaimer occasionally, not every time
    if random.random() < 0.3:  # 30% chance
        response_text += disclaimer
    
    return response_text

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
_MAX_MESSAGES = 15   # Increased for better context
_MAX_MESSAGES_FOR_SEND = 8  # Increased for better conversation flow

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
        raw = client.lrange(_redis_key(session_id), 0, -1)
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
        client.ltrim(_redis_key(session_id), -_MAX_MESSAGES, -1)
    else:
        conversation_memory[session_id].append(msg)
        if len(conversation_memory[session_id]) > _MAX_MESSAGES:
            conversation_memory[session_id] = conversation_memory[session_id][-_MAX_MESSAGES:]

def cleanup_old_sessions():
    """Only used for in-memory fallback."""
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
        _gemini_client = genai.GenerativeModel(model_name)
    return _gemini_client

# ---------------------------
# Enhanced Main Query Function
# ---------------------------
def query_rag(user_query, top_k=1, session_id="default"):
    """
    Enhanced natural conversation with legal expertise
    """
    try:
        cleanup_old_sessions()

        # ----------------
        # 1. Handle casual messages naturally
        casual_response = handle_casual_message(user_query)
        if casual_response:
            _append_message(session_id, "assistant", casual_response)
            return casual_response

        backend = CURRENT_BACKEND
        config = MODEL_CONFIG.get(backend, {})

        # ----------------
        # 2. Load conversation context
        conversation_history = _load_conversation(session_id)
        _append_message(session_id, "user", user_query)
        conversation_history = _load_conversation(session_id)
        
        # Use more context for better conversation flow
        messages_to_send = conversation_history[-_MAX_MESSAGES_FOR_SEND:]

        # Prepend system prompt
        messages_with_system = [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}] + messages_to_send

        # ----------------
        # 3. Backend-specific call with enhanced parameters
        response_text = ""
        if backend == LLMBackend.GROQ.value:
            client = _get_groq_client()
            completion = client.chat.completions.create(
                model=config["model"],
                messages=messages_with_system,
                temperature=0.8,  # Slightly higher for more natural responses
                max_tokens=1024,
                top_p=0.9
            )
            response_text = completion.choices[0].message.content

        elif backend == LLMBackend.GEMINI.value:
            model = _get_gemini_client()
            # Build conversation context for Gemini
            conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in messages_with_system])
            full_prompt = f"{conversation_context}\nUser: {user_query}\nAssistant:"
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=1024,
                    top_p=0.9
                )
            )
            response_text = getattr(response, "text", None) or str(response)

        elif backend == LLMBackend.OLLAMA.value:
            conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in messages_with_system])
            full_prompt = f"{conversation_context}\nUser: {user_query}\nAssistant:"
            
            response = requests.post(
                config["api_url"],
                json={
                    "model": config["model"], 
                    "prompt": full_prompt, 
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "").strip()

        else:
            response_text = "⚠️ No valid backend selected."

        # ----------------
        # 4. Enhance response for better conversation flow
        if response_text and not response_text.startswith("⚠️"):
            response_text = enhance_legal_response(response_text, user_query, conversation_history)

        # ----------------
        # 5. Handle clearly non-legal topics gently
        if should_redirect_to_legal(user_query) and not is_legal_query(user_query):
            redirect_response = f"I'd be happy to help with that, but I'm specially trained to assist with legal matters to provide the most accurate help. If you have any questions about laws, rights, or legal procedures in India, I'm here to help with those! What legal topic can I assist you with today?"
            _append_message(session_id, "assistant", redirect_response)
            return redirect_response

        # ----------------
        # 6. Persist assistant response
        _append_message(session_id, "assistant", response_text)
        return response_text

    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {str(e)}"
    except Exception as e:
        return f"⚠️ Error with {backend} backend: {str(e)}"

# ---------------------------
# Legal Analysis Functions (Enhanced)
# ---------------------------
def analyze_legal_strength(user_query, case_details):
    """Analyze the legal strength of a case"""
    strength_analysis_prompt = f"""
    Based on this query: "{user_query}"
    And these case details: {case_details}
    
    Provide a friendly but professional analysis:
    
    **Legal Strength Assessment:**
    - Basis in Law: [Explain which laws apply]
    - Evidence Needs: [What would strengthen the case]
    - Likely Challenges: [Potential obstacles]
    - Next Steps: [Practical recommendations]
    
    Keep it conversational but informative.
    """
    
    return query_rag(strength_analysis_prompt)

def generate_legal_checklist(case_type):
    """Generate case-specific legal checklist"""
    checklist_prompt = f"""
    Create a helpful checklist for someone dealing with a {case_type} case in India.
    
    Make it practical and easy to follow:
    - Documents they'll need
    - Steps to take
    - People to contact
    - Timeline expectations
    - Costs to consider
    
    Format it in a friendly, reassuring way.
    """
    
    return query_rag(checklist_prompt)

def estimate_case_timeline(case_type, complexity):
    """Provide realistic case timeline estimates"""
    timeline_prompt = f"""
    Give a realistic timeline estimate for a {case_type} case in India.
    Complexity level: {complexity}
    
    Break it down into phases:
    - Initial steps (what they can do now)
    - Filing and early stages
    - Court proceedings
    - Possible outcomes
    
    Be honest about timeframes but also encouraging.
    """
    
    return query_rag(timeline_prompt)

# ---------------------------
# Utility functions
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
    if backend_name is None:
        backend_name = CURRENT_BACKEND
    return MODEL_CONFIG.get(backend_name, {})

def get_available_backends():
    return {
        backend.value: {
            "model": MODEL_CONFIG[backend.value]["model"],
            "display_name": f"{backend.value.title()} ({MODEL_CONFIG[backend.value]['model']})",
            "requires_key": MODEL_CONFIG[backend.value]["requires_key"]
        }
        for backend in LLMBackend
    }

def get_current_backend():
    return {
        "backend": CURRENT_BACKEND,
        "model": MODEL_CONFIG[CURRENT_BACKEND]["model"],
        "display_name": f"{CURRENT_BACKEND.title()} ({MODEL_CONFIG[CURRENT_BACKEND]['model']})",
        "requires_key": MODEL_CONFIG[CURRENT_BACKEND]["requires_key"]
    }

def get_backend_status():
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