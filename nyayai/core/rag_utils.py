# nyayai/core/rag_utils.py

from dotenv import load_dotenv
import os

# Placeholder documents for testing - these will be returned as example responses
LEGAL_RESPONSES = {
    "security_deposit": "According to standard rental agreements, security deposits must be returned within 30 days of lease termination, minus any deductions for damages beyond normal wear and tear.",
    "workplace_harassment": "Workplace harassment is illegal under employment laws. You should document incidents and report to HR or relevant authorities. Legal remedies may include compensation and disciplinary action.",
    "tenant_complaints": "Tenants have the right to habitable living conditions. For maintenance issues, provide written notice to landlord and allow reasonable time for repairs before escalating.",
    "default": "I can help with legal information on various topics. Please provide more details about your specific situation for a more accurate response."
}

load_dotenv()

def get_rag_response(user_input):
    """
    Simulate RAG response using predefined legal templates.
    In a real implementation, this would query a vector database.
    """
    user_input_lower = user_input.lower()
    
    # Simple keyword matching for demo purposes
    if any(keyword in user_input_lower for keyword in ['security deposit', 'deposit', 'refund']):
        return LEGAL_RESPONSES["security_deposit"]
    
    elif any(keyword in user_input_lower for keyword in ['harassment', 'workplace', 'employment']):
        return LEGAL_RESPONSES["workplace_harassment"]
    
    elif any(keyword in user_input_lower for keyword in ['tenant', 'rent', 'maintenance', 'landlord']):
        return LEGAL_RESPONSES["tenant_complaints"]
    
    else:
        return LEGAL_RESPONSES["default"]

# Simple function to demonstrate RAG concept
def query_legal_knowledge(user_query):
    """
    This function simulates querying a legal knowledge base.
    In production, this would connect to your actual vector database.
    """
    return get_rag_response(user_query)