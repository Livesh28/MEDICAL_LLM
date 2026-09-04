#!/usr/bin/env python3
"""
Phase 18 Module: Intent Router Layer for VR Voice Assistant
Classifies incoming trainee queries into deterministic VR intents or RAG clinical QA intents.
Prevents sending deterministic VR queries blindly to LLMs.
"""

import re
from typing import Dict, Any, Optional

# Intent Classification Categories
INTENT_NEXT_STEP = "NEXT_STEP"
INTENT_REPEAT = "REPEAT"
INTENT_WHY_WRONG = "WHY_WRONG"
INTENT_HELP = "HELP"
INTENT_VR_CONTEXT = "VR_CONTEXT"
INTENT_CLINICAL_QA = "CLINICAL_QA"
INTENT_OPEN_QUESTION = "OPEN_QUESTION"
INTENT_UNSUPPORTED = "UNSUPPORTED"

def classify_intent(question: str) -> str:
    q_lower = question.lower().strip()
    
    # 1. Negative / Unsupported Questions Check
    unsupported_patterns = [
        r"blood pressure", r"medication", r"medical history", r"patient age", r"lab result",
        r"change procedure", r"skip step", r"capillary stick", r"height and weight", r"allergies",
        r"patient name", r"chart", r"prescribed", r"patient's", r"patient condition", r"how old"
    ]
    for pattern in unsupported_patterns:
        if re.search(pattern, q_lower):
            return INTENT_UNSUPPORTED
            
    # 2. Deterministic VR Simulation Query Patterns
    if any(k in q_lower for k in ["what should i do next", "what do i do next", "what is next", "next step", "what now", "where to start", "what's next"]):
        return INTENT_NEXT_STEP
    if any(k in q_lower for k in ["repeat", "say again", "pardon", "repeat that", "what did you say", "say step again"]):
        return INTENT_REPEAT
    if any(k in q_lower for k in ["why was that wrong", "why wrong", "mistake", "error", "why error", "why failed", "what went wrong", "why did that fail"]):
        return INTENT_WHY_WRONG
    if any(k in q_lower for k in ["help me", "i am stuck", "i'm stuck", "annotator", "guidance", "show me"]):
        return INTENT_HELP
    if any(k in q_lower for k in ["what object", "which object", "where should i put", "where do i place", "where does this go"]):
        return INTENT_VR_CONTEXT
        
    # 3. Open Clinical Questions vs Specific Clinical QA
    if any(k in q_lower for k in ["tell me about", "explain", "overview of", "describe"]):
        return INTENT_OPEN_QUESTION

    # 4. Default Clinical QA / RAG Query
    return INTENT_CLINICAL_QA

def format_deterministic_vr_response(
    intent: str,
    current_step: Optional[int],
    step_name: Optional[str],
    last_mistake: Optional[str],
    vr_steps_data: Optional[Dict[int, Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Formulates a deterministic VR response directly from StepManager state without LLM generation.
    """
    if current_step is None:
        current_step = 0
        
    step_info = vr_steps_data.get(current_step, {}) if vr_steps_data else {}
    s_name = step_name or step_info.get("name", f"Step {current_step}")
    s_obj = step_info.get("expected_object", "specified object")
    s_desc = step_info.get("vr_description", "Perform the expected interaction.")
    
    if intent == INTENT_NEXT_STEP:
        ans = f"Insert the {s_obj} into the Tube Slot." if current_step == 11 else f"Your next step is Step {current_step} ({s_name}): {s_desc}"
        return {"answer": ans, "engine": "vr_stepmanager_deterministic", "grounded": True, "confidence": "high", "sources": [], "intent": INTENT_NEXT_STEP}
        
    elif intent == INTENT_REPEAT:
        ans = f"Repeating Step {current_step} ({s_name}): {s_desc}"
        return {"answer": ans, "engine": "vr_stepmanager_deterministic", "grounded": True, "confidence": "high", "sources": [], "intent": INTENT_REPEAT}
        
    elif intent == INTENT_WHY_WRONG:
        mistake_text = last_mistake or "Invalid interaction sequence"
        ans = f"In Step {current_step} ({s_name}), the StepManager logged error: '{mistake_text}'. Ensure you interact with the {s_obj} correctly."
        return {"answer": ans, "engine": "vr_stepmanager_deterministic", "grounded": True, "confidence": "high", "sources": [], "intent": INTENT_WHY_WRONG}
        
    elif intent == INTENT_HELP:
        ans = f"Help for Step {current_step} ({s_name}): Look for the Annotator target highlighting the {s_obj}."
        return {"answer": ans, "engine": "vr_stepmanager_deterministic", "grounded": True, "confidence": "high", "sources": [], "intent": INTENT_HELP}
        
    elif intent == INTENT_VR_CONTEXT:
        ans = f"For Step {current_step} ({s_name}), you should use the '{s_obj}'."
        return {"answer": ans, "engine": "vr_stepmanager_deterministic", "grounded": True, "confidence": "high", "sources": [], "intent": INTENT_VR_CONTEXT}
        
    elif intent == INTENT_UNSUPPORTED:
        ans = "This information is not provided in the current simulation."
        return {"answer": ans, "engine": "vr_safety_guardrail", "grounded": True, "confidence": "high", "sources": [], "intent": INTENT_UNSUPPORTED}
        
    return None
