from .models import State

def grade_easy_task(state: State) -> float:
    score = 0.0
    history = " ".join(state.conversation_history).lower()
    
    if "https://example.com/reset" in history:
        score += 0.8
    if state.is_closed:
        score += 0.2
    if state.refund_issued > 0:
        score -= 0.5
    if state.is_escalated:
        score -= 0.5
        
    return max(0.0, min(1.0, score))

def grade_medium_task(state: State) -> float:
    score = 0.0
    if state.refund_issued == 150.0:
        score += 0.6
    if state.kb_queries_made > 0:
        score += 0.2
    if state.is_closed:
        score += 0.2
    if state.is_escalated:
        score -= 0.3
        
    return max(0.0, min(1.0, score))

def grade_hard_task(state: State) -> float:
    score = 0.0
    history = " ".join(state.conversation_history).lower()
    
    # Needs to apologize
    if "apologize" in history or "sorry" in history:
        score += 0.2
        
    # Needs refund
    if state.refund_issued > 0:
        score += 0.5
        
    # Closure or successful escalation
    if state.is_closed:
        score += 0.3
        
    # Penalize if they just closed without refunding
    if state.is_closed and state.refund_issued == 0:
        score = 0.0
        
    return max(0.0, min(1.0, score))
