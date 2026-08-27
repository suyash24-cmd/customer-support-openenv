from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

class Ticket(BaseModel):
    id: str
    customer_id: str
    issue_type: str
    description: str
    sentiment: float = Field(..., description="Customer sentiment from -1.0 (angry) to 1.0 (happy)")

class CustomerContext(BaseModel):
    is_premium: bool
    past_tickets: int
    total_spend: float
    months_subscribed: int

class Observation(BaseModel):
    ticket: Ticket
    customer_context: CustomerContext
    kb_search_results: Optional[str] = None
    conversation_history: List[str] = []
    is_escalated: bool = False
    is_closed: bool = False
    last_action_feedback: str = ""

class Action(BaseModel):
    action_type: Literal["Reply", "SearchKB", "IssueRefund", "Escalate", "CloseTicket"]
    content: Optional[str] = Field(None, description="Text for reply or KB search query")
    amount: Optional[float] = Field(None, description="Amount if issuing refund")

class State(BaseModel):
    ticket: Ticket
    customer_context: CustomerContext
    kb_articles: Dict[str, str]
    conversation_history: List[str]
    is_escalated: bool
    is_closed: bool
    refund_issued: float
    step_count: int
    max_steps: int
    task_id: str
    kb_queries_made: int = 0
    redundant_actions: int = 0
