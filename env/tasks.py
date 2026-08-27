from .models import Ticket, CustomerContext, State

# Knowledge base
KB_ARTICLES = {
    "password_reset": "To reset a password, send the user the reset link: https://example.com/reset. Do not issue refunds for password issues.",
    "warranty_policy": "Defective electronics are covered within the first 12 months. Issue a full refund if reported within this period.",
    "billing_issues": "If a premium customer is double billed, immediately issue a refund for the duplicate amount and apologize. Do not escalate unless the amount is over $500."
}

def get_easy_task() -> State:
    return State(
        ticket=Ticket(
            id="TKT-001",
            customer_id="CUST-100",
            issue_type="Login Issue",
            description="I forgot my password and cannot log into my account.",
            sentiment=0.0
        ),
        customer_context=CustomerContext(
            is_premium=False,
            past_tickets=1,
            total_spend=50.0,
            months_subscribed=5
        ),
        kb_articles=KB_ARTICLES,
        conversation_history=["Customer: I forgot my password and cannot log into my account."],
        is_escalated=False,
        is_closed=False,
        refund_issued=0.0,
        step_count=0,
        max_steps=5,
        task_id="easy"
    )

def get_medium_task() -> State:
    return State(
        ticket=Ticket(
            id="TKT-002",
            customer_id="CUST-200",
            issue_type="Defective Product",
            description="The router I bought 2 months ago stopped working completely.",
            sentiment=-0.5
        ),
        customer_context=CustomerContext(
            is_premium=True,
            past_tickets=0,
            total_spend=150.0,
            months_subscribed=12
        ),
        kb_articles=KB_ARTICLES,
        conversation_history=["Customer: The router I bought 2 months ago stopped working completely."],
        is_escalated=False,
        is_closed=False,
        refund_issued=0.0,
        step_count=0,
        max_steps=8,
        task_id="medium"
    )

def get_hard_task() -> State:
    return State(
        ticket=Ticket(
            id="TKT-003",
            customer_id="CUST-300",
            issue_type="Billing Error",
            description="I was charged twice this month! I am a premium user and this is unacceptable. I want my money back or I am canceling.",
            sentiment=-0.9
        ),
        customer_context=CustomerContext(
            is_premium=True,
            past_tickets=3,
            total_spend=1200.0,
            months_subscribed=24
        ),
        kb_articles=KB_ARTICLES,
        conversation_history=["Customer: I was charged twice this month! I am a premium user and this is unacceptable. I want my money back or I am canceling."],
        is_escalated=False,
        is_closed=False,
        refund_issued=0.0,
        step_count=0,
        max_steps=10,
        task_id="hard"
    )
