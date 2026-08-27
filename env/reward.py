from .models import State, Action

def calculate_reward(state: State, action: Action, previous_state: State) -> float:
    reward = 0.0
    
    # Penalize redundant actions
    if previous_state.redundant_actions < state.redundant_actions:
        reward -= 0.2
        
    # Small step penalty to encourage efficiency
    reward -= 0.05
    
    if action.action_type == "SearchKB":
        # Positive reward for doing research, but only a few times
        if state.kb_queries_made <= 2:
            reward += 0.1
        else:
            reward -= 0.1 # Spamming KB
            
    elif action.action_type == "IssueRefund":
        if state.task_id == "easy":
            reward -= 1.0 # Should not refund for password reset
        elif state.task_id == "medium" and action.amount == 150.0:
            reward += 0.5 # Correct refund amount for router
        elif state.task_id == "hard":
            if action.amount and action.amount > 0:
                reward += 0.5 # Progress on hard task
                
    elif action.action_type == "Escalate":
        if state.task_id == "easy":
            reward -= 1.0 # Unnecessary escalation
        elif state.task_id == "hard":
            reward += 0.2 # Valid escalation if complex, but refund is better
            
    elif action.action_type == "CloseTicket":
        pass # Final reward is handled by grader, maybe give small positive if closing

    return reward
