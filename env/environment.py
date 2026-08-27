from typing import Tuple, Dict, Any
from .models import Action, Observation, State
from .tasks import get_easy_task, get_medium_task, get_hard_task
from .reward import calculate_reward
from .graders import grade_easy_task, grade_medium_task, grade_hard_task
import copy

class CustomerSupportEnv:
    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level
        self._state: State = None
        
    def reset(self) -> Observation:
        if self.task_level == "easy":
            self._state = get_easy_task()
        elif self.task_level == "medium":
            self._state = get_medium_task()
        elif self.task_level == "hard":
            self._state = get_hard_task()
        else:
            raise ValueError(f"Unknown task level: {self.task_level}")
            
        return self._get_obs()
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step.")
            
        if self._state.is_closed or self._state.step_count >= self._state.max_steps:
            return self._get_obs(), 0.0, True, {"msg": "Episode already terminated."}
            
        prev_state = copy.deepcopy(self._state)
        self._state.step_count += 1
        
        last_feedback = ""
        
        if action.action_type == "Reply":
            if not action.content:
                self._state.redundant_actions += 1
                last_feedback = "Error: Reply content is missing."
            else:
                self._state.conversation_history.append(f"Agent: {action.content}")
                last_feedback = "Reply sent to customer."
                
        elif action.action_type == "SearchKB":
            if not action.content:
                self._state.redundant_actions += 1
                last_feedback = "Error: Search query is missing."
            else:
                self._state.kb_queries_made += 1
                query = action.content.lower()
                results = [v for k, v in self._state.kb_articles.items() if k in query.replace(" ", "_") or any(word in k for word in query.split())]
                if results:
                    last_feedback = f"KB Results: {str(results)}"
                else:
                    last_feedback = "KB Results: No matching articles found."
                    
        elif action.action_type == "IssueRefund":
            if not action.amount or action.amount <= 0:
                self._state.redundant_actions += 1
                last_feedback = "Error: Valid refund amount required."
            else:
                self._state.refund_issued += action.amount
                self._state.conversation_history.append(f"System: Issued refund of ${action.amount}.")
                last_feedback = f"Refund of ${action.amount} processed."
                
        elif action.action_type == "Escalate":
            self._state.is_escalated = True
            self._state.is_closed = True
            last_feedback = "Ticket escalated to tier 2 support. Ticket closed locally."
            
        elif action.action_type == "CloseTicket":
            self._state.is_closed = True
            last_feedback = "Ticket closed by agent."
            
        reward = calculate_reward(self._state, action, prev_state)
        done = self._state.is_closed or self._state.step_count >= self._state.max_steps
        
        # Add final grading bonus if done
        if done:
            final_score = 0.0
            if self.task_level == "easy":
                final_score = grade_easy_task(self._state)
            elif self.task_level == "medium":
                final_score = grade_medium_task(self._state)
            elif self.task_level == "hard":
                final_score = grade_hard_task(self._state)
                
            # Scale grade to a reward bonus (0 to 1 -> 0 to 10)
            reward += final_score * 10.0
            info = {"grade": final_score, "feedback": last_feedback}
        else:
            info = {"feedback": last_feedback}
            
        # Update obs feedback
        obs = self._get_obs()
        obs.last_action_feedback = last_feedback
        
        return obs, reward, done, info
        
    def _get_obs(self) -> Observation:
        return Observation(
            ticket=self._state.ticket,
            customer_context=self._state.customer_context,
            kb_search_results=None, # Passed via feedback mostly, or could persist
            conversation_history=self._state.conversation_history.copy(),
            is_escalated=self._state.is_escalated,
            is_closed=self._state.is_closed,
            last_action_feedback=""
        )
        
    def get_state(self) -> State:
        return self._state
