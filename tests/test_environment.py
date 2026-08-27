from env.environment import CustomerSupportEnv
from env.models import Action


def test_reset_returns_observation_for_each_task_level():
    for level in ["easy", "medium", "hard"]:
        env = CustomerSupportEnv(task_level=level)
        obs = env.reset()
        assert obs.ticket is not None
        assert obs.is_closed is False


def test_reset_unknown_task_level_raises():
    env = CustomerSupportEnv(task_level="impossible")
    try:
        env.reset()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_step_before_reset_raises():
    env = CustomerSupportEnv(task_level="easy")
    try:
        env.step(Action(action_type="Reply", content="hi"))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_knowledge_base_search_returns_relevant_article():
    env = CustomerSupportEnv(task_level="medium")
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="SearchKB", content="warranty policy"))
    assert "warranty" in obs.last_action_feedback.lower() or "kb results" in obs.last_action_feedback.lower()
    assert done is False


def test_correct_refund_closes_and_grades_well():
    env = CustomerSupportEnv(task_level="medium")
    env.reset()
    env.step(Action(action_type="SearchKB", content="warranty"))
    env.step(Action(action_type="IssueRefund", amount=150.0))
    obs, reward, done, info = env.step(Action(action_type="CloseTicket"))
    assert done is True
    assert info["grade"] > 0.5
    assert env.get_state().refund_issued == 150.0


def test_escalation_sets_flags_and_closes():
    env = CustomerSupportEnv(task_level="hard")
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="Escalate"))
    assert done is True
    assert obs.is_escalated is True
    assert obs.is_closed is True


def test_invalid_refund_amount_is_penalized_as_redundant():
    env = CustomerSupportEnv(task_level="medium")
    env.reset()
    obs, reward, done, info = env.step(Action(action_type="IssueRefund", amount=0))
    assert "error" in obs.last_action_feedback.lower()
    assert env.get_state().redundant_actions == 1


def test_episode_terminates_at_max_steps():
    env = CustomerSupportEnv(task_level="easy")
    env.reset()
    max_steps = env.get_state().max_steps
    done = False
    for _ in range(max_steps):
        obs, reward, done, info = env.step(Action(action_type="SearchKB", content="x"))
    assert done is True
