import pytest
from crisisflow.environment import CrisisEnv

def test_crisis_env_initialization():
    env = CrisisEnv(difficulty="easy")
    state = env.reset()
    assert isinstance(state, dict)
    assert "patients" in state
    assert "ambulances" in state
    assert "hospitals" in state
    assert "stats" in state

def test_crisis_env_step():
    env = CrisisEnv(difficulty="easy")
    state = env.reset()
    
    # Simple action: dispatch first ambulance to first patient if they exist
    if state["patients"] and state["ambulances"]:
        action = {"ambulance_id": 0, "patient_id": 0}
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(next_state, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "score" in info
