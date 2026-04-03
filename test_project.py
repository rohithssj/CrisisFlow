from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.agents.baseline_agent import BaselineAgent

def test():
    env = CrisisEnv(difficulty="medium")
    agent = BaselineAgent(env)
    
    state = env.reset()
    print(f"Environment reset. Initial state difficulty: {state['difficulty']}")
    print(f"Initial stats: {state['stats']}")
    
    action = agent.select_action(state)
    print(f"Agent selected action for first patient: {action}")
    
    if action:
        next_state, reward, done, info = env.step(action)
        print(f"Execution successful. Reward: {reward}")
    else:
        print("No action found for initial state.")

if __name__ == "__main__":
    try:
        test()
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
