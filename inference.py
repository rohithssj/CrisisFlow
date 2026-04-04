"""
inference.py — Execution script for the CrisisFlow simulation.
Uses the upgraded BaselineAgent to run a typical episode in the environment.
"""

import time
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.agents.baseline_agent import BaselineAgent

def run_simulation(difficulty="medium"):
    # 1. Initialize environment and agent
    print(f"--- INITIALIZING CRISISFLOW ({difficulty.upper()} MODE) ---")
    # env initialization updated to not pass a seed for dynamic sessions
    env = CrisisEnv(difficulty=difficulty)
    agent = BaselineAgent(env)
    
    # 2. Reset episode
    state = env.reset()
    done = False
    
    print("\nStarting simulation loop...")
    print(f"{'Step':<6} | {'Action':<25} | {'Reward':<8} | {'Active':<8} | {'Rescued':<8} | {'Dead':<8}")
    print("-" * 80)

    # 3. Main loop
    while not done:
        actions = agent.select_action(state)
        next_state, reward, done, info = env.step(actions)
        
        # Log progress
        step = info['steps_taken']
        
        # Calculate active patients from Observation
        active = sum(1 for p in next_state.patients if not p.get('rescued') and not p.get('dead'))
        rescued = info['rescued']
        dead = info['dead']
        
        # Simplified dispatch count logging
        if actions:
            action_str = f"Dispatched {len(actions)} Units"
        else:
            action_str = "Wait (No Action)"
        
        # Handle reward model or float
        reward_val = reward.score if hasattr(reward, 'score') else reward
        print(f"{step:<6} | {action_str:<25} | {reward_val:<8.3f} | {active:<8} | {rescued:<8} | {dead:<8}")
        
        state = next_state

    # 4. Final summary
    print("-" * 80)
    print("\n--- SIMULATION COMPLETE ---")
    
    # Updated metrics printout for professional summary
    print(f"Survival Rate     : {info['survival_rate']}%")
    print(f"Critical Rescued  : {info['critical_rescued']}")
    print(f"Overflow Rescues  : {info['overflow_rescues']}")
    print(f"Avg Response Time : {info['avg_response_time']} steps")
    print(f"Final Score       : {info['score']}")
    print("-" * 30)

if __name__ == "__main__":
    run_simulation(difficulty="medium")
