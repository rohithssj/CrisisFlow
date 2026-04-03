"""
inference.py — Execution script for the CrisisFlow simulation.
Uses the BaselineAgent to run a typical episode in the environment.
"""

import time
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.agents.baseline_agent import BaselineAgent

def run_simulation(difficulty="medium"):
    # 1. Initialize environment and agent
    print(f"--- INITIALIZING CRISISFLOW ({difficulty.upper()} MODE) ---")
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
        # Get action from baseline agent
        action = agent.select_action(state)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Log progress
        step = info['steps_taken']
        active = next_state['stats']['active_patients']
        rescued = info['rescued']
        dead = info['dead']
        
        action_str = f"Amb {action['ambulance_id']} -> Pat {action['patient_id']}" if action else "Wait (No Action)"
        
        print(f"{step:<6} | {action_str:<25} | {reward:<8.3f} | {active:<8} | {rescued:<8} | {dead:<8}")
        
        # Update current state
        state = next_state
        
        # Optional: Add small sleep to make output readable in real-time
        # time.sleep(0.01)

    # 4. Final summary
    print("-" * 80)
    print("\n--- SIMULATION COMPLETE ---")
    print(f"Final Score:       {info['score']:.4f}")
    print(f"Total Rescued:     {info['rescued']}")
    print(f"Total Dead:        {info['dead']}")
    print(f"Total Steps:       {info['steps_taken']}")
    print(f"Avg Response Time: {info['avg_response_time']} steps")
    print("-" * 30)

if __name__ == "__main__":
    run_simulation(difficulty="medium")
