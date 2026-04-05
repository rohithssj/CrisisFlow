import os
import yaml
import math
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.environment.models import Action
from crisisflow.agents.baseline_agent import BaselineAgent
from crisisflow.agents.improved_agent import ImprovedAgent
from grader import grade

def run_simulation(agent_class, task_path):
    with open(task_path) as f:
        task = yaml.safe_load(f)

    env = CrisisEnv(
        num_patients=task["num_patients"],
        num_ambulances=task["num_ambulances"],
        hospital_capacity=task["hospital_capacity"],
        seed=task["seed"]
    )

    agent = agent_class(env)
    obs = env.reset()
    done = False
    
    while not done:
        actions = agent.select_action(obs)
        action = Action(assignments=actions)
        obs, reward, done, info = env.step(action)

    metrics = {
        "survival_rate": info.get("survival_rate", 0),
        "avg_response_time": info.get("avg_response_time", 0),
        "deaths": info.get("deaths", 0)
    }
    
    score = grade(metrics)
    return metrics, score

def main():
    tasks = ["crisisflow/tasks/medium.yaml"] # Can add more tasks if available
    
    print(f"{'Agent':<20} | {'Task':<20} | {'Survive %':<10} | {'Deaths':<8} | {'Score':<8}")
    print("-" * 75)
    
    for task_path in tasks:
        task_name = os.path.basename(task_path)
        
        # Test Baseline
        base_metrics, base_score = run_simulation(BaselineAgent, task_path)
        print(f"{'BaselineAgent':<20} | {task_name:<20} | {base_metrics['survival_rate']*100:>9.1f}% | {base_metrics['deaths']:>7} | {base_score:>7.2f}")
        
        # Test Improved
        imp_metrics, imp_score = run_simulation(ImprovedAgent, task_path)
        print(f"{'ImprovedAgent':<20} | {task_name:<20} | {imp_metrics['survival_rate']*100:>9.1f}% | {imp_metrics['deaths']:>7} | {imp_score:>7.2f}")

if __name__ == "__main__":
    main()
