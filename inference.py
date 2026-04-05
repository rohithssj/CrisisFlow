import os
import yaml
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.environment.models import Action
from crisisflow.agents.baseline_agent import BaselineAgent
from grader import grade

# Load environment variables (MANDATORY)
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Load task (use medium by default)
with open("crisisflow/tasks/medium.yaml") as f:
    task = yaml.safe_load(f)

# Initialize environment
env = CrisisEnv(
    num_patients=task["num_patients"],
    num_ambulances=task["num_ambulances"],
    hospital_capacity=task["hospital_capacity"],
    seed=task["seed"]
)

agent = BaselineAgent(env)

print("[START]")

obs = env.reset()
done = False
step = 0

while not done:
    actions = agent.select_action(obs)
    # The agent returns a list of dictionaries, convert it into Action model
    action = Action(assignments=actions)

    obs, reward, done, info = env.step(action)

    action_desc = "Moving"
    if hasattr(action, "assignments") and action.assignments:
        action_desc = f"Assigned {len(action.assignments)} tasks"

    if info.get("rescued_patient_id") is not None:
        action_desc = f"Rescued Patient {info['rescued_patient_id']}"
    elif info.get("death_occurred"):
        action_desc = "Patient Died"

    reward_val = reward.score if hasattr(reward, 'score') else reward
    print(f"[STEP {step}] Reward: {reward_val:.4f} | Action: {action_desc}")

    step += 1

# Compute final score
metrics = {
    "survival_rate": info.get("survival_rate", 0),
    "avg_response_time": info.get("avg_response_time", 0),
    "deaths": info.get("deaths", 0)
}

final_score = grade(metrics)

print(f"[END] Score: {final_score:.2f}")
