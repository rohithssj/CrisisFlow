import yaml
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.environment.models import Action
from crisisflow.agents.baseline_agent import BaselineAgent
from grader import grade

with open("crisisflow/tasks/easy.yaml") as f:
    task = yaml.safe_load(f)

env = CrisisEnv(**task)
agent = BaselineAgent(env)

obs = env.reset()
done = False

while not done:
    actions = agent.select_action(obs)
    # Convert list of dicts from baseline agent to Action model
    action = Action(assignments=actions)
    obs, reward, done, info = env.step(action)

metrics = {
    "survival_rate": info.get("survival_rate", 0),
    "avg_response_time": info.get("avg_response_time", 0),
    "deaths": info.get("dead", 0),
    "critical_saved": info.get("critical_rescued", 0)
}

print("Simulation finished successfully")
print("Metrics:", metrics)
print("Grade:", grade(metrics))
