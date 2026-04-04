from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.environment.models import Action

env = CrisisEnv()

print("Testing reset()...")
obs = env.reset()
print("RESET OK:", obs)

print("\nTesting step()...")
action = Action(assignments=[])
obs, reward, done, _ = env.step(action)

print("STEP OK")
print("Reward:", reward)
print("Done:", done)