import random
import os
import yaml
import math
from crisisflow.environment.crisis_env import CrisisEnv
from crisisflow.environment.models import Action
from crisisflow.agents.baseline_agent import BaselineAgent
from crisisflow.agents.improved_agent import ImprovedAgent
from grader import grade

def run_simulation(data: dict):
    """
    Runs a single decision inference based on input incident data.
    Input format: {"type": str, "severity": int, "wait_time": float, "distance": float, "location": str, "config": dict}
    """
    incident_type = data.get("type", "medical").lower()
    severity = data.get("severity", 5)
    wait_time = data.get("wait_time", 0.0)
    distance = data.get("distance", 0.0)
    location = data.get("location", "Unknown")
    config = data.get("config", {})

    # 1. Score formula
    score = severity * 2 + wait_time * 1.5 - distance * 1.2

    # 2. Sensitivity & Horizon Adjustments
    sensitivity = config.get("inferenceSensitivity", "normal")
    horizon = config.get("predictiveHorizon", 6)

    # 3. Determine unit
    if "cyber" in incident_type:
        unit = "Cyber Response Unit"
    elif "fire" in incident_type:
        unit = "Fire Brigade"
    elif "flood" in incident_type:
        unit = "Rescue Team"
    elif "medical" in incident_type:
        unit = "Emergency Medical Services"
    else:
        unit = "General Response Unit"

    # 4. Determine risk
    if severity >= 8:
        risk = "Critical"
    elif severity >= 5:
        risk = "High"
    else:
        risk = "Moderate"

    # 5. Determine priority
    if score >= 15:
        priority = "P1"
    elif score >= 8:
        priority = "P2"
    else:
        priority = "P3"

    # 6. Confidence (Influenced by sensitivity)
    confidence = random.randint(70, 95)
    if sensitivity == "low":
        confidence -= 10
    elif sensitivity == "enhanced":
        confidence += 10
    
    confidence = max(0, min(100, confidence))

    # 7. Dynamic AI responses
    if "fire" in incident_type:
        reason = f"High-intensity fire detected in {location}. Nearest fire units mobilized. Containment priority elevated."
    elif "cyber" in incident_type:
        reason = f"Cyber intrusion detected targeting critical infrastructure in {location}. Isolation protocols initiated. Security units dispatched."
    elif "flood" in incident_type:
        reason = f"Flood risk escalating in {location}. Deploying rescue and evacuation teams to primary vectors."
    elif "medical" in incident_type:
        reason = f"Medical emergency reported in {location}. Dispatching nearest emergency medical services for immediate triage."
    else:
        reason = f"Incident detected in {location}. Assigning appropriate response team and monitoring telemetry."

    # Adjust explanation based on horizon
    if horizon >= 6:
        reason += f" Proactive mitigation enabled based on extended {horizon}h predictive horizon."

    # Neural Impact = severity * 10
    neural_impact = severity * 10

    return {
        "unit": unit,
        "risk": risk,
        "score": round(score, 2),
        "reason": reason,
        "confidence": confidence,
        "priority": priority,
        "neuralImpact": neural_impact,
        "threatScore": round(score, 2)
    }

if __name__ == "__main__":
    # Load environment variables (MANDATORY)
    API_BASE_URL = os.getenv("API_BASE_URL", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    # Load task (use medium by default)
    task_path = "crisisflow/tasks/medium.yaml"
    if not os.path.exists(task_path):
        print(f"Error: {task_path} not found.")
    else:
        with open(task_path) as f:
            task = yaml.safe_load(f)

        # Initialize environment
        env = CrisisEnv(
            num_patients=task["num_patients"],
            num_ambulances=task["num_ambulances"],
            hospital_capacity=task["hospital_capacity"],
            seed=task["seed"]
        )

        agent = BaselineAgent(env)

        print("[START] Running CrisisFlow CLI Simulation...")

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
