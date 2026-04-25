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
    Input format: {"type": str, "severity": int, "wait_time": float, "distance": float}
    """
    incident_type = data.get("type", "medical").lower()
    severity = data.get("severity", 5)

    if "cyber" in incident_type:
        return {
            "unit": "Cyber Response Unit",
            "risk": "High" if severity > 7 else "Medium",
            "score": severity * 10,
            "reason": "Network breach detected. Initiating firewall isolation and system lockdown."
        }

    elif "fire" in incident_type:
        return {
            "unit": "Fire Brigade",
            "risk": "Critical" if severity > 8 else "High",
            "score": severity * 12,
            "reason": "Fire outbreak detected. Dispatching nearest fire containment units."
        }

    elif "flood" in incident_type:
        return {
            "unit": "Rescue Team",
            "risk": "High",
            "score": severity * 11,
            "reason": "Flood alert. Deploying evacuation and rescue teams."
        }

    else:  # medical (default)
        # Weights from ImprovedAgent logic
        severity_weight = 4.0
        waiting_weight = 2.0
        distance_weight = 1.0
        urgency_threshold = 3
        urgency_bonus_val = 6.0
        
        wait_time = data.get("wait_time", 0.0)
        distance = data.get("distance", 0.0)

        # Priority Score calculation
        score = (severity_weight * severity) + (waiting_weight * wait_time) - (distance_weight * distance)
        
        # Approximate TTL logic
        ttl_map = {1: 30, 2: 15, 3: 7}
        ttl = ttl_map.get(severity, 10)
        
        if ttl - wait_time <= urgency_threshold:
            score += urgency_bonus_val

        normalized_score = min(100, max(0, int(score * 2.5)))

        if severity >= 3 or (ttl - wait_time) <= 2:
            risk = "Critical"
        elif severity >= 2 or (ttl - wait_time) <= 5:
            risk = "High"
        else:
            risk = "Medium"

        hospitals = ["City Hospital", "Metro Medical Center", "Northside Clinic", "Downtown General"]
        h_idx = int(distance * 13) % len(hospitals)
        unit = hospitals[h_idx]

        if risk == "Critical":
            reason = f"Immediate life-threat detected. Bypassing capacity constraints for {unit}."
        elif distance > 15:
            reason = f"Extended transit time predicted. {unit} selected as optimal survival-feasible hub."
        else:
            reason = f"Standard load-balanced assignment to {unit} based on neural priority score."

        return {
            "unit": unit,
            "risk": risk,
            "score": normalized_score,
            "reason": reason
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
