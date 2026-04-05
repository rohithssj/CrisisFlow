import math
from typing import List, Dict, Any
from crisisflow.agents.base_agent import BaseAgent

class ImprovedAgent(BaseAgent):
    """
    Improved decision-making agent for CrisisFlow.
    Uses a priority scoring system to assign the nearest ambulance to the highest priority patients.
    
    Priority Score = (3.0 * severity) + (1.5 * waiting_time) - (1.0 * distance)
    """

    def select_action(self, state: Any) -> List[Dict[str, Any]]:
        """
        Selects actions based on a priority scoring formula.
        
        Steps:
        1. Identify active patients (not rescued, not dead).
        2. Identify available ambulances (not busy, not on cooldown).
        3. Compute priority for all possible (ambulance, patient) pairs.
        4. Greedily assign the highest priority pairs.
        """
        # Support both new Observation model and old dict
        state_dict = state.model_dump() if hasattr(state, "model_dump") else state
        
        patients = state_dict.get("patients", [])
        ambulances = state_dict.get("ambulances", [])
        
        # Identify patients already being targeted by busy ambulances
        targeted_pat_ids = {a['target_patient_id'] for a in ambulances if a.get("busy") and a.get("target_patient_id") is not None}
        
        # Filter for active patients (not rescued, not dead) who are NOT already being helped
        active_pats = [p for p in patients if not p.get("rescued") and not p.get("dead") and p['id'] not in targeted_pat_ids]
        
        # Identify available ambulances (not busy, not on cooldown)
        available_ambs = [a for a in ambulances if not a.get("busy") and not a.get("on_cooldown", False)]

        if not active_pats or not available_ambs:
            return []

        # Weights for the priority score
        severity_weight = 3.0
        waiting_weight = 1.5
        distance_weight = 1.0
        
        # Survival thresholds from environment logic
        death_thresholds = {1: 30, 2: 15, 3: 7}
        # Approximate speed/traffic to estimate travel time (used for survival check only)
        # Using a conservative estimate (lower speed/higher traffic)
        avg_speed = 0.04
        avg_traffic = 1.4

        scored_pairs = []
        for amb in available_ambs:
            for pat in active_pats:
                # Euclidean distance
                dist = math.sqrt((amb['x'] - pat['x'])**2 + (amb['y'] - pat['y'])**2)
                
                # Check for survival feasibility (Survival Check)
                # If patient is doomed even if we send an ambulance, skip this pair
                est_travel_steps = max(1, int((dist / avg_speed) * avg_traffic))
                if pat.get("time_waiting", 0) + est_travel_steps >= death_thresholds.get(pat.get("severity", 1), 30):
                    continue

                # Priority Score calculation: priority = (3.0 * severity) + (1.5 * waiting_time) - (1.0 * distance)
                priority = (severity_weight * pat.get("severity", 1)) + \
                           (waiting_weight * pat.get("time_waiting", 0)) - \
                           (distance_weight * dist)
                
                scored_pairs.append({
                    "priority": priority,
                    "ambulance_id": amb['id'],
                    "patient_id": pat['id']
                })

        # Sort by priority score in descending order
        scored_pairs.sort(key=lambda x: x["priority"], reverse=True)
        
        assignments = []
        assigned_ambs = set()
        assigned_pats = set()
        
        for pair in scored_pairs:
            amb_id = pair["ambulance_id"]
            pat_id = pair["patient_id"]
            
            # Ensure each ambulance and patient is assigned only once in this step
            if amb_id not in assigned_ambs and pat_id not in assigned_pats:
                assignments.append({
                    "ambulance_id": amb_id,
                    "patient_id": pat_id
                })
                assigned_ambs.add(amb_id)
                assigned_pats.add(pat_id)
        
        return assignments
