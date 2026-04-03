import math
from typing import Optional
from crisisflow.agents.base_agent import BaseAgent

class BaselineAgent(BaseAgent):
    """
    A deterministic baseline agent using a greedy heuristic.
    It prioritizes high-severity patients and assigns the nearest available ambulance.
    """

    def select_action(self, state: dict) -> Optional[dict]:
        """
        Implementation of greedy dispatch logic.
        """
        patients = state.get("patients", [])
        ambulances = state.get("ambulances", [])

        # 1. Find all active patients (not rescued and not dead)
        active_patients = [
            p for p in patients if not p.get("rescued") and not p.get("dead")
        ]

        if not active_patients:
            return None

        # 2. Select highest severity patient first (sort by severity descending, then time_waiting descending)
        active_patients.sort(key=lambda p: (p.get("severity", 0), p.get("time_waiting", 0)), reverse=True)
        target_patient = active_patients[0]

        # 3. Find all available ambulances
        available_ambulances = [a for a in ambulances if not a.get("busy")]

        if not available_ambulances:
            return None

        # 4. Assign nearest available ambulance to the selected patient
        def get_dist(amb, pat):
            return math.sqrt((amb.get("x", 0) - pat.get("x", 0))**2 + (amb.get("y", 0) - pat.get("y", 0))**2)

        nearest_ambulance = min(
            available_ambulances,
            key=lambda a: get_dist(a, target_patient)
        )

        return {
            "ambulance_id": nearest_ambulance.get("id"),
            "patient_id": target_patient.get("id")
        }
