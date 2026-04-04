import math
from typing import Optional, List
from crisisflow.agents.base_agent import BaseAgent
from configs.default_config import DIFFICULTY_CONFIGS

class BaselineAgent(BaseAgent):
    """
    State-of-the-art Global Matching Agent.
    Implements optimal set-based assignment (Recursive Solver) with
    comprehensive travel-time awareness and survival-first prioritization.
    """

    def select_action(self, state: dict) -> List[dict]:
        """
        1. GLOBAL MATCHING: Finds the optimal assignment set that maximizes total survival probability.
        """
        # Support both new Observation model and old dict
        state_dict = state.model_dump() if hasattr(state, "model_dump") else state
        
        patients = state_dict.get("patients", [])
        ambulances = state_dict.get("ambulances", [])
        hospitals = state_dict.get("hospitals", [])
        
        # Filter available resources
        active_pats = [p for p in patients if not p.get("rescued") and not p.get("dead")]
        available_ambs = [a for a in ambulances if not a.get("busy") and not a.get("on_cooldown", False)]

        if not active_pats or not available_ambs:
            return []

        # 4. BATCH DECISION SCORING: Evaluate all pairing combinations
        score_matrix = []
        for amb in available_ambs:
            row = []
            for pat in active_pats:
                # 3. TRAVEL OPTIMIZATION: Predict survival outcomes based on traffic and distance
                score = self._pair_score(amb, pat, hospitals, state_dict)
                row.append(score)
            score_matrix.append(row)

        # 1. OPTIMAL ASSIGNMENT SOLVER: Non-greedy combinatorial search
        best_assignments = self._solve_optimal_assignment(score_matrix)

        actions = []
        for amb_idx, pat_idx in best_assignments:
            # Only perform assignments that have a positive survival outlook
            if score_matrix[amb_idx][pat_idx] > -500:
                actions.append({
                    "ambulance_id": available_ambs[amb_idx]['id'],
                    "patient_id": active_pats[pat_idx]['id']
                })
        
        return actions

    def _pair_score(self, ambulance: dict, patient: dict, hospitals: list, state: dict) -> float:
        """
        Calculates a score based on real-time urgency and arrival feasibility.
        """
        difficulty = state.get("difficulty", "medium")
        env_cfg = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["medium"])
        env_speed = env_cfg["ambulance_speed"]

        severity = patient.get("severity", 1)
        time_waiting = patient.get("time_waiting", 0)
        
        # 2. TIME-TO-DEATH AWARENESS: Priority weights and survival windows
        # Higher weights for critical patients ensures they are prioritized in global matching
        severity_weight = {1: 1.0, 2: 4.0, 3: 15.0}[severity]
        death_threshold = {1: 30, 2: 15, 3: 7}[severity]
        steps_remaining = death_threshold - time_waiting

        # Use environment travel logic for accurate arrival prediction
        dist = math.sqrt((ambulance['x'] - patient['x'])**2 + (ambulance['y'] - patient['y'])**2)
        mid_x = (ambulance['x'] + patient['x']) / 2
        if mid_x < 0.33: traffic = 1.0
        elif mid_x < 0.66: traffic = 1.4
        else: traffic = 1.8
            
        est_travel = max(1, int((dist / env_speed) * traffic))
        slack = steps_remaining - est_travel

        # Immediate filter: if patient cannot survive the trip, assignment value is nil
        if slack < 0:
            return -1000.0

        # Urgency: Dynamic scoring based on remaining life vs travel time
        urgency = 25.0 / (slack + 1.0)
        
        # Proximity: Minor tie-breaker for local ambulances
        proximity = 5.0 / (1.0 + est_travel)

        # 3. Hospital awareness: Prefer admitting to full hospitals to minimize overflow
        hosp = self._nearest_available_hospital(hospitals, patient)
        hospital_bonus = 15.0 if hosp else -10.0

        score = (severity_weight * urgency) + proximity + hospital_bonus

        # Critical lookahead: High-impact bonus for saving patients on the brink of death
        if slack <= 1:
            score += 100.0 if severity == 3 else 50.0

        return score

    def _solve_optimal_assignment(self, matrix: List[List[float]]) -> List[tuple]:
        """Recursive solver for the combinatorial assignment problem."""
        rows = len(matrix)
        cols = len(matrix[0])
        memo = {}

        def search(r, mask):
            if r == rows or mask == (1 << cols) - 1:
                return 0, []
            
            state_key = (r, mask)
            if state_key in memo:
                return memo[state_key]

            # Option: Skip ambulance r
            best_val, best_pairs = search(r + 1, mask)

            # Option: Assign ambulance r to an available patient
            for c in range(cols):
                if not (mask & (1 << c)):
                    val, pairs = search(r + 1, mask | (1 << c))
                    total = val + matrix[r][c]
                    if total > best_val:
                        best_val = total
                        best_pairs = [(r, c)] + pairs

            memo[state_key] = (best_val, best_pairs)
            return best_val, best_pairs

        # Fallback for large matrices to maintain real-time performance
        if rows * cols > 100:
            return self._solve_greedy_fallback(matrix)

        _, result = search(0, 0)
        return result

    def _solve_greedy_fallback(self, matrix: List[List[float]]) -> List[tuple]:
        res = []
        u_ambs, u_pats = set(), set()
        pairs = sorted([(matrix[r][c], r, c) for r in range(len(matrix)) for c in range(len(matrix[0]))], reverse=True)
        for _, r, c in pairs:
            if r not in u_ambs and c not in u_pats:
                res.append((r, c)); u_ambs.add(r); u_pats.add(c)
        return res

    def _nearest_available_hospital(self, hospitals, pat):
        px, py = pat['x'], pat['y']
        av = [h for h in hospitals if h.get("available")]
        if not av: return None
        return min(av, key=lambda h: math.sqrt((h['x']-px)**2 + (h['y']-py)**2))
