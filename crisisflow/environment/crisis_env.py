"""
crisis_env.py — Core simulation environment
============================================
Simulates a city grid where emergencies happen and resources must be dispatched.

The environment exposes the OpenEnv API:
  reset()       → returns initial state dict
  step(actions) → returns (state, reward, done, info)
  state()       → returns current state dict
"""

import random
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
from configs.default_config import DIFFICULTY_CONFIGS
from crisisflow.environment.models import Observation, Action, Reward


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Patient:
    id: int
    x: float
    y: float
    severity: int          # 1 (minor) → 3 (critical)
    time_waiting: int = 0  # steps waited without help
    ttl: int = 50          # default ttl
    rescued: bool = False
    dead: bool = False
    # Track if patient was rescued without an available hospital bed
    rescued_without_hospital: bool = False

    def tick(self, difficulty: str = "medium"):
        """Advance one time step. Critical patients degrade faster."""
        if not self.rescued and not self.dead:
            self.time_waiting += 1
            # Death logic: only if wait_time > ttl AND not yet rescued
            if self.time_waiting >= self.ttl:
                self.dead = True


@dataclass
class Ambulance:
    id: int
    x: float
    y: float
    busy: bool = False
    target_patient_id: Optional[int] = None
    steps_to_arrive: int = 0
    # ADDED: Ambulance cooldown after delivering a patient
    cooldown_remaining: int = 0

    def dispatch(self, patient: Patient, travel_steps: int):
        self.busy = True
        self.target_patient_id = patient.id
        self.steps_to_arrive = travel_steps

    def tick(self):
        # 3. AMBULANCE COOLDOWN: Decrement cooldown if active
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            
        if self.busy and self.steps_to_arrive > 0:
            self.steps_to_arrive -= 1
        if self.busy and self.steps_to_arrive == 0:
            self.busy = False
            self.target_patient_id = None
            # 3. AMBULANCE COOLDOWN: Set cooldown after delivery
            self.cooldown_remaining = 3


@dataclass
class Hospital:
    id: int
    x: float
    y: float
    capacity: int
    current_load: int = 0

    @property
    def available(self) -> bool:
        return self.current_load < self.capacity

    def admit(self):
        if self.available:
            self.current_load += 1

    def tick(self):
        """Hospitals discharge one patient per 5 steps."""
        if self.current_load > 0 and random.random() < 0.2:
            self.current_load -= 1


# ── Main Environment ──────────────────────────────────────────────────────────

class CrisisEnv:
    """
    OpenEnv-compatible crisis response environment.

    Actions are list of dicts:
      [{"ambulance_id": int, "patient_id": int}, ...]

    Multiple actions can dispatch multiple ambulances per step.
    """

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None, **kwargs):
        # Fallback to the task "name" if "difficulty" wasn't explicitly passed but name was
        difficulty_name = kwargs.get("name", difficulty)
        
        assert difficulty_name in ("easy", "medium", "hard"), \
            f"difficulty must be 'easy', 'medium', or 'hard', got '{difficulty_name}'"
            
        self.difficulty = difficulty_name
        
        # Base config on difficulty
        self.config = DIFFICULTY_CONFIGS[self.difficulty].copy()
        
        # Override with any task-specific parameters
        if "num_patients" in kwargs:
            self.config["num_crises"] = kwargs["num_patients"]
        if "num_ambulances" in kwargs:
            self.config["num_ambulances"] = kwargs["num_ambulances"]
        if "hospital_capacity" in kwargs:
            self.config["hospital_capacity"] = kwargs["hospital_capacity"]
            
        # 2a. RANDOM SCENARIOS: Dynamic seed if none provided
        task_seed = kwargs.get("seed", seed)
        self.seed = task_seed if task_seed is not None else random.randint(0, 99999)
        self._rng = random.Random(self.seed)

        # Will be populated on reset()
        self.patients: list[Patient] = []
        self.ambulances: list[Ambulance] = []
        self.hospitals: list[Hospital] = []
        self.step_count: int = 0
        self.done: bool = False
        self._next_patient_id: int = 0
        self._crisis_schedule: list[int] = []  # steps when new crises spawn

        # Score tracking
        self._rescued_count: int = 0
        self._dead_count: int = 0
        self._total_wait_time: int = 0
        self._response_times: list[int] = []
        
        # 1. BETTER METRICS: Track new counters
        self._critical_rescued = 0
        self._overflow_rescues = 0

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to a fresh episode. Returns the initial observation."""
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        cfg = self.config
        self._rng = random.Random(self.seed)
        self.step_count = 0
        self.done = False
        self._next_patient_id = 0
        self._rescued_count = 0
        self._dead_count = 0
        self._total_wait_time = 0
        self._response_times = []
        
        # 1. BETTER METRICS: Reset counters
        self._critical_rescued = 0
        self._overflow_rescues = 0

        # Place hospitals at fixed landmark positions
        hospital_positions = [
            (0.15, 0.2), (0.85, 0.2), (0.5, 0.85)
        ][:cfg["num_hospitals"]]
        self.hospitals = [
            Hospital(id=i, x=px, y=py, capacity=cfg["hospital_capacity"])
            for i, (px, py) in enumerate(hospital_positions)
        ]

        # Place ambulances near hospitals
        self.ambulances = []
        for i in range(cfg["num_ambulances"]):
            h = self.hospitals[i % len(self.hospitals)]
            self.ambulances.append(Ambulance(
                id=i,
                x=h.x + self._rng.uniform(-0.05, 0.05),
                y=h.y + self._rng.uniform(-0.05, 0.05)
            ))

        # Schedule crisis spawning across the episode
        total_crises = cfg["num_crises"]
        max_steps = cfg["max_steps"]
        spawn_steps = sorted(self._rng.sample(
            range(0, int(max_steps * 0.7)), total_crises
        ))
        self._crisis_schedule = spawn_steps

        # Spawn initial batch of crises (first 30% immediately)
        initial_count = max(1, total_crises // 3)
        self.patients = []
        for _ in range(initial_count):
            self._spawn_patient()

        return self.state()

    def state(self) -> Observation:
        return self._get_observation()

    def _get_observation(self) -> Observation:
        legacy = self._get_legacy_state()
        return Observation(
            patients=legacy["patients"],
            ambulances=legacy["ambulances"],
            hospitals=legacy["hospitals"],
            time_step=legacy["step"]
        )

    def step(self, action: Union[Action, List[dict], dict, None]) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute actions from Action model (or legacy format) and advance the world by one time step.
        """
        assert not self.done, "Episode is over. Call reset() first."

        if action and hasattr(action, 'assignments'):
            actions = action.assignments
        elif isinstance(action, list):
            actions = action
        elif isinstance(action, dict):
            actions = [action]
        else:
            actions = []

        reward_parts = {}

        # 1. Execute the agent's actions
        dispatch_count = 0
        for action in actions:
            amb_id = action.get("ambulance_id")
            pat_id = action.get("patient_id")
            amb = self._get_ambulance(amb_id)
            pat = self._get_patient(pat_id)

            already_targeted = any(
                a.target_patient_id == pat_id for a in self.ambulances if a.busy
            )
            
            # 3. AMBULANCE COOLDOWN: Check if ambulance is on cooldown before dispatch
            is_valid_dispatch = (
                amb and pat and 
                not amb.busy and 
                amb.cooldown_remaining == 0 and
                not pat.rescued and 
                not pat.dead and 
                not already_targeted
            )

            if is_valid_dispatch:
                travel_steps = self._travel_time(amb.x, amb.y, pat.x, pat.y)
                amb.dispatch(pat, travel_steps)
                self._response_times.append(pat.time_waiting)
                dispatch_count += 1
                reward_parts[f"dispatch_{amb_id}"] = 0.05

        # 2. Advance ambulances — mark patients as rescued on arrival
        for amb in self.ambulances:
            if amb.busy and amb.steps_to_arrive == 1:
                pat = self._get_patient(amb.target_patient_id)
                if pat and not pat.dead:
                    pat.rescued = True
                    self._rescued_count += 1
                    
                    # 1. BETTER METRICS: Increment critical rescued counter
                    if pat.severity == 3:
                        self._critical_rescued += 1
                    
                    # 3. HOSPITAL CAPACITY OVERFLOW: Check capacity during rescue
                    hosp = self._nearest_available_hospital(pat.x, pat.y)
                    if hosp:
                        hosp.admit()
                    else:
                        pat.rescued_without_hospital = True
                        # 1. BETTER METRICS: Increment overflow rescues counter
                        self._overflow_rescues += 1
                        
                    reward_parts[f"rescue_{pat.id}"] = self._rescue_reward(pat)
            amb.tick()

        # 3. Tick patients (ageing, potential death)
        before_dead = self._dead_count
        for pat in self.patients:
            pat.tick(difficulty=self.difficulty)
            if pat.dead and not pat.rescued:
                if pat.id not in self._counted_dead:
                    self._counted_dead.add(pat.id)
                    self._dead_count += 1

        new_deaths = self._dead_count - before_dead
        if new_deaths > 0:
            penalty_per_death = -0.5 if self.difficulty == "hard" else -0.3
            reward_parts["death_penalty"] = penalty_per_death * new_deaths

        # 4. Tick hospitals
        for hosp in self.hospitals:
            hosp.tick()

        # 5. Spawn scheduled new crises
        for spawn_step in self._crisis_schedule:
            if spawn_step == self.step_count:
                self._spawn_patient()

        self.step_count += 1

        # 6. Compute reward and check termination
        total_reward = self._compute_reward(reward_parts)
        self.done = self._check_done()
        info = self._build_info()

        return self.state(), Reward(score=total_reward, details=reward_parts), self.done, info

    def _get_legacy_state(self) -> dict:
        """Return the current observable state as a plain dict (used to build observation)."""
        return {
            "step": self.step_count,
            "difficulty": self.difficulty,
            "patients": [
                {
                    "id": p.id,
                    "x": round(p.x, 3),
                    "y": round(p.y, 3),
                    "severity": p.severity,
                    "time_waiting": p.time_waiting,
                    "ttl": p.ttl,
                    "rescued": p.rescued,
                    "dead": p.dead,
                }
                for p in self.patients
            ],
            "ambulances": [
                {
                    "id": a.id,
                    "x": round(a.x, 3),
                    "y": round(a.y, 3),
                    "busy": a.busy,
                    "target_patient_id": a.target_patient_id,
                    "steps_to_arrive": a.steps_to_arrive,
                    # 3. AMBULANCE COOLDOWN: Added cooldown visibility
                    "on_cooldown": a.cooldown_remaining > 0,
                    "cooldown_remaining": a.cooldown_remaining,
                    # 1. TRAFFIC DELAY: Current zone of the ambulance
                    "traffic_zone": self._get_zone_name(a.x)
                }
                for a in self.ambulances
            ],
            "hospitals": [
                {
                    "id": h.id,
                    "x": round(h.x, 3),
                    "y": round(h.y, 3),
                    "capacity": h.capacity,
                    "current_load": h.current_load,
                    "available": h.available,
                }
                for h in self.hospitals
            ],
            "stats": {
                "rescued": self._rescued_count,
                "dead": self._dead_count,
                "active_patients": sum(
                    1 for p in self.patients if not p.rescued and not p.dead
                ),
            },
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    # 1. TRAFFIC DELAY: Determin traffic multiplier based on route mid-point
    def _traffic_multiplier(self, x1, y1, x2, y2) -> float:
        mid_x = (x1 + x2) / 2
        if mid_x < 0.33:
            return 1.0  # Low Traffic
        elif mid_x < 0.66:
            return 1.4  # Medium Traffic
        else:
            return 1.8  # High Traffic

    def _get_zone_name(self, x: float) -> str:
        if x < 0.33: return "low_traffic"
        if x < 0.66: return "medium_traffic"
        return "high_traffic"

    @property
    def _counted_dead(self):
        if not hasattr(self, "_counted_dead_set"):
            self._counted_dead_set = set()
        return self._counted_dead_set

    def _spawn_patient(self):
        """2b. RANDOM SCENARIOS: Spawn with 30% clustering logic."""
        severity_weights = self.config["severity_weights"]
        severity = self._rng.choices([1, 2, 3], weights=severity_weights)[0]
        
        # 30% of the time, spawn near an existing patient
        if self.patients and self._rng.random() < 0.3:
            base_pat = self._rng.choice(self.patients)
            x_offset = self._rng.uniform(-0.15, 0.15)
            y_offset = self._rng.uniform(-0.15, 0.15)
            x = max(0.05, min(0.95, base_pat.x + x_offset))
            y = max(0.05, min(0.95, base_pat.y + y_offset))
        else:
            # 70% random spawn
            x = self._rng.uniform(0.05, 0.95)
            y = self._rng.uniform(0.05, 0.95)

        pat = Patient(
            id=self._next_patient_id,
            x=x,
            y=y,
            severity=severity,
            ttl={1: 25, 2: 12, 3: 5} if self.difficulty == "hard" else {1: 30, 2: 15, 3: 7}[severity]
        )
        self.patients.append(pat)
        self._next_patient_id += 1

    def _get_ambulance(self, amb_id: Optional[int]) -> Optional[Ambulance]:
        if amb_id is None:
            return None
        for a in self.ambulances:
            if a.id == amb_id:
                return a
        return None

    def _get_patient(self, pat_id: Optional[int]) -> Optional[Patient]:
        if pat_id is None:
            return None
        for p in self.patients:
            if p.id == pat_id:
                return p
        return None

    def _travel_time(self, x1, y1, x2, y2) -> int:
        """Euclidean distance on [0,1] grid → discrete travel steps with Traffic."""
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = self.config["ambulance_speed"]
        base_steps = dist / speed
        # 1. TRAFFIC DELAY: Apply traffic multiplier
        multiplier = self._traffic_multiplier(x1, y1, x2, y2)
        return max(1, int(base_steps * multiplier))

    def _nearest_available_hospital(self, x, y) -> Optional[Hospital]:
        available = [h for h in self.hospitals if h.available]
        if not available:
            return None
        return min(available, key=lambda h: math.sqrt((h.x - x) ** 2 + (h.y - y) ** 2))

    def _rescue_reward(self, patient: Patient) -> float:
        """
        Reward for rescuing a patient.
        Higher reward for: more critical patients, faster response, early rescue.
        """
        severity_bonus = {1: 0.1, 2: 0.2, 3: 0.4}[patient.severity]
        # Speed bonus: penalise long waits (max wait before death)
        max_wait = {1: 30, 2: 15, 3: 7}[patient.severity]
        speed_factor = max(0.0, 1.0 - patient.time_waiting / max_wait)
        reward = severity_bonus * (0.5 + 0.5 * speed_factor)
        
        # 3. HOSPITAL CAPACITY OVERFLOW: Penalty if rescued without hospital bed
        if patient.rescued_without_hospital:
            multiplier = 0.5 if self.difficulty != "hard" else 0.4 # Higher penalty for hard
            reward *= multiplier
            
        return reward

    def _compute_reward(self, parts: dict) -> float:
        """
        Aggregate reward components and normalise to a meaningful range.
        Components:
          + rescue bonus (per patient, severity-weighted)
          + dispatch bonus (small immediate reward for valid dispatches)
          - death penalty (per patient death)
          - idle penalty (all ambulances free but active patients exist)
        """
        reward = sum(parts.values())
        
        # New: Waiting penalty for HARD mode to encourage rapid response
        if self.difficulty == "hard":
            total_wait = sum(p.time_waiting for p in self.patients if not p.rescued and not p.dead)
            reward -= 0.001 * total_wait

        # Small penalty for idle ambulances when patients need help
        active = [p for p in self.patients if not p.rescued and not p.dead]
        idle_ambs = [a for a in self.ambulances if not a.busy and a.cooldown_remaining == 0]
        if active and idle_ambs:
            critical_active = [p for p in active if p.severity == 3]
            if critical_active:
                reward -= 0.01 * len(idle_ambs)

        return round(max(-1.0, min(1.0, reward)), 4)

    def _check_done(self) -> bool:
        cfg = self.config
        if self.step_count >= cfg["max_steps"]:
            return True
        # All crises resolved (rescued or dead) and no more scheduled
        remaining_spawns = sum(
            1 for s in self._crisis_schedule if s > self.step_count
        )
        all_resolved = all(p.rescued or p.dead for p in self.patients)
        return all_resolved and remaining_spawns == 0

    def _build_info(self) -> dict:
        """Build the final score (0.0 → 1.0) and episode info."""
        total = len(self.patients)
        rescued = self._rescued_count
        
        if total == 0:
            score = 1.0
            survival_rate = 0.0
        else:
            # 1. FIX: Calculate survival rate normalized to 0.0 - 1.0
            survival_rate = round(rescued / total, 4)
            
            # Score = weighted rescue rate
            rescue_value = 0.0
            for p in self.patients:
                if p.rescued:
                    severity_weight = {1: 1.0, 2: 2.0, 3: 3.0}[p.severity]
                    speed_bonus = max(0, 1 - p.time_waiting / 20)
                    base_val = severity_weight * (0.7 + 0.3 * speed_bonus)
                    # Apply penalty for score if rescued without hospital
                    if p.rescued_without_hospital:
                        base_val *= 0.5
                    rescue_value += base_val
                    
            max_possible = sum(
                {1: 1.0, 2: 2.0, 3: 3.0}[p.severity] for p in self.patients
            )
            score = rescue_value / max_possible if max_possible > 0 else 1.0

        avg_response = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times else 0
        )

        return {
            "score": round(min(1.0, max(0.0, score)), 4),
            "survival_rate": survival_rate,
            "critical_saved": self._critical_rescued,
            "overflow_rescues": self._overflow_rescues,
            "rescued": rescued,
            "deaths": self._dead_count,
            "total_patients": total,
            "avg_response_time": round(avg_response, 2),
            "steps_taken": self.step_count,
        }

    def final_score(self) -> float:
        """Convenience: returns the 0.0–1.0 episode score."""
        return self._build_info()["score"]