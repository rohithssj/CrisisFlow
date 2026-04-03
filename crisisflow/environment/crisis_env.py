"""
crisis_env.py — Core simulation environment
============================================
Simulates a city grid where emergencies happen and resources must be dispatched.

The environment exposes the OpenEnv API:
  reset()       → returns initial state dict
  step(action)  → returns (state, reward, done, info)
  state()       → returns current state dict
"""

import random
import math
from dataclasses import dataclass, field, asdict
from typing import Optional
from configs.default_config import DIFFICULTY_CONFIGS


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Patient:
    id: int
    x: float
    y: float
    severity: int          # 1 (minor) → 3 (critical)
    time_waiting: int = 0  # steps waited without help
    rescued: bool = False
    dead: bool = False

    def tick(self):
        """Advance one time step. Critical patients degrade faster."""
        if not self.rescued and not self.dead:
            self.time_waiting += 1
            death_threshold = {1: 30, 2: 15, 3: 7}[self.severity]
            if self.time_waiting >= death_threshold:
                self.dead = True


@dataclass
class Ambulance:
    id: int
    x: float
    y: float
    busy: bool = False
    target_patient_id: Optional[int] = None
    steps_to_arrive: int = 0

    def dispatch(self, patient: Patient, travel_steps: int):
        self.busy = True
        self.target_patient_id = patient.id
        self.steps_to_arrive = travel_steps

    def tick(self):
        if self.busy and self.steps_to_arrive > 0:
            self.steps_to_arrive -= 1
        if self.busy and self.steps_to_arrive == 0:
            self.busy = False
            self.target_patient_id = None


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

    Actions are dicts:
      {"ambulance_id": int, "patient_id": int}

    One action dispatches one ambulance to one patient per step.
    The agent may also pass None to skip a step.
    """

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        assert difficulty in ("easy", "medium", "hard"), \
            f"difficulty must be 'easy', 'medium', or 'hard', got '{difficulty}'"
        self.difficulty = difficulty
        self.config = DIFFICULTY_CONFIGS[difficulty]
        self.seed = seed
        self._rng = random.Random(seed)

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

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset to a fresh episode. Returns the initial state dict."""
        cfg = self.config
        self._rng = random.Random(self.seed)
        self.step_count = 0
        self.done = False
        self._next_patient_id = 0
        self._rescued_count = 0
        self._dead_count = 0
        self._total_wait_time = 0
        self._response_times = []

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

    def step(self, action: Optional[dict]) -> tuple[dict, float, bool, dict]:
        """
        Execute one action and advance the world by one time step.

        action: {"ambulance_id": int, "patient_id": int}  or  None

        Returns: (state, reward, done, info)
        """
        assert not self.done, "Episode is over. Call reset() first."

        reward_parts = {}

        # 1. Execute the agent's action
        dispatch_success = False
        if action is not None:
            amb_id = action.get("ambulance_id")
            pat_id = action.get("patient_id")
            amb = self._get_ambulance(amb_id)
            pat = self._get_patient(pat_id)

            already_targeted = any(
                a.target_patient_id == pat_id for a in self.ambulances if a.busy
            )
            if amb and pat and not amb.busy and not pat.rescued and not pat.dead and not already_targeted:
                travel_steps = self._travel_time(amb.x, amb.y, pat.x, pat.y)
                amb.dispatch(pat, travel_steps)
                self._response_times.append(pat.time_waiting)
                dispatch_success = True
                reward_parts["dispatch_bonus"] = 0.05

        # 2. Advance ambulances — mark patients as rescued on arrival
        for amb in self.ambulances:
            if amb.busy and amb.steps_to_arrive == 1:
                pat = self._get_patient(amb.target_patient_id)
                if pat and not pat.dead:
                    pat.rescued = True
                    self._rescued_count += 1
                    # Admit to nearest available hospital
                    hosp = self._nearest_available_hospital(pat.x, pat.y)
                    if hosp:
                        hosp.admit()
                    reward_parts["rescue"] = self._rescue_reward(pat)
            amb.tick()

        # 3. Tick patients (ageing, potential death)
        before_dead = self._dead_count
        for pat in self.patients:
            pat.tick()
            if pat.dead and not pat.rescued:
                if pat.id not in self._counted_dead:
                    self._counted_dead.add(pat.id)
                    self._dead_count += 1

        new_deaths = self._dead_count - before_dead
        if new_deaths > 0:
            reward_parts["death_penalty"] = -0.3 * new_deaths

        # 4. Tick hospitals
        for hosp in self.hospitals:
            hosp.tick()

        # 5. Spawn scheduled new crises
        for spawn_step in self._crisis_schedule:
            if spawn_step == self.step_count:
                self._spawn_patient()

        self.step_count += 1

        # 6. Compute reward and check termination
        reward = self._compute_reward(reward_parts)
        self.done = self._check_done()
        info = self._build_info()

        return self.state(), reward, self.done, info

    def state(self) -> dict:
        """Return the current observable state as a plain dict."""
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

    @property
    def _counted_dead(self):
        if not hasattr(self, "_counted_dead_set"):
            self._counted_dead_set = set()
        return self._counted_dead_set

    def _spawn_patient(self):
        severity_weights = self.config["severity_weights"]
        severity = self._rng.choices([1, 2, 3], weights=severity_weights)[0]
        pat = Patient(
            id=self._next_patient_id,
            x=self._rng.uniform(0.05, 0.95),
            y=self._rng.uniform(0.05, 0.95),
            severity=severity,
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
        """Euclidean distance on [0,1] grid → discrete travel steps."""
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed = self.config["ambulance_speed"]
        return max(1, int(dist / speed))

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
        return severity_bonus * (0.5 + 0.5 * speed_factor)

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

        # Small penalty for idle ambulances when patients need help
        active = [p for p in self.patients if not p.rescued and not p.dead]
        idle_ambs = [a for a in self.ambulances if not a.busy]
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
        if total == 0:
            score = 1.0
        else:
            # Score = weighted rescue rate
            rescue_value = 0.0
            for p in self.patients:
                if p.rescued:
                    severity_weight = {1: 1.0, 2: 2.0, 3: 3.0}[p.severity]
                    speed_bonus = max(0, 1 - p.time_waiting / 20)
                    rescue_value += severity_weight * (0.7 + 0.3 * speed_bonus)
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
            "rescued": self._rescued_count,
            "dead": self._dead_count,
            "total_patients": len(self.patients),
            "avg_response_time": round(avg_response, 2),
            "steps_taken": self.step_count,
        }

    def final_score(self) -> float:
        """Convenience: returns the 0.0–1.0 episode score."""
        return self._build_info()["score"]