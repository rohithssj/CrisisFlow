from pydantic import BaseModel
from typing import List, Dict, Any

class Observation(BaseModel):
    patients: List[Dict[str, Any]]
    ambulances: List[Dict[str, Any]]
    hospitals: List[Dict[str, Any]]
    time_step: int

class Action(BaseModel):
    assignments: List[Dict[str, Any]]

class Reward(BaseModel):
    score: float
    details: Dict[str, float]
