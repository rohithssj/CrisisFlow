"""
Configuration constants for the CrisisFlow environment.
Each config level adjusts the difficulty of the emergency response simulation.
"""

DIFFICULTY_CONFIGS = {
    # Easy: More resources, fewer crises, slower decay
    "easy": {
        "num_ambulances": 12,
        "num_hospitals": 4,
        "hospital_capacity": 25,
        "num_crises": 15,
        "max_steps": 500,
        "ambulance_speed": 0.06,
        "severity_weights": [0.7, 0.2, 0.1],  # Mostly minor
    },
    # Medium: Balanced resources and crises
    "medium": {
        "num_ambulances": 8,
        "num_hospitals": 3,
        "hospital_capacity": 15,
        "num_crises": 35,
        "max_steps": 800,
        "ambulance_speed": 0.04,
        "severity_weights": [0.4, 0.4, 0.2],
    },
    # Hard: Limited resources, high frequency critical crises
    "hard": {
        "num_ambulances": 3,
        "num_hospitals": 2,
        "hospital_capacity": 6,
        "num_crises": 72,
        "max_steps": 1000,
        "ambulance_speed": 0.03,
        "severity_weights": [0.1, 0.3, 0.6],  # 60% critical
    },
}
