# Placeholder for scenario-specific task implementations
class TaskScenario:
    """
    Template for specific crisis management scenarios.
    """
    def __init__(self, name):
        self.name = name
        self.initial_conditions = {}
        
    def reset(self):
        # Reset local task state
        pass
