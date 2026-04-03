import logging

def setup_logger(level=logging.INFO):
    """
    Configures basic logging for CrisisFlow.
    """
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger("CrisisFlow")

def save_metrics(metrics, path):
    """
    Utility for persisting training metrics.
    """
    # implementation details
    pass
