def grade(metrics: dict) -> float:
    survival = metrics.get("survival_rate", 0)  # already normalized
    response = metrics.get("avg_response_time", 1)
    deaths = metrics.get("deaths", 0)

    score = survival - (0.1 * deaths) - (0.05 * response)

    # clamp between 0 and 1
    return max(0.0, min(1.0, round(score, 4)))