from grader import grade

metrics = {
    "survival_rate": 0.9,
    "avg_response_time": 2.0,
    "deaths": 1
}

score = grade(metrics)
print("Score:", score)