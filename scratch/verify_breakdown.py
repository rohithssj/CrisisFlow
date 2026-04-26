import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_decision_breakdown(incident_type, severity, location="Sector 7-G"):
    print(f"\nTesting type: {incident_type}, severity: {severity}, location: {location}")
    payload = {
        "type": incident_type,
        "severity": severity,
        "wait_time": 10.0,
        "distance": 15.0,
        "location": location
    }
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("SUCCESS:")
            print(f"  Explanation: {data['explanation']}")
            print(f"  Factors: {data['factors']}")
            
            # Verify required keys
            required = ["unit", "risk", "score", "confidence", "priority", "reason", "factors", "explanation"]
            missing = [k for k in required if k not in data]
            if not missing:
                print("  [OK] All required keys (including breakdown) present")
            else:
                print(f"  [ERROR] Missing keys: {missing}")
        else:
            print("FAILED:", response.status_code, response.text)
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    test_decision_breakdown("cyber", 9, "Server Farm 01")
    test_decision_breakdown("fire", 4, "Residential Block B")
