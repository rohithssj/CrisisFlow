import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_decision(incident_type, severity):
    print(f"\nTesting type: {incident_type}, severity: {severity}")
    payload = {
        "type": incident_type,
        "severity": severity,
        "wait_time": 5.0,
        "distance": 10.0
    }
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        if response.status_code == 200:
            print("SUCCESS:", json.dumps(response.json(), indent=2))
        else:
            print("FAILED:", response.status_code, response.text)
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    # Test cases
    test_decision("cyber", 8)
    test_decision("fire", 9)
    test_decision("flood", 7)
    test_decision("medical", 5)
    test_decision("unknown_type", 6) # Should default to medical
