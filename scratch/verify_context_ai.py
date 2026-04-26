import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_decision(incident_type, severity, location="Sector 7-G"):
    print(f"\nTesting type: {incident_type}, severity: {severity}, location: {location}")
    payload = {
        "type": incident_type,
        "severity": severity,
        "wait_time": 5.0,
        "distance": 10.0,
        "location": location
    }
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("SUCCESS:")
            print(f"  Reason: {data['reason']}")
            print(f"  Confidence: {data['confidence']}%")
            print(f"  Priority: {data['priority']}")
            print(f"  Risk: {data['risk']}")
            print(f"  Unit: {data['unit']}")
            
            # Check if location is in reason
            if location in data['reason']:
                print("  [OK] Location present in response")
            else:
                print("  [WARNING] Location NOT present in response")
                
            # Verify required keys
            required = ["unit", "risk", "score", "confidence", "priority", "reason"]
            missing = [k for k in required if k not in data]
            if not missing:
                print("  [OK] All required keys present")
            else:
                print(f"  [ERROR] Missing keys: {missing}")
        else:
            print("FAILED:", response.status_code, response.text)
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    print("Ensure CrisisFlow API is running on http://127.0.0.1:8000")
    # Test cases
    test_decision("cyber", 8, "Core Node 1")
    test_decision("fire", 9, "Sector 4 Industrial")
    test_decision("flood", 7, "Coastal Zone A")
    test_decision("medical", 5, "Downtown Plaza")
