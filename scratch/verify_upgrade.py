import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_upgraded_api():
    payload = {
        "type": "fire",
        "severity": 9,
        "wait_time": 5,
        "distance": 3,
        "location": "Hyderabad"
    }
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        if response.status_code == 200:
            print("SUCCESS:")
            print(json.dumps(response.json(), indent=2))
            
            data = response.json()
            # Verify expectations
            if data["unit"] == "Fire Brigade":
                print("[OK] Unit matches 'Fire Brigade'")
            else:
                print(f"[FAIL] Unit is {data['unit']}, expected 'Fire Brigade'")
                
            if data["risk"] == "Critical":
                print("[OK] Risk matches 'Critical'")
            else:
                print(f"[FAIL] Risk is {data['risk']}, expected 'Critical'")
                
            if data["priority"] == "P1":
                print("[OK] Priority matches 'P1'")
            else:
                print(f"[FAIL] Priority is {data['priority']}, expected 'P1'")
                
            if "Hyderabad" in data["reason"]:
                print("[OK] Reason includes location 'Hyderabad'")
            else:
                print("[FAIL] Reason does not include location")
        else:
            print(f"FAILED: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_upgraded_api()
