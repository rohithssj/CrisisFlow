import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_valid_decision():
    print("\nTesting Valid Decision...")
    payload = {
        "severity": 3,
        "wait_time": 5.5,
        "distance": 10.2
    }
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_invalid_decision():
    print("\nTesting Invalid Decision (Validation Error)...")
    payload = {
        "severity": -1, # Should trigger validation error
        "wait_time": 5.5,
        "distance": 10.2
    }
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_missing_fields():
    print("\nTesting Missing Fields...")
    payload = {
        "severity": 3
    }
    try:
        response = requests.post(f"{BASE_URL}/decision", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Note: Ensure the API server is running before executing this script.
    test_root()
    test_valid_decision()
    test_invalid_decision()
    test_missing_fields()
