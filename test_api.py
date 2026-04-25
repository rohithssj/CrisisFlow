import requests

def test_api():
    url = "http://localhost:8000/decision"
    
    # 1. Valid request (expected to fail with 500 due to inference.py crash)
    print("Testing Valid Request (Expecting 500)...")
    payload = {"severity": 5, "wait_time": 10.0, "distance": 2.0}
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Invalid request (severity=0)
    print("\nTesting Invalid Request (severity=0)...")
    payload = {"severity": 0, "wait_time": 10.0, "distance": 2.0}
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Invalid request (negative distance)
    print("\nTesting Invalid Request (negative distance)...")
    payload = {"severity": 5, "wait_time": 10.0, "distance": -2.0}
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
