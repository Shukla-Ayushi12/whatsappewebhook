# test_api.py
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# --- Config ---
API_KEY = os.getenv("WHATSPREP_API_KEY")
BASE_URL = "https://latam.whatsprep.com/api"

headers = {
    "jwt-token": API_KEY,
    "Content-Type": "application/json"
}

# ─────────────────────────────────────────
# TEST 1 — Create a new student
# ─────────────────────────────────────────
print("=" * 40)
print("TEST 1: Create Student")
print("=" * 40)

create_payload = {
    "name": "Ashu",
    "mobile": "+6593822595",
    "level": "Primary 4",
    "gender": "male"
}

print(f"📤 Sending to: {BASE_URL}/create-student")
print(f"📦 Payload: {create_payload}")
print("-" * 40)

try:
    response = requests.post(
        f"{BASE_URL}/create-student",
        headers=headers,
        json=create_payload,
        timeout=10
    )
    print(f"📡 Status: {response.status_code}")
    print(f"📡 Response: {response.text}")
except Exception as e:
    print(f"⚠️  Error: {e}")

# ─────────────────────────────────────────
# TEST 2 — Get student by phone number
# ─────────────────────────────────────────
print("\n" + "=" * 40)
print("TEST 2: Get Student by Phone")
print("=" * 40)

lookup_payload = {
    "mobile": "+6593822595"
}

print(f"📤 Sending to: {BASE_URL}/user-details-by-contact")
print(f"📦 Payload: {lookup_payload}")
print("-" * 40)

try:
    response = requests.post(
        f"{BASE_URL}/user-details-by-contact",
        headers=headers,
        json=lookup_payload,
        timeout=10
    )
    print(f"📡 Status: {response.status_code}")
    print(f"📡 Response: {response.text}")
except Exception as e:
    print(f"⚠️  Error: {e}")
