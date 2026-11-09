#!/usr/bin/env python
"""Optional smoke check for a locally configured PathRAG account.

Set PATHRAG_TEST_USERNAME and PATHRAG_TEST_PASSWORD in the shell before use.
This script deliberately does not print credentials or access tokens.
"""
import os

import requests

base_url = os.getenv("PATHRAG_API_URL", "http://localhost:8000").rstrip("/")
username = os.getenv("PATHRAG_TEST_USERNAME")
password = os.getenv("PATHRAG_TEST_PASSWORD")

if not username or not password:
    raise SystemExit(
        "Set PATHRAG_TEST_USERNAME and PATHRAG_TEST_PASSWORD before running this check."
    )

response = requests.post(
    f"{base_url}/token",
    data={"username": username, "password": password},
    timeout=15,
)
if response.status_code != 200:
    raise SystemExit(f"Login failed with HTTP {response.status_code}.")

token = response.json().get("access_token")
if not token:
    raise SystemExit("The API response did not contain an access token.")

user_response = requests.get(
    f"{base_url}/users/me",
    headers={"Authorization": f"Bearer {token}"},
    timeout=15,
)
if user_response.status_code != 200:
    raise SystemExit(f"Authenticated request failed with HTTP {user_response.status_code}.")

print("Login and authenticated-user smoke check passed.")
