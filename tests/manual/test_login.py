#!/usr/bin/env python
"""Test login API"""
import requests

# Test login
url = "http://localhost:8000/token"
data = {"username": "user1", "password": "Pass@123"}

print("Testing login...")
print(f"URL: {url}")
print(f"Data: {data}")

try:
    response = requests.post(url, data=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        token = response.json().get("access_token")
        print(f"\n✅ Login SUCCESS!")
        print(f"Token: {token[:50]}..." if token else "No token")
        
        # Test getting current user
        headers = {"Authorization": f"Bearer {token}"}
        user_response = requests.get("http://localhost:8000/users/me", headers=headers)
        print(f"\nUser Info Status: {user_response.status_code}")
        print(f"User Info: {user_response.json()}")
    else:
        print(f"\n❌ Login FAILED!")
except Exception as e:
    print(f"\n❌ Error: {e}")
