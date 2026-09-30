#!/usr/bin/env python
"""Test script to check if main app can be imported"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from main import app
    print("SUCCESS: App imported OK")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
