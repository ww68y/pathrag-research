#!/usr/bin/env python
"""Install required dependencies"""
import subprocess
import sys

packages = [
    "PyPDF2",
    "python-docx", 
    "docx2txt",
    "python-pptx",
    "openpyxl",
    "striprtf",
    "odfpy",
    "ebooklib",
    "beautifulsoup4"
]

print("Installing dependencies...")
for pkg in packages:
    print(f"  Installing {pkg}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True)
    if result.returncode == 0:
        print(f"    OK")
    else:
        print(f"    FAILED: {result.stderr.decode()}")

print("\nDone!")
