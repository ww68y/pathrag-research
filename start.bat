@echo off
REM Start script for PathRAG application (both API and UI)

echo Starting PathRAG Application...

REM Check if Python virtual environment exists
if not exist .venv (
    echo Python virtual environment not found. Creating one...
    python -m venv .venv
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating Python virtual environment...
call .venv\Scripts\activate.bat

REM Install backend dependencies
echo Installing backend dependencies...
pip install -r requirements.txt
echo Backend dependencies installed.

REM Install frontend dependencies
echo Installing frontend dependencies...
cd ui
call npm install
echo Frontend dependencies installed.
cd ..

REM Start backend in background
echo Starting backend API on port 8000...
start cmd /k "call venv\Scripts\activate.bat && python main.py"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

REM Start frontend
echo Starting frontend UI on port 3000...
cd ui
REM Set environment variables for frontend
set PORT=3000
set REACT_APP_API_URL=http://localhost:8000
REM Use cross-env to ensure PORT is set correctly
start cmd /k "npx cross-env PORT=3000 npm start"

echo Both services are running. Close the terminal windows to stop.
