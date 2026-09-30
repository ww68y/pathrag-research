#!/usr/bin/env python
"""Simple script to start PathRAG API server"""
import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pathrag")

# Load environment variables
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")

# Import and create app
try:
    from models.database import create_tables, User, SessionLocal
    from api.auth.jwt_handler import get_password_hash
    
    # Create tables
    create_tables()
    logger.info("Database tables created/verified")
    
    # Create default users if needed
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            default_users = [
                {"username": "user1", "email": "user1@example.com", "password": "Pass@123"},
                {"username": "user2", "email": "user2@example.com", "password": "Pass@123"},
                {"username": "user3", "email": "user3@example.com", "password": "Pass@123"}
            ]
            for user_data in default_users:
                hashed_password = get_password_hash(user_data["password"])
                user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    hashed_password=hashed_password
                )
                db.add(user)
            db.commit()
            logger.info("Default users created")
        else:
            users = db.query(User).all()
            logger.info(f"Database has {len(users)} users: {[u.username for u in users]}")
    finally:
        db.close()
    
    # Import and run app
    import uvicorn
    from main import app
    
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
    
except Exception as e:
    logger.error(f"Failed to start server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
