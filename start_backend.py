#!/usr/bin/env python3
"""
Start backend server with proper database configuration
"""

import os
import sys
import subprocess

def start_backend():
    """Start the backend server with proper configuration"""
    
    # Set database URL (adjust password if needed)
    os.environ['DATABASE_URL'] = 'postgresql+psycopg2://postgres:postgres@localhost:5433/clalit'
    
    print("🚀 Starting Clalit NLP Backend Server...")
    print(f"📊 Database URL: {os.environ['DATABASE_URL']}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'backend.app.main:app', 
            '--reload', 
            '--host', '0.0.0.0', 
            '--port', '8000'
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_backend() 