#!/usr/bin/env python3
"""
Quick Start Script for Scientific Agent System

Simply run: python run.py

This will:
1. Check if dependencies are installed
2. Start the web server
3. Open your browser (optional)
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'fastapi',
        'uvicorn',
        'google.generativeai',
        'networkx',
        'pydantic'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('.', '_').split('.')[0])
        except ImportError:
            missing.append(pkg)
    
    return missing

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'
    ])
    print("✓ Dependencies installed!")

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           🔬 Scientific Agent System - Quick Start            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        response = input("Install dependencies now? [Y/n]: ").strip().lower()
        if response != 'n':
            install_dependencies()
        else:
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)
    
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not set in environment.")
        print("You can set it in the web UI after starting the server.")
        print("Or set it now: export GEMINI_API_KEY='your_key_here'\n")
    
    # Start server
    print("\n🚀 Starting server...")
    print("=" * 50)
    print("📍 Open your browser to: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("=" * 50)
    print("\nPress Ctrl+C to stop the server.\n")
    
    # Run the server
    import uvicorn
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from api.server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
