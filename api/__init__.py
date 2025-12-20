"""
Scientific Agent System - API Package

This package provides the FastAPI backend:
- REST endpoints for pipeline management
- WebSocket for real-time updates
- Static file serving
"""

from api.server import app, run_server

__all__ = ["app", "run_server"]
