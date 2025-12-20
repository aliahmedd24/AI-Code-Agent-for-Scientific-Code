"""
FastAPI Backend - REST API and WebSocket Server

Provides:
- REST endpoints for pipeline management
- WebSocket for real-time progress updates
- File serving for reports and visualizations
- Health checks and status monitoring
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import uvicorn

from core.orchestrator import PipelineOrchestrator, PipelineResult, PipelineEvent
from core.knowledge_graph import get_global_graph


# Pydantic models for API
class PipelineRequest(BaseModel):
    """Request to start a new pipeline run."""
    paper_url: str
    repo_url: str
    auto_fix_errors: bool = True
    use_docker: bool = True


class PipelineStatus(BaseModel):
    """Current pipeline status."""
    run_id: str
    stage: str
    progress: float
    message: str
    started_at: str
    events: List[Dict[str, Any]]


class APIConfig(BaseModel):
    """API configuration."""
    gemini_api_key: str
    gemini_api_keys: Optional[List[str]] = None  # Additional keys for rotation
    github_token: Optional[str] = None


# Global state
class AppState:
    def __init__(self):
        self.config: Optional[APIConfig] = None
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        self.completed_runs: Dict[str, PipelineResult] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)


app_state = AppState()

# Create FastAPI app
app = FastAPI(
    title="Scientific Agent System API",
    description="LLM-driven pipeline for analyzing scientific papers and generating code",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            if websocket in self.active_connections[run_id]:
                self.active_connections[run_id].remove(websocket)
    
    async def broadcast(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            for conn in disconnected:
                self.disconnect(conn, run_id)


manager = ConnectionManager()


# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        with open(ui_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="""
    <html>
    <head><title>Scientific Agent System</title></head>
    <body style="font-family: sans-serif; padding: 40px; background: #1a1a2e; color: #fff;">
        <h1>🔬 Scientific Agent System API</h1>
        <p>UI file not found at expected location.</p>
        <p>API Documentation: <a href="/docs" style="color: #818cf8;">/docs</a></p>
    </body>
    </html>
    """)


@app.post("/api/config")
async def set_config(config: APIConfig):
    """Set API configuration (API keys)."""
    app_state.config = config
    
    # Register all API keys with the key manager
    try:
        from core.api_key_manager import get_key_manager
        key_manager = get_key_manager()
        
        # Add primary key
        key_manager.add_key(config.gemini_api_key)
        
        # Add additional keys if provided
        if config.gemini_api_keys:
            for key in config.gemini_api_keys:
                if key and key.strip():
                    key_manager.add_key(key.strip())
        
        key_count = key_manager.get_available_count()
    except ImportError:
        key_count = 1
    
    return {
        "status": "configured",
        "has_github_token": config.github_token is not None,
        "api_keys_registered": key_count
    }


@app.get("/api/config/status")
async def get_config_status():
    """Check if API is configured."""
    key_stats = None
    try:
        from core.api_key_manager import get_key_manager
        key_manager = get_key_manager()
        key_stats = key_manager.get_stats()
    except:
        pass
    
    return {
        "configured": app_state.config is not None,
        "has_github_token": app_state.config.github_token is not None if app_state.config else False,
        "api_keys": key_stats
    }


@app.post("/api/pipeline/start")
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start a new pipeline run."""
    if not app_state.config:
        raise HTTPException(status_code=400, detail="API not configured. Set config first.")
    
    # Generate run ID
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize run state
    app_state.active_runs[run_id] = {
        "id": run_id,
        "status": "starting",
        "stage": "initialized",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "request": request.dict(),
        "events": []
    }
    
    # Start pipeline in background
    background_tasks.add_task(
        run_pipeline_task,
        run_id,
        request.paper_url,
        request.repo_url,
        request.auto_fix_errors,
        request.use_docker
    )
    
    return {"run_id": run_id, "status": "started"}


async def run_pipeline_task(
    run_id: str,
    paper_url: str,
    repo_url: str,
    auto_fix_errors: bool,
    use_docker: bool
):
    """Background task to run the pipeline."""
    try:
        orchestrator = PipelineOrchestrator(
            gemini_api_key=app_state.config.gemini_api_key,
            github_token=app_state.config.github_token,
            output_dir=str(app_state.output_dir / run_id),
            use_docker=use_docker
        )
        
        # Progress callback
        async def on_event(event: PipelineEvent):
            progress_map = {
                "initialized": 0,
                "parsing_paper": 10,
                "analyzing_repo": 30,
                "mapping_concepts": 50,
                "generating_code": 60,
                "setting_up_environment": 70,
                "executing_code": 80,
                "generating_report": 90,
                "completed": 100,
                "failed": 100
            }
            
            app_state.active_runs[run_id]["stage"] = event.stage.value
            app_state.active_runs[run_id]["progress"] = progress_map.get(event.stage.value, 0)
            app_state.active_runs[run_id]["events"].append({
                "stage": event.stage.value,
                "message": event.message,
                "timestamp": event.timestamp,
                "is_error": event.is_error,
                "data": event.data
            })
            
            # Broadcast to WebSocket clients
            await manager.broadcast(run_id, {
                "type": "progress",
                "run_id": run_id,
                "stage": event.stage.value,
                "progress": progress_map.get(event.stage.value, 0),
                "message": event.message,
                "is_error": event.is_error,
                "data": event.data
            })
        
        orchestrator.add_callback(on_event)
        
        # Run pipeline
        result = await orchestrator.run_pipeline(
            paper_url=paper_url,
            repo_url=repo_url,
            auto_fix_errors=auto_fix_errors
        )
        
        # Store result
        app_state.completed_runs[run_id] = result
        app_state.active_runs[run_id]["status"] = "completed" if result.success else "failed"
        app_state.active_runs[run_id]["result"] = {
            "success": result.success,
            "report_path": result.report_path,
            "total_time": result.total_time,
            "error": result.error
        }
        
        # Final broadcast
        await manager.broadcast(run_id, {
            "type": "completed",
            "run_id": run_id,
            "success": result.success,
            "report_path": result.report_path,
            "total_time": result.total_time,
            "error": result.error
        })
        
    except Exception as e:
        app_state.active_runs[run_id]["status"] = "failed"
        app_state.active_runs[run_id]["error"] = str(e)
        
        await manager.broadcast(run_id, {
            "type": "error",
            "run_id": run_id,
            "error": str(e)
        })


@app.get("/api/pipeline/{run_id}/status")
async def get_pipeline_status(run_id: str):
    """Get status of a pipeline run."""
    if run_id not in app_state.active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return app_state.active_runs[run_id]


@app.get("/api/pipeline/{run_id}/result")
async def get_pipeline_result(run_id: str):
    """Get full result of a completed pipeline run."""
    if run_id not in app_state.completed_runs:
        if run_id in app_state.active_runs:
            return {"status": "running", "run": app_state.active_runs[run_id]}
        raise HTTPException(status_code=404, detail="Run not found")
    
    result = app_state.completed_runs[run_id]
    
    return {
        "success": result.success,
        "paper_info": result.paper_info,
        "repo_info": result.repo_info,
        "concept_mappings": result.concept_mappings,
        "generated_code": result.generated_code,
        "execution_results": result.execution_results,
        "visualizations": [
            {"filename": v["filename"], "format": v["format"]}
            for v in result.visualizations
        ],
        "report_path": result.report_path,
        "total_time": result.total_time,
        "error": result.error
    }


@app.get("/api/pipeline/{run_id}/report")
async def get_report(run_id: str):
    """Get the generated HTML report."""
    if run_id not in app_state.completed_runs:
        raise HTTPException(status_code=404, detail="Run not found or not completed")
    
    result = app_state.completed_runs[run_id]
    if result.report_path and os.path.exists(result.report_path):
        return FileResponse(result.report_path, media_type="text/html")
    
    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/api/pipeline/{run_id}/knowledge-graph")
async def get_knowledge_graph(run_id: str):
    """Get the knowledge graph data."""
    if run_id not in app_state.completed_runs:
        raise HTTPException(status_code=404, detail="Run not found or not completed")
    
    result = app_state.completed_runs[run_id]
    return result.knowledge_graph


@app.get("/api/pipeline/{run_id}/visualization/{filename}")
async def get_visualization(run_id: str, filename: str):
    """Get a specific visualization image."""
    if run_id not in app_state.completed_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    result = app_state.completed_runs[run_id]
    for viz in result.visualizations:
        if viz["filename"] == filename:
            import base64
            from fastapi.responses import Response
            
            data = base64.b64decode(viz["data"])
            media_type = f"image/{viz['format']}"
            return Response(content=data, media_type=media_type)
    
    raise HTTPException(status_code=404, detail="Visualization not found")


@app.get("/api/pipeline/{run_id}/code/{filename}")
async def get_code_file(run_id: str, filename: str):
    """Get a generated code file."""
    if run_id not in app_state.completed_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    result = app_state.completed_runs[run_id]
    for code in result.generated_code:
        if code["filename"] == filename:
            return {
                "filename": code["filename"],
                "language": code["language"],
                "content": code["content"],
                "purpose": code["purpose"]
            }
    
    raise HTTPException(status_code=404, detail="Code file not found")


@app.get("/api/runs")
async def list_runs():
    """List all pipeline runs."""
    runs = []
    
    for run_id, run_data in app_state.active_runs.items():
        runs.append({
            "id": run_id,
            "status": run_data["status"],
            "stage": run_data["stage"],
            "progress": run_data["progress"],
            "started_at": run_data["started_at"],
            "paper_url": run_data["request"]["paper_url"],
            "repo_url": run_data["request"]["repo_url"]
        })
    
    return {"runs": sorted(runs, key=lambda x: x["started_at"], reverse=True)}


@app.delete("/api/pipeline/{run_id}")
async def delete_run(run_id: str):
    """Delete a pipeline run and its data."""
    if run_id in app_state.active_runs:
        del app_state.active_runs[run_id]
    
    if run_id in app_state.completed_runs:
        del app_state.completed_runs[run_id]
    
    # Clean up output directory
    run_dir = app_state.output_dir / run_id
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
    
    return {"status": "deleted", "run_id": run_id}


# WebSocket endpoint
@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket for real-time pipeline updates."""
    await manager.connect(websocket, run_id)
    
    try:
        # Send current state if run exists
        if run_id in app_state.active_runs:
            await websocket.send_json({
                "type": "state",
                "run_id": run_id,
                "data": app_state.active_runs[run_id]
            })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    
    finally:
        manager.disconnect(websocket, run_id)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_runs": len(app_state.active_runs),
        "completed_runs": len(app_state.completed_runs)
    }


# Mount static files for UI
def mount_static_files():
    """Mount static file directories."""
    ui_dir = Path(__file__).parent.parent / "ui"
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")
    
    outputs_dir = app_state.output_dir
    if outputs_dir.exists():
        app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    mount_static_files()
    print("🚀 Scientific Agent System API started")
    print("📚 Documentation available at /docs")


# Main entry point
def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
