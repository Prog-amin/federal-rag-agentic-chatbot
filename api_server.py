from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import uvicorn
from typing import Dict, Any, Optional
import os
import aiosqlite
from database import db_manager
from agent import federal_agent
from data_pipeline import DataPipeline

# FastAPI app initialization
app = FastAPI(
    title="Federal Registry RAG Agent",
    description="AI Agent for querying US Federal Registry documents",
    version="1.0.0"
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Static files (for CSS/JS if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    status: str
    response: str
    timestamp: float

class PipelineRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PipelineResponse(BaseModel):
    status: str
    message: str

# Global state
pipeline_running = False

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    await db_manager.initialize_database()
    print("Database initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Application shutting down")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """Chat endpoint for interacting with the federal registry agent"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Get response from agent
        response = await federal_agent.get_response(request.message)
        
        return ChatResponse(
            status=response["status"],
            response=response["response"],
            timestamp=response["timestamp"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/pipeline/run", response_model=PipelineResponse)
async def run_data_pipeline(request: Optional[PipelineRequest] = None) -> PipelineResponse:
    """Run the data pipeline to update federal documents"""
    global pipeline_running
    
    if pipeline_running:
        return PipelineResponse(
            status="error",
            message="Pipeline is already running. Please wait for it to complete."
        )
    
    try:
        pipeline_running = True
        
        pipeline = DataPipeline()
        
        # Run pipeline with provided dates or defaults
        start_date = request.start_date if request else None
        end_date = request.end_date if request else None
        
        success = await pipeline.run_pipeline(start_date, end_date)
        
        if success:
            return PipelineResponse(
                status="success",
                message="Data pipeline completed successfully. Federal documents have been updated."
            )
        else:
            return PipelineResponse(
                status="error",
                message="Data pipeline failed. Check logs for more details."
            )
    
    except Exception as e:
        return PipelineResponse(
            status="error",
            message=f"Pipeline error: {str(e)}"
        )
    
    finally:
        pipeline_running = False

@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status"""
    return {
        "running": pipeline_running,
        "message": "Pipeline is currently running" if pipeline_running else "Pipeline is idle"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        health_ok = await db_manager.health_check()
        
        if health_ok:
            return {
                "status": "healthy",
                "database": "connected",
                "agent": "ready"
            }
        else:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "agent": "not ready"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        # Get document count
        doc_count = await db_manager.get_document_count()
        
        # Get latest documents to determine date range
        latest_docs = await db_manager.get_latest_documents(5)
        latest_date = latest_docs[0]['publication_date'] if latest_docs else None
        
        # Get recent documents to count agencies
        recent_docs = await db_manager.get_recent_documents(days=30, limit=1000)
        agencies = set()
        for doc in recent_docs:
            agency = doc.get('agency', '').strip()
            if agency:
                agencies.add(agency)
        
        # Get recent pipeline runs using direct SQL
        async with aiosqlite.connect(db_manager.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT status, COUNT(*) as count 
                FROM pipeline_logs 
                WHERE run_date >= date('now', '-7 days')
                GROUP BY status
            """) as cursor:
                pipeline_stats_rows = await cursor.fetchall()
                pipeline_stats = {row['status']: row['count'] for row in pipeline_stats_rows}
        
        return {
            "total_documents": doc_count,
            "latest_document_date": latest_date,
            "unique_agencies": len(agencies),
            "recent_pipeline_runs": pipeline_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/search")
async def search_documents(q: str, limit: int = 10):
    """Search documents endpoint"""
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        results = await db_manager.search_documents(q, limit)
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/api/documents/recent")
async def get_recent_documents(days: int = 7, limit: int = 20):
    """Get recent documents endpoint"""
    try:
        documents = await db_manager.get_recent_documents(days, limit)
        return {
            "documents": documents,
            "count": len(documents),
            "days": days
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent documents: {str(e)}")

@app.get("/api/documents/by-agency")
async def get_documents_by_agency(agency: str, limit: int = 20):
    """Get documents by agency endpoint"""
    try:
        if not agency.strip():
            raise HTTPException(status_code=400, detail="Agency name cannot be empty")
        
        documents = await db_manager.get_documents_by_agency(agency, limit)
        return {
            "agency": agency,
            "documents": documents,
            "count": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting documents by agency: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )