#!/usr/bin/env python3
"""
Main FastAPI app for Hugging Face Spaces deployment
RAG Agentic System for Federal Registry Documents
"""
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import uvicorn
from typing import Dict, Any, Optional
import os
import logging

from database import db_manager
from agent import FederalRegistryAgent
from data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
agent: Optional[FederalRegistryAgent] = None
pipeline: Optional[DataPipeline] = None
pipeline_running = False

async def run_initial_pipeline():
    """Run the initial pipeline to populate the database"""
    global pipeline, pipeline_running
    
    if pipeline is None:
        logger.error("❌ Pipeline not initialized, cannot run initial pipeline")
        return False
    
    try:
        logger.info("🔄 Running initial data pipeline...")
        pipeline_running = True
        
        # Check if database already has documents
        doc_count = await db_manager.get_document_count()
        
        if doc_count > 0:
            logger.info(f"✅ Database already has {doc_count} documents, skipping initial pipeline")
            return True
        
        # Run the pipeline for the first time
        success = await pipeline.run_pipeline()
        
        if success:
            final_count = await db_manager.get_document_count()
            logger.info(f"✅ Initial pipeline completed successfully! Added {final_count} documents")
            return True
        else:
            logger.error("❌ Initial pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running initial pipeline: {e}")
        return False
    finally:
        pipeline_running = False

async def initialize_system():
    """Initialize the system components"""
    global agent, pipeline
    try:
        # Initialize database
        await db_manager.initialize_database()
        logger.info("✅ Database initialized")
        
        # Initialize agent
        agent = FederalRegistryAgent()
        logger.info("✅ Agent initialized")
        
        # Initialize pipeline
        pipeline = DataPipeline()
        logger.info("✅ Pipeline initialized")
        
        # Run initial pipeline to populate database
        await run_initial_pipeline()
        
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing system: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("🚀 Starting up application...")
    success = await initialize_system()
    if success:
        logger.info("✅ System initialized successfully")
    else:
        logger.error("❌ Failed to initialize system")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down application")

# FastAPI app initialization with lifespan
app = FastAPI(
    title="Federal Registry RAG Agent",
    description="AI Agent for querying US Federal Registry documents",
    version="1.0.0",
    lifespan=lifespan
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Static files (for CSS/JS if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """Chat endpoint for interacting with the federal registry agent"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Get response from agent
        response = await agent.get_response(request.message)
        
        return ChatResponse(
            status=response["status"],
            response=response["response"],
            timestamp=response["timestamp"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/pipeline/run", response_model=PipelineResponse)
async def run_data_pipeline(request: PipelineRequest = Body(default=PipelineRequest())) -> PipelineResponse:
    """Run the data pipeline to update federal documents"""
    global pipeline_running
    
    if pipeline_running:
        return PipelineResponse(
            status="error",
            message="Pipeline is already running. Please wait for it to complete."
        )
    
    if pipeline is None:
        return PipelineResponse(
            status="error",
            message="Pipeline not initialized"
        )
    
    try:
        pipeline_running = True
        
        # Run pipeline with provided dates or defaults
        start_date = request.start_date
        end_date = request.end_date
        
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
        count = await db_manager.get_document_count()
        
        return {
            "status": "healthy",
            "database": "connected",
            "agent": "ready" if agent else "not initialized",
            "document_count": count
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
        doc_count = await db_manager.get_document_count()
        latest_docs = await db_manager.get_latest_documents(5)
        
        return {
            "total_documents": doc_count,
            "latest_documents": latest_docs,
            "system_status": "healthy" if agent else "initializing"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info"
    )