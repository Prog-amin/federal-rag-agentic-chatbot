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
from datetime import datetime, timedelta
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

async def comprehensive_pipeline_check():
    """Comprehensive pipeline check and initialization - integrates all run_pipeline.py functions"""
    global pipeline, pipeline_running
    
    try:
        logger.info("🔍 Starting comprehensive pipeline check...")
        
        # 1. Check data status first (from check_data_status_safe)
        logger.info("📊 Checking current data status...")
        recent_docs = await db_manager.get_recent_documents(days=365, limit=1000)
        total_recent = len(recent_docs)
        
        if total_recent > 0:
            # Analyze existing data - filter out None values properly
            valid_dates = []
            for doc in recent_docs:
                pub_date = doc.get('publication_date')
                if pub_date is not None and isinstance(pub_date, str) and pub_date.strip():
                    valid_dates.append(pub_date)
            
            if valid_dates:
                latest_date = max(valid_dates)
                oldest_date = min(valid_dates)
                logger.info(f"📈 Found {total_recent} documents (Range: {oldest_date} to {latest_date})")
                
                # Check if data is recent (within last 7 days)
                try:
                    latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                    days_old = (datetime.now() - latest_dt).days
                    
                    if days_old > 7:
                        logger.info(f"⚠️ Latest data is {days_old} days old, running update pipeline...")
                        await run_update_pipeline()
                    else:
                        logger.info("✅ Data is current, no pipeline update needed")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Could not parse latest date '{latest_date}': {e}")
                    logger.info("🔄 Running update pipeline due to date parsing issue...")
                    await run_update_pipeline()
            else:
                logger.info("⚠️ Documents found but no valid dates, running full pipeline...")
                await run_full_pipeline()
        else:
            logger.info("📥 No documents found, running initial pipeline...")
            await run_full_pipeline()
        
        # 2. Check pipeline logs (from check_pipeline_logs)
        await check_recent_pipeline_logs()
        
        # 3. Final status check
        await display_final_status()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive pipeline check failed: {e}")
        return False

async def run_full_pipeline():
    """Run full pipeline (equivalent to run_pipeline from run_pipeline.py)"""
    global pipeline_running
    
    if pipeline is None:
        logger.error("❌ Pipeline not initialized")
        return False
        
    try:
        pipeline_running = True
        logger.info("🔄 Starting full data pipeline...")
        logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = await pipeline.run_pipeline()
        
        if success:
            logger.info("✅ Full pipeline completed successfully!")
            logger.info(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        else:
            logger.error("❌ Full pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Full pipeline error: {e}")
        return False
    finally:
        pipeline_running = False

async def run_update_pipeline():
    """Run pipeline for recent data updates"""
    global pipeline_running
    
    if pipeline is None:
        return False
        
    try:
        pipeline_running = True
        # Get data for last 7 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"🔄 Running update pipeline for {start_date} to {end_date}")
        success = await pipeline.run_pipeline(start_date, end_date)
        
        if success:
            logger.info("✅ Update pipeline completed successfully!")
            return True
        else:
            logger.error("❌ Update pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Update pipeline error: {e}")
        return False
    finally:
        pipeline_running = False

async def check_recent_pipeline_logs():
    """Check recent pipeline logs (from check_pipeline_logs)"""
    try:
        logger.info("📋 Checking recent pipeline logs...")
        
        # Get recent documents to simulate log checking
        recent_docs = await db_manager.get_recent_documents(days=1, limit=5)
        if recent_docs:
            logger.info(f"✅ Recent activity: {len(recent_docs)} documents processed recently")
        else:
            logger.info("ℹ️ No recent pipeline activity found")
            
    except Exception as e:
        logger.error(f"❌ Error checking pipeline logs: {e}")

async def display_final_status():
    """Display final system status (combines multiple status checks)"""
    try:
        logger.info("📊 Final system status:")
        
        # Get comprehensive stats
        doc_count = await db_manager.get_document_count()
        recent_docs = await db_manager.get_recent_documents(days=7, limit=10)
        
        logger.info(f"📈 Total documents: {doc_count}")
        logger.info(f"📅 Recent documents (7 days): {len(recent_docs)}")
        
        if recent_docs:
            # Agency distribution
            agencies = {}
            for doc in recent_docs:
                agency = doc.get('agency', 'Unknown')
                if agency and agency.strip():
                    agencies[agency] = agencies.get(agency, 0) + 1
                    
            if agencies:
                logger.info("🏛️ Recent agency activity:")
                sorted_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:3]
                for agency, count in sorted_agencies:
                    logger.info(f"   • {agency}: {count} documents")
        
        logger.info("✅ System ready for queries!")
        
    except Exception as e:
        logger.error(f"❌ Error displaying final status: {e}")

async def initialize_system():
    """Initialize the system components with comprehensive pipeline integration"""
    global agent, pipeline
    
    try:
        # Initialize database
        await db_manager.initialize_database()
        logger.info("✅ Database initialized")
        
        # Initialize pipeline
        pipeline = DataPipeline()
        logger.info("✅ Pipeline initialized")
        
        # Run comprehensive pipeline check (integrates all run_pipeline.py functions)
        pipeline_success = await comprehensive_pipeline_check()
        
        # Initialize agent after data is ready
        agent = FederalRegistryAgent()
        logger.info("✅ Agent initialized")
        
        if pipeline_success:
            logger.info("🎉 System initialization completed successfully!")
            logger.info("💡 RAG system is ready with comprehensive data coverage")
        else:
            logger.warning("⚠️ System initialized but pipeline had issues")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing system: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("🚀 Starting up RAG Federal Registry System...")
    success = await initialize_system()
    if success:
        logger.info("✅ RAG System initialized successfully")
    else:
        logger.error("❌ Failed to initialize RAG system")
    yield
    # Shutdown
    logger.info("🔄 Shutting down application")

# FastAPI app initialization with lifespan
app = FastAPI(
    title="Federal Registry RAG Agent",
    description="AI Agent for querying US Federal Registry documents with comprehensive data coverage",
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
    """Run comprehensive data pipeline with date range support"""
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
        # Run comprehensive pipeline based on parameters
        if request.start_date and request.end_date:
            # Specific date range
            pipeline_running = True
            success = await pipeline.run_pipeline(request.start_date, request.end_date)
            message = f"Pipeline completed for date range {request.start_date} to {request.end_date}"
        else:
            # Comprehensive check and update
            success = await comprehensive_pipeline_check()
            message = "Comprehensive pipeline check and update completed"
        
        if success:
            # Get final stats
            doc_count = await db_manager.get_document_count()
            return PipelineResponse(
                status="success",
                message=f"{message}. Total documents: {doc_count}"
            )
        else:
            return PipelineResponse(
                status="error",
                message="Pipeline failed. Check logs for more details."
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
    """Get current pipeline status with detailed information"""
    try:
        doc_count = await db_manager.get_document_count()
        recent_docs = await db_manager.get_recent_documents(days=7, limit=5)
        
        return {
            "running": pipeline_running,
            "message": "Pipeline is currently running" if pipeline_running else "Pipeline is idle",
            "total_documents": doc_count,
            "recent_documents": len(recent_docs),
            "last_update": recent_docs[0].get('publication_date') if recent_docs else None
        }
    except Exception as e:
        return {
            "running": pipeline_running,
            "message": f"Status check error: {str(e)}"
        }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Test database connection
        count = await db_manager.get_document_count()
        recent_count = len(await db_manager.get_recent_documents(days=7, limit=100))
        
        return {
            "status": "healthy",
            "database": "connected",
            "agent": "ready" if agent else "not initialized",
            "pipeline": "ready" if pipeline else "not initialized",
            "document_count": count,
            "recent_documents": recent_count,
            "data_freshness": "current" if recent_count > 0 else "needs_update"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/stats")
async def get_database_stats():
    """Get comprehensive database statistics"""
    try:
        doc_count = await db_manager.get_document_count()
        latest_docs = await db_manager.get_latest_documents(5)
        recent_docs = await db_manager.get_recent_documents(days=30, limit=100)
        
        # Agency distribution from recent docs
        agencies = {}
        for doc in recent_docs:
            agency = doc.get('agency', 'Unknown')
            if agency and agency.strip():
                agencies[agency] = agencies.get(agency, 0) + 1
        
        top_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_documents": doc_count,
            "latest_documents": latest_docs,
            "recent_activity": {
                "last_30_days": len(recent_docs),
                "top_agencies": [{"agency": k, "count": v} for k, v in top_agencies]
            },
            "system_status": "healthy" if agent and pipeline else "initializing"
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