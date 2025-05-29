#!/usr/bin/env python3
"""
Main FastAPI app for Hugging Face Spaces deployment
RAG Agentic System for Federal Registry Documents
Enhanced version with comprehensive features from both Streamlit and FastAPI implementations
"""
from fastapi import FastAPI, HTTPException, Request, Body, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import asyncio
import uvicorn
from typing import Dict, Any, Optional, List
import os
import logging
import json
from datetime import datetime, timedelta
from database import db_manager
from agent import FederalRegistryAgent
from data_pipeline import DataPipeline

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User query message")
    include_sources: Optional[bool] = Field(default=True, description="Include source documents in response")

class ChatResponse(BaseModel):
    status: str
    response: str
    timestamp: float
    sources: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None

class PipelineRequest(BaseModel):
    start_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="End date in YYYY-MM-DD format")
    force_full: Optional[bool] = Field(default=False, description="Force full pipeline run")

class PipelineResponse(BaseModel):
    status: str
    message: str
    documents_processed: Optional[int] = None
    duration: Optional[float] = None

class SystemStats(BaseModel):
    total_documents: int
    recent_documents: int
    agency_count: int
    latest_publication_date: Optional[str]
    data_freshness_days: Optional[int]
    top_agencies: List[Dict[str, Any]]

# Global state
agent: Optional[FederalRegistryAgent] = None
pipeline: Optional[DataPipeline] = None
pipeline_running = False
system_initialized = False
initialization_error = None

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
        
        # 2. Check pipeline logs and health
        await check_recent_pipeline_logs()
        
        # 3. Verify database health
        health_status = await db_manager.health_check()
        if not health_status:
            logger.error("❌ Database health check failed")
            return False
        
        # 4. Final status check
        await display_final_status()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive pipeline check failed: {e}")
        return False

async def run_full_pipeline():
    """Run full pipeline with enhanced logging and error handling"""
    global pipeline_running
    
    if pipeline is None:
        logger.error("❌ Pipeline not initialized")
        return False
        
    try:
        pipeline_running = True
        start_time = datetime.now()
        logger.info("🔄 Starting full data pipeline...")
        logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = await pipeline.run_pipeline()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            logger.info("✅ Full pipeline completed successfully!")
            logger.info(f"⏰ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⌛ Duration: {duration:.2f} seconds")
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
    """Run pipeline for recent data updates with enhanced configuration"""
    global pipeline_running
    
    if pipeline is None:
        return False
        
    try:
        pipeline_running = True
        start_time = datetime.now()
        
        # Get data for last 7 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"🔄 Running update pipeline for {start_date} to {end_date}")
        success = await pipeline.run_pipeline(start_date, end_date)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if success:
            logger.info(f"✅ Update pipeline completed successfully in {duration:.2f}s!")
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
    """Enhanced pipeline log checking with more detailed analysis"""
    try:
        logger.info("📋 Checking recent pipeline logs...")
        
        # Get recent documents to simulate log checking
        recent_docs = await db_manager.get_recent_documents(days=1, limit=10)
        if recent_docs:
            logger.info(f"✅ Recent activity: {len(recent_docs)} documents processed recently")
            
            # Analyze document types
            doc_types = {}
            for doc in recent_docs:
                doc_type = doc.get('type', 'Unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            if doc_types:
                logger.info("📝 Recent document types:")
                for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"   • {doc_type}: {count}")
        else:
            logger.info("ℹ️ No recent pipeline activity found")
            
    except Exception as e:
        logger.error(f"❌ Error checking pipeline logs: {e}")

async def display_final_status():
    """Enhanced final system status display with comprehensive metrics"""
    try:
        logger.info("📊 Final system status:")
        
        # Get comprehensive stats
        doc_count = await db_manager.get_document_count()
        agency_count = await db_manager.get_agency_count()
        recent_docs = await db_manager.get_recent_documents(days=7, limit=100)
        latest_date = await db_manager.get_latest_publication_date()
        
        logger.info(f"📈 Total documents: {doc_count}")
        logger.info(f"🏛️ Total agencies: {agency_count}")
        logger.info(f"📅 Recent documents (7 days): {len(recent_docs)}")
        logger.info(f"📆 Latest publication: {latest_date}")
        
        if recent_docs:
            # Agency distribution
            agencies = {}
            for doc in recent_docs:
                agency = doc.get('agency', 'Unknown')
                if agency and agency.strip():
                    agencies[agency] = agencies.get(agency, 0) + 1
                    
            if agencies:
                logger.info("🏛️ Recent agency activity:")
                sorted_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:5]
                for agency, count in sorted_agencies:
                    logger.info(f"   • {agency}: {count} documents")
        
        logger.info("✅ System ready for queries!")
        
    except Exception as e:
        logger.error(f"❌ Error displaying final status: {e}")

async def initialize_system():
    """Enhanced system initialization with comprehensive error handling"""
    global agent, pipeline, system_initialized, initialization_error
    
    try:
        logger.info("🚀 Starting system initialization...")
        
        # Initialize database with health check
        await db_manager.initialize_database()
        logger.info("✅ Database initialized")
        
        # Test database connection
        health = await db_manager.health_check()
        if not health:
            raise Exception("Database health check failed")
        logger.info("✅ Database health check passed")
        
        # Initialize pipeline
        pipeline = DataPipeline()
        logger.info("✅ Pipeline initialized")
        
        # Run comprehensive pipeline check (integrates all run_pipeline.py functions)
        logger.info("🔄 Running comprehensive pipeline check...")
        pipeline_success = await comprehensive_pipeline_check()
        
        # Initialize agent after data is ready
        agent = FederalRegistryAgent()
        logger.info("✅ Agent initialized")
        
        if pipeline_success:
            logger.info("🎉 System initialization completed successfully!")
            logger.info("💡 RAG system is ready with comprehensive data coverage")
            system_initialized = True
            initialization_error = None
        else:
            logger.warning("⚠️ System initialized but pipeline had issues")
            system_initialized = True
            initialization_error = "Pipeline initialization had issues"
            
        return True
        
    except Exception as e:
        error_msg = f"Error initializing system: {e}"
        logger.error(f"❌ {error_msg}")
        initialization_error = error_msg
        system_initialized = False
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifecycle management"""
    # Startup
    logger.info("🚀 Starting up RAG Federal Registry System...")
    success = await initialize_system()
    if success:
        logger.info("✅ RAG System initialized successfully")
    else:
        logger.error("❌ Failed to initialize RAG system")
        logger.error(f"Initialization error: {initialization_error}")
    yield
    # Shutdown
    logger.info("🔄 Shutting down application...")
    global pipeline_running
    if pipeline_running:
        logger.info("⏳ Waiting for pipeline to complete...")
        # Could add more graceful shutdown logic here
    logger.info("👋 Application shutdown complete")

# Enhanced FastAPI app initialization
app = FastAPI(
    title="Federal Registry RAG Agent",
    description="AI Agent for querying US Federal Registry documents with comprehensive data coverage",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Static files (for CSS/JS if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Enhanced endpoints

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the main chat interface with system status"""
    system_status = {
        "initialized": system_initialized,
        "error": initialization_error,
        "pipeline_running": pipeline_running
    }
    return templates.TemplateResponse("chat.html", {
        "request": request, 
        "system_status": system_status
    })

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """Enhanced chat endpoint with timing and source information"""
    start_time = datetime.now()
    
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if not system_initialized:
            raise HTTPException(
                status_code=503, 
                detail=f"System not initialized: {initialization_error or 'Unknown error'}"
            )
            
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Get response from agent
        response = await agent.get_response(request.message)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract sources if available and requested
        sources = None
        if request.include_sources and isinstance(response.get("response"), dict):
            sources = response["response"].get("sources", [])
        
        return ChatResponse(
            status=response["status"],
            response=response["response"] if isinstance(response["response"], str) 
                    else response["response"].get("text", str(response["response"])),
            timestamp=response["timestamp"],
            sources=sources,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/pipeline/run", response_model=PipelineResponse)
async def run_data_pipeline(
    background_tasks: BackgroundTasks,
    request: PipelineRequest = Body(default_factory=lambda: PipelineRequest(start_date=None, end_date=None))
) -> PipelineResponse:
    """Enhanced pipeline execution with background tasks support"""
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
    
    start_time = datetime.now()
    
    try:
        # Run pipeline based on parameters
        if request.force_full:
            # Force full pipeline run
            pipeline_running = True
            success = await run_full_pipeline()
            message = "Full pipeline run completed"
        elif request.start_date and request.end_date:
            # Specific date range
            pipeline_running = True
            success = await pipeline.run_pipeline(request.start_date, request.end_date)
            message = f"Pipeline completed for date range {request.start_date} to {request.end_date}"
        else:
            # Comprehensive check and update
            success = await comprehensive_pipeline_check()
            message = "Comprehensive pipeline check and update completed"
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if success:
            # Get final stats
            doc_count = await db_manager.get_document_count()
            return PipelineResponse(
                status="success",
                message=f"{message}. Total documents: {doc_count}",
                documents_processed=doc_count,
                duration=duration
            )
        else:
            return PipelineResponse(
                status="error",
                message="Pipeline failed. Check logs for more details.",
                duration=duration
            )
            
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Pipeline execution error: {e}")
        return PipelineResponse(
            status="error",
            message=f"Pipeline error: {str(e)}",
            duration=duration
        )
    finally:
        pipeline_running = False

@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Enhanced pipeline status with detailed system information"""
    try:
        doc_count = await db_manager.get_document_count()
        agency_count = await db_manager.get_agency_count()
        recent_docs = await db_manager.get_recent_documents(days=7, limit=10)
        latest_date = await db_manager.get_latest_publication_date()
        
        # Calculate data freshness
        data_freshness_days = None
        if latest_date:
            try:
                latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                data_freshness_days = (datetime.now() - latest_dt).days
            except:
                pass
        
        return {
            "running": pipeline_running,
            "system_initialized": system_initialized,
            "initialization_error": initialization_error,
            "message": "Pipeline is currently running" if pipeline_running else "Pipeline is idle",
            "total_documents": doc_count,
            "total_agencies": agency_count,
            "recent_documents": len(recent_docs),
            "last_update": latest_date,
            "data_freshness_days": data_freshness_days,
            "data_status": "current" if data_freshness_days and data_freshness_days <= 7 else "needs_update"
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return {
            "running": pipeline_running,
            "system_initialized": system_initialized,
            "message": f"Status check error: {str(e)}",
            "error": str(e)
        }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint with detailed diagnostics"""
    try:
        # Test database connection
        db_health = await db_manager.health_check()
        count = await db_manager.get_document_count() if db_health else 0
        recent_count = len(await db_manager.get_recent_documents(days=7, limit=100)) if db_health else 0
        
        # Check data freshness
        latest_date = await db_manager.get_latest_publication_date() if db_health else None
        data_freshness_days = None
        if latest_date:
            try:
                latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                data_freshness_days = (datetime.now() - latest_dt).days
            except:
                pass
        
        overall_status = "healthy" if (
            db_health and 
            system_initialized and 
            agent is not None and 
            pipeline is not None and
            count > 0
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "database": "connected" if db_health else "disconnected",
            "agent": "ready" if agent else "not initialized",
            "pipeline": "ready" if pipeline else "not initialized",
            "system_initialized": system_initialized,
            "initialization_error": initialization_error,
            "document_count": count,
            "recent_documents": recent_count,
            "latest_publication": latest_date,
            "data_freshness_days": data_freshness_days,
            "data_freshness": "current" if data_freshness_days and data_freshness_days <= 7 else "needs_update",
            "pipeline_running": pipeline_running
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "system_initialized": system_initialized,
            "pipeline_running": pipeline_running
        }

@app.get("/api/stats", response_model=SystemStats)
async def get_database_stats():
    """Enhanced database statistics with comprehensive metrics"""
    try:
        doc_count = await db_manager.get_document_count()
        agency_count = await db_manager.get_agency_count()
        latest_docs = await db_manager.get_latest_documents(10)
        recent_docs = await db_manager.get_recent_documents(days=30, limit=500)
        latest_date = await db_manager.get_latest_publication_date()
        
        # Calculate data freshness
        data_freshness_days = None
        if latest_date:
            try:
                latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                data_freshness_days = (datetime.now() - latest_dt).days
            except:
                pass
        
        # Agency distribution from recent docs
        agencies = {}
        for doc in recent_docs:
            agency = doc.get('agency', 'Unknown')
            if agency and agency.strip():
                agencies[agency] = agencies.get(agency, 0) + 1
        
        top_agencies = sorted(agencies.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return SystemStats(
            total_documents=doc_count,
            recent_documents=len(recent_docs),
            agency_count=agency_count,
            latest_publication_date=latest_date,
            data_freshness_days=data_freshness_days,
            top_agencies=[{"agency": k, "count": v} for k, v in top_agencies]
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/recent-documents")
async def get_recent_documents(days: int = 7, limit: int = 20):
    """Get recent documents with filtering options"""
    try:
        recent_docs = await db_manager.get_recent_documents(days=days, limit=limit)
        return {
            "documents": recent_docs,
            "count": len(recent_docs),
            "days": days,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Recent documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recent documents: {str(e)}")

# Example queries endpoint for UI
@app.get("/api/example-queries")
async def get_example_queries():
    """Get example queries for the chat interface"""
    return {
        "queries": [
            "What are the recent executive orders from the last 7 days?",
            "Find documents about artificial intelligence regulations",
            "Show me documents from the Department of Defense",
            "Search for climate change policies",
            "What documents were published yesterday?",
            "Find EPA environmental regulations",
            "Show me recent FDA drug approvals",
            "What are the latest immigration policy changes?",
            "Find documents about cybersecurity regulations",
            "Show me recent Department of Education policies"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
        access_log=True
    )