#!/usr/bin/env python3
"""
Main FastAPI app for Hugging Face Spaces deployment
RAG Agentic System for Federal Registry Documents
Enhanced version with comprehensive features and improved error handling
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
import traceback

# Import your modules with error handling
try:
    from database import db_manager
    DATABASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Database module not available: {e}")
    DATABASE_AVAILABLE = False
    db_manager = None

try:
    from agent import FederalRegistryAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Agent module not available: {e}")
    AGENT_AVAILABLE = False
    FederalRegistryAgent = None

try:
    from data_pipeline import DataPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data pipeline module not available: {e}")
    PIPELINE_AVAILABLE = False
    DataPipeline = None

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
agent: Optional[Any] = None
pipeline: Optional[Any] = None
pipeline_running = False
system_initialized = False
initialization_error = None

# Mock implementations for when modules are not available
class MockDBManager:
    async def initialize_database(self):
        logger.info("Using mock database manager")
        return True
    
    async def health_check(self):
        return True
    
    async def get_document_count(self):
        return 100
    
    async def get_agency_count(self):
        return 25
    
    async def get_recent_documents(self, days=7, limit=10):
        return [
            {
                "title": "Mock Federal Document",
                "agency": "Mock Agency",
                "publication_date": "2024-12-01",
                "type": "Rule"
            }
        ]
    
    async def get_latest_publication_date(self):
        return "2024-12-01"
    
    async def get_latest_documents(self, limit=10):
        return await self.get_recent_documents(days=1, limit=limit)

class MockAgent:
    async def get_response(self, message):
        return {
            "status": "success",
            "response": f"Mock response to: {message}. This is a demo response as the full system is not initialized.",
            "timestamp": datetime.now().timestamp()
        }

class MockPipeline:
    async def run_pipeline(self, start_date=None, end_date=None):
        logger.info(f"Mock pipeline run: {start_date} to {end_date}")
        await asyncio.sleep(1)  # Simulate processing
        return True

async def safe_db_operation(operation_name, *args, **kwargs):
    """Safely execute database operations with fallback"""
    try:
        if DATABASE_AVAILABLE and db_manager:
            operation = getattr(db_manager, operation_name, None)
            if operation:
                return await operation(*args, **kwargs)
        
        # Use mock operations as fallback
        mock_db = MockDBManager()
        mock_method = getattr(mock_db, operation_name, None)
        if mock_method:
            return await mock_method(*args, **kwargs)
        return None
    except Exception as e:
        logger.error(f"Database operation {operation_name} failed: {e}")
        return None

async def comprehensive_pipeline_check():
    """Comprehensive pipeline check with enhanced error handling"""
    global pipeline, pipeline_running
    try:
        logger.info("🔍 Starting comprehensive pipeline check...")
        
        # Check if we have a real pipeline
        if not PIPELINE_AVAILABLE or pipeline is None:
            logger.warning("⚠️ Pipeline not available, using mock")
            return True
        
        # 1. Check data status
        logger.info("📊 Checking current data status...")
        recent_docs = await safe_db_operation('get_recent_documents', days=365, limit=1000)
        
        if recent_docs and len(recent_docs) > 0:
            total_recent = len(recent_docs)
            logger.info(f"📈 Found {total_recent} documents")
            
            # Check if data is recent
            try:
                valid_dates = [doc.get('publication_date') for doc in recent_docs 
                             if doc.get('publication_date')]
                if valid_dates:
                    latest_date = max(valid_dates)
                    latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                    days_old = (datetime.now() - latest_dt).days
                    
                    if days_old > 7:
                        logger.info(f"⚠️ Latest data is {days_old} days old, running update...")
                        await run_update_pipeline()
                    else:
                        logger.info("✅ Data is current")
            except Exception as e:
                logger.warning(f"Date parsing error: {e}")
        else:
            logger.info("📥 No documents found, running initial pipeline...")
            await run_full_pipeline()
        
        return True
    except Exception as e:
        logger.error(f"❌ Pipeline check failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def run_full_pipeline():
    """Run full pipeline with enhanced error handling"""
    global pipeline_running
    
    try:
        pipeline_running = True
        start_time = datetime.now()
        logger.info("🔄 Starting full data pipeline...")
        
        if pipeline and hasattr(pipeline, 'run_pipeline'):
            success = await pipeline.run_pipeline()
        else:
            # Mock pipeline run
            mock_pipeline = MockPipeline()
            success = await mock_pipeline.run_pipeline()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if success:
            logger.info(f"✅ Full pipeline completed in {duration:.2f}s!")
            return True
        else:
            logger.error("❌ Full pipeline failed")
            return False
    except Exception as e:
        logger.error(f"❌ Full pipeline error: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        pipeline_running = False

async def run_update_pipeline():
    """Run pipeline for recent data updates"""
    global pipeline_running
    
    try:
        pipeline_running = True
        start_time = datetime.now()
        
        # Get data for last 7 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"🔄 Running update pipeline for {start_date} to {end_date}")
        
        if pipeline and hasattr(pipeline, 'run_pipeline'):
            success = await pipeline.run_pipeline(start_date, end_date)
        else:
            # Mock pipeline run
            mock_pipeline = MockPipeline()
            success = await mock_pipeline.run_pipeline(start_date, end_date)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if success:
            logger.info(f"✅ Update pipeline completed in {duration:.2f}s!")
            return True
        else:
            logger.error("❌ Update pipeline failed")
            return False
    except Exception as e:
        logger.error(f"❌ Update pipeline error: {e}")
        return False
    finally:
        pipeline_running = False

async def initialize_system():
    """Enhanced system initialization with comprehensive error handling"""
    global agent, pipeline, system_initialized, initialization_error
    
    try:
        logger.info("🚀 Starting system initialization...")
        
        # Initialize database
        if DATABASE_AVAILABLE and db_manager:
            try:
                await db_manager.initialize_database()
                health = await db_manager.health_check()
                if health:
                    logger.info("✅ Database initialized and healthy")
                else:
                    logger.warning("⚠️ Database initialized but health check failed")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                # Continue with mock database
        else:
            logger.info("✅ Using mock database (modules not available)")
        
        # Initialize pipeline
        if PIPELINE_AVAILABLE and DataPipeline:
            try:
                pipeline = DataPipeline()
                logger.info("✅ Pipeline initialized")
            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}")
                pipeline = MockPipeline()
                logger.info("✅ Using mock pipeline")
        else:
            pipeline = MockPipeline()
            logger.info("✅ Using mock pipeline (module not available)")
        
        # Run pipeline check
        try:
            pipeline_success = await comprehensive_pipeline_check()
            if pipeline_success:
                logger.info("✅ Pipeline check completed")
            else:
                logger.warning("⚠️ Pipeline check had issues")
        except Exception as e:
            logger.error(f"Pipeline check failed: {e}")
            pipeline_success = False
        
        # Initialize agent
        if AGENT_AVAILABLE and FederalRegistryAgent:
            try:
                agent = FederalRegistryAgent()
                logger.info("✅ Agent initialized")
            except Exception as e:
                logger.error(f"Agent initialization failed: {e}")
                agent = MockAgent()
                logger.info("✅ Using mock agent")
        else:
            agent = MockAgent()
            logger.info("✅ Using mock agent (module not available)")
        
        # Final status
        logger.info("🎉 System initialization completed!")
        system_initialized = True
        initialization_error = None
        
        return True
        
    except Exception as e:
        error_msg = f"Error initializing system: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        initialization_error = error_msg
        system_initialized = False
        
        # Set up mock components as fallback
        if agent is None:
            agent = MockAgent()
        if pipeline is None:
            pipeline = MockPipeline()
        
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifecycle management"""
    # Startup
    logger.info("🚀 Starting up RAG Federal Registry System...")
    
    try:
        success = await initialize_system()
        if success:
            logger.info("✅ RAG System initialized successfully")
        else:
            logger.warning("⚠️ RAG System initialized with warnings/errors")
            logger.info("💡 System will run in demo/mock mode")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        logger.error(traceback.format_exc())
        # Ensure basic functionality with mocks
        global agent, pipeline, system_initialized
        if agent is None:
            agent = MockAgent()
        if pipeline is None:
            pipeline = MockPipeline()
        system_initialized = True
        logger.info("🔧 Running in emergency mock mode")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down application...")
    global pipeline_running
    if pipeline_running:
        logger.info("⏳ Waiting for pipeline to complete...")
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

# Templates setup with error handling
templates = None
try:
    if os.path.exists("templates"):
        templates = Jinja2Templates(directory="templates")
        logger.info("✅ Templates directory found")
    else:
        logger.warning("⚠️ Templates directory not found")
except Exception as e:
    logger.error(f"Template setup error: {e}")

# Static files (for CSS/JS if needed)
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("✅ Static files directory mounted")
except Exception as e:
    logger.error(f"Static files setup error: {e}")

# Enhanced endpoints
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the main chat interface with system status"""
    if templates:
        system_status = {
            "initialized": system_initialized,
            "error": initialization_error,
            "pipeline_running": pipeline_running
        }
        try:
            return templates.TemplateResponse("chat.html", {
                "request": request, 
                "system_status": system_status
            })
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
    
    # Fallback HTML response
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federal Registry RAG Agent</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .status {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .success {{ background-color: #d4edda; color: #155724; }}
            .warning {{ background-color: #fff3cd; color: #856404; }}
            .error {{ background-color: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <h1>Federal Registry RAG Agent</h1>
        <div class="status {'success' if system_initialized else 'error'}">
            <h3>System Status</h3>
            <p>Initialized: {'Yes' if system_initialized else 'No'}</p>
            <p>Pipeline Running: {'Yes' if pipeline_running else 'No'}</p>
            {f'<p>Error: {initialization_error}</p>' if initialization_error else ''}
        </div>
        <p>API endpoints available at <a href="/docs">/docs</a></p>
        <p>Health check at <a href="/api/health">/api/health</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint with detailed diagnostics"""
    try:
        # Test database connection
        db_health = True
        count = 0
        recent_count = 0
        latest_date = None
        
        if DATABASE_AVAILABLE and db_manager:
            try:
                db_health = await db_manager.health_check()
                if db_health:
                    count = await db_manager.get_document_count()
                    recent_docs = await db_manager.get_recent_documents(days=7, limit=100)
                    recent_count = len(recent_docs) if recent_docs else 0
                    latest_date = await db_manager.get_latest_publication_date()
            except Exception as e:
                logger.error(f"Database health check error: {e}")
                db_health = False
        else:
            # Mock values
            mock_db = MockDBManager()
            count = await mock_db.get_document_count()
            recent_docs = await mock_db.get_recent_documents(days=7, limit=100)
            recent_count = len(recent_docs)
            latest_date = await mock_db.get_latest_publication_date()
        
        # Check data freshness
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
        ) else "partial" if system_initialized else "unhealthy"
        
        return {
            "status": overall_status,
            "database": "connected" if db_health else "mock/disconnected",
            "agent": "ready" if agent else "not initialized",
            "pipeline": "ready" if pipeline else "not initialized",
            "system_initialized": system_initialized,
            "initialization_error": initialization_error,
            "document_count": count,
            "recent_documents": recent_count,
            "latest_publication": latest_date,
            "data_freshness_days": data_freshness_days,
            "data_freshness": "current" if data_freshness_days and data_freshness_days <= 7 else "needs_update",
            "pipeline_running": pipeline_running,
            "modules_available": {
                "database": DATABASE_AVAILABLE,
                "agent": AGENT_AVAILABLE,
                "pipeline": PIPELINE_AVAILABLE
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        logger.error(traceback.format_exc())
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
        # Use safe operations or mocks
        if DATABASE_AVAILABLE and db_manager:
            doc_count = await safe_db_operation('get_document_count') or 0
            agency_count = await safe_db_operation('get_agency_count') or 0
            recent_docs = await safe_db_operation('get_recent_documents', days=30, limit=500) or []
            latest_date = await safe_db_operation('get_latest_publication_date')
        else:
            # Mock data
            mock_db = MockDBManager()
            doc_count = await mock_db.get_document_count()
            agency_count = await mock_db.get_agency_count()
            recent_docs = await mock_db.get_recent_documents(days=30, limit=500)
            latest_date = await mock_db.get_latest_publication_date()
        
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
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

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
            success = await run_full_pipeline()
            message = "Full pipeline run completed"
        elif request.start_date and request.end_date:
            # Specific date range
            pipeline_running = True
            try:
                if hasattr(pipeline, 'run_pipeline'):
                    success = await pipeline.run_pipeline(request.start_date, request.end_date)
                else:
                    success = await MockPipeline().run_pipeline(request.start_date, request.end_date)
                message = f"Pipeline completed for date range {request.start_date} to {request.end_date}"
            finally:
                pipeline_running = False
        else:
            # Comprehensive check and update
            success = await comprehensive_pipeline_check()
            message = "Comprehensive pipeline check and update completed"
        
        duration = (datetime.now() - start_time).total_seconds()
        
        if success:
            # Get final stats
            doc_count = await safe_db_operation('get_document_count') or 0
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

@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Enhanced pipeline status with detailed system information"""
    try:
        doc_count = await safe_db_operation('get_document_count') or 0
        agency_count = await safe_db_operation('get_agency_count') or 0
        recent_docs = await safe_db_operation('get_recent_documents', days=7, limit=10) or []
        latest_date = await safe_db_operation('get_latest_publication_date')
        
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

@app.get("/api/recent-documents")
async def get_recent_documents(days: int = 7, limit: int = 20):
    """Get recent documents with filtering options"""
    try:
        recent_docs = await safe_db_operation('get_recent_documents', days=days, limit=limit) or []
        return {
            "documents": recent_docs,
            "count": len(recent_docs),
            "days": days,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Recent documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recent documents: {str(e)}")

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
    port = int(os.environ.get("PORT", 7860))  # Add this line
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,  # Change from hardcoded 7860
        reload=False,
        log_level="info",
        access_log=True
    )