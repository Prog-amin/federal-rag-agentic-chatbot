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
agent: Optional[FederalRegistryAgent] = None
pipeline: Optional[DataPipeline] = None
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
        
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing system: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    success = await initialize_system()
    if success:
        logger.info("🚀 System initialized successfully")
    else:
        logger.error("❌ Failed to initialize system")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down application")

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the main chat interface"""
    if not os.path.exists("templates/chat.html"):
        # Return a simple HTML interface if template doesn't exist
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Federal Registry RAG Agent</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin: 10px 0; }
                .input-group { display: flex; gap: 10px; margin: 10px 0; }
                .input-group input { flex: 1; padding: 10px; }
                .input-group button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
                .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .user-message { background: #e3f2fd; }
                .bot-message { background: #f5f5f5; }
                .status { margin: 20px 0; padding: 10px; background: #fff3cd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🏛️ Federal Registry RAG Agent</h1>
            <p>Ask questions about US Federal Registry documents including executive orders, regulations, and government publications.</p>
            
            <div id="chat-container" class="chat-container"></div>
            
            <div class="input-group">
                <input type="text" id="message-input" placeholder="Ask about federal documents..." onkeypress="handleEnter(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
            
            <div class="status">
                <h3>System Controls</h3>
                <button onclick="getStatus()">Refresh Status</button>
                <button onclick="runPipeline()">Update Documents</button>
                <div id="status-display">Ready</div>
            </div>
            
            <div class="status">
                <h3>Sample Queries</h3>
                <button onclick="setQuery('What are the recent executive orders from the last 7 days?')">Recent Executive Orders</button>
                <button onclick="setQuery('Find documents about artificial intelligence regulations')">AI Regulations</button>
                <button onclick="setQuery('Show me documents from the Department of Defense')">DoD Documents</button>
            </div>
            
            <script>
                function handleEnter(event) {
                    if (event.key === 'Enter') {
                        sendMessage();
                    }
                }
                
                function setQuery(query) {
                    document.getElementById('message-input').value = query;
                }
                
                async function sendMessage() {
                    const input = document.getElementById('message-input');
                    const message = input.value.trim();
                    if (!message) return;
                    
                    const chatContainer = document.getElementById('chat-container');
                    
                    // Add user message
                    chatContainer.innerHTML += `<div class="message user-message"><strong>You:</strong> ${message}</div>`;
                    input.value = '';
                    
                    // Add loading message
                    chatContainer.innerHTML += `<div class="message bot-message" id="loading"><strong>Agent:</strong> Thinking...</div>`;
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: message})
                        });
                        
                        const data = await response.json();
                        
                        // Remove loading message
                        document.getElementById('loading').remove();
                        
                        // Add bot response
                        chatContainer.innerHTML += `<div class="message bot-message"><strong>Agent:</strong> ${data.response}</div>`;
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                        
                    } catch (error) {
                        document.getElementById('loading').innerHTML = '<strong>Agent:</strong> Error: Failed to get response';
                    }
                }
                
                async function getStatus() {
                    try {
                        const response = await fetch('/api/health');
                        const data = await response.json();
                        document.getElementById('status-display').innerHTML = JSON.stringify(data, null, 2);
                    } catch (error) {
                        document.getElementById('status-display').innerHTML = 'Error getting status';
                    }
                }
                
                async function runPipeline() {
                    document.getElementById('status-display').innerHTML = 'Running pipeline...';
                    try {
                        const response = await fetch('/api/pipeline/run', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({})
                        });
                        const data = await response.json();
                        document.getElementById('status-display').innerHTML = data.message;
                    } catch (error) {
                        document.getElementById('status-display').innerHTML = 'Error running pipeline';
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
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