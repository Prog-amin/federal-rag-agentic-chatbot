#!/usr/bin/env python3
"""
Main Gradio app for Hugging Face Spaces deployment
RAG Agentic System for Federal Registry Documents
"""
import gradio as gr
import asyncio
import os
import logging
from typing import List, Tuple
from agent import FederalRegistryAgent
from database import db_manager
from data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
agent = None
pipeline = None

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

def run_async(coro):
    """Helper to run async functions in Gradio"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def chat_with_agent(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """Handle chat interaction with the agent"""
    if not message.strip():
        return "", history
    
    if agent is None:
        error_msg = "❌ System not initialized. Please wait for initialization to complete."
        history.append((message, error_msg))
        return "", history
    
    try:
        # Get response from agent
        response_data = run_async(agent.get_response(message))
        
        if response_data["status"] == "success":
            response = response_data["response"]
        else:
            response = f"❌ Error: {response_data['response']}"
        
        # Add to history
        history.append((message, response))
        
    except Exception as e:
        error_response = f"❌ Error processing your request: {str(e)}"
        history.append((message, error_response))
    
    return "", history

def run_data_pipeline() -> str:
    """Run the data pipeline to update documents"""
    if pipeline is None:
        return "❌ Pipeline not initialized"
    
    try:
        logger.info("🔄 Starting data pipeline...")
        success = run_async(pipeline.run_pipeline())
        
        if success:
            return "✅ Data pipeline completed successfully! New documents have been loaded."
        else:
            return "❌ Data pipeline failed. Check logs for details."
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return f"❌ Error running pipeline: {str(e)}"

def get_system_status() -> str:
    """Get current system status"""
    status_info = []
    
    # Check database
    try:
        doc_count = run_async(get_document_count())
        status_info.append(f"📊 Documents in database: {doc_count}")
    except Exception as e:
        status_info.append(f"❌ Database error: {str(e)}")
    
    # Check agent
    if agent:
        status_info.append("🤖 Agent: Ready")
    else:
        status_info.append("❌ Agent: Not initialized")
    
    # Check pipeline
    if pipeline:
        status_info.append("⚙️ Pipeline: Ready")
    else:
        status_info.append("❌ Pipeline: Not initialized")
    
    return "\n".join(status_info)

async def get_document_count() -> int:
    """Get total document count from database"""
    try:
        await db_manager._ensure_initialized()
        pool = db_manager.pool
        if pool is None:
            return 0
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM federal_documents")
                result = await cursor.fetchone()
                return result[0] if result else 0
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return 0

# Sample queries for users
SAMPLE_QUERIES = [
    "What are the recent executive orders from the last 7 days?",
    "Find documents about artificial intelligence regulations",
    "Show me documents from the Department of Defense",
    "Search for documents about climate change policies",
    "What documents were published in January 2025?",
]

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Federal Registry RAG Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏛️ Federal Registry RAG Agent
        
        Ask questions about US Federal Registry documents including executive orders, regulations, and government publications.
        The system uses AI to search through a database of federal documents and provide comprehensive answers.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    value=[],
                    height=400,
                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=("👤", "🤖"),
                    label="Chat with Federal Registry Agent"
                )
                
                msg = gr.Textbox(
                    placeholder="Ask about federal documents...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                    submit_btn = gr.Button("📤 Send", variant="primary", size="sm")
                
                # Sample queries
                gr.Markdown("### 💡 Sample Queries:")
                for query in SAMPLE_QUERIES:
                    sample_btn = gr.Button(
                        query, 
                        variant="secondary", 
                        size="sm",
                        scale=1
                    )
                    sample_btn.click(
                        lambda q=query: q,
                        outputs=msg
                    )
            
            with gr.Column(scale=1):
                # System controls
                gr.Markdown("### ⚙️ System Controls")
                
                status_display = gr.Textbox(
                    label="System Status",
                    value="Initializing system...",
                    interactive=False,
                    lines=6
                )
                
                refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                
                gr.Markdown("### 📊 Data Management")
                
                pipeline_output = gr.Textbox(
                    label="Pipeline Status",
                    value="Ready to run data pipeline",
                    interactive=False,
                    lines=3
                )
                
                run_pipeline_btn = gr.Button("🔄 Update Documents", variant="primary")
                
                gr.Markdown("""
                ### ℹ️ How to Use
                1. **Chat**: Ask questions about federal documents
                2. **Update**: Run pipeline to get latest documents  
                3. **Status**: Check system health
                
                ### 📋 Example Queries
                - Recent executive orders
                - Agency-specific documents
                - Topic-based searches
                - Date range queries
                """)
        
        # Event handlers
        def submit_message(message, history):
            return chat_with_agent(message, history)
        
        def clear_chat():
            return []
        
        # Connect events
        msg.submit(
            submit_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=chatbot
        )
        
        refresh_status_btn.click(
            get_system_status,
            outputs=status_display
        )
        
        run_pipeline_btn.click(
            run_data_pipeline,
            outputs=pipeline_output
        )
        
        # Initialize status on load
        demo.load(
            get_system_status,
            outputs=status_display
        )
    
    return demo

def main():
    """Main function to run the application"""
    print("🚀 Initializing Federal Registry RAG Agent...")
    
    # Initialize system
    init_success = run_async(initialize_system())
    
    if not init_success:
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch with appropriate settings for HF Spaces
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False
    )

if __name__ == "__main__":
    main()