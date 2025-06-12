import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
import aiohttp
from config import LLM_CONFIG

class FederalRegistryAgent:
    def __init__(self):
        self.base_url = LLM_CONFIG['base_url']
        self.model = LLM_CONFIG['model']
        self.api_key = LLM_CONFIG['api_key']
        self.tools: List[Dict[str, Any]] = []
        self.tool_functions: Dict[str, Callable] = {}
        self._tools_initialized = False
        
        self.system_prompt = """You are a US Federal Registry AI assistant with access to a federal documents database.

        **Response Logic:**
        1. **Federal document queries**: Use tools to search database first
        2. **No results found**: Provide general federal knowledge, suggest alternatives
        3. **General federal topics**: Give helpful knowledge, offer document searches
        4. **Off-topic questions**: Brief helpful answer, then redirect to federal topics

        **Guidelines:**
        - Try database search first for federal queries
        - Label general knowledge as "Based on federal processes knowledge..."
        - Keep responses concise and actionable
        - Provide document URLs when available
        - For off-topic: "I can help briefly - [answer]. I specialize in federal documents. Anything about federal regulations I can help with?"

        Your specialty: US federal documents, but you can discuss related government topics when helpful."""

    async def _initialize_tools(self):
        """Initialize tools with proper database setup"""
        if self._tools_initialized:
            return
            
        try:
            # Ensure database is initialized first
            from database import db_manager
            await db_manager.initialize_database()
            
            # Import agent tools after database is ready
            from agent_tools import AGENT_TOOLS_SCHEMA, TOOL_FUNCTIONS
            self.tools = AGENT_TOOLS_SCHEMA
            self.tool_functions = TOOL_FUNCTIONS
            self._tools_initialized = True
            print("✅ Agent tools initialized successfully")
            
        except ImportError as e:
            print(f"Warning: Could not import agent tools: {e}")
            self.tools = []
            self.tool_functions = {}
        except Exception as e:
            print(f"Error initializing tools: {e}")
            self.tools = []
            self.tool_functions = {}

    async def call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Call the local LLM server using OpenAI-compatible API"""
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            if tools is not None and len(tools) > 0:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        print(f"LLM API error {response.status}: {error_text}")
                        return {"error": f"API error: {response.status}"}
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {"error": str(e)}

    async def execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result"""
        function_info = tool_call.get("function", {})
        function_name = function_info.get("name", "")
        function_args_str = function_info.get("arguments", "{}")
        
        try:
            function_args = json.loads(function_args_str)
        except json.JSONDecodeError:
            return json.dumps({
                "status": "error",
                "message": f"Invalid JSON in function arguments: {function_args_str}"
            })

        if function_name not in self.tool_functions:
            return json.dumps({
                "status": "error",
                "message": f"Unknown function: {function_name}"
            })

        try:
            result = await self.tool_functions[function_name](**function_args)
            return result
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error executing {function_name}: {str(e)}"
            })

    async def process_user_query(self, user_message: str) -> str:
        """Process user query and return response"""
        # Ensure tools are initialized
        await self._initialize_tools()
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            try:
                # Call LLM with tools if available
                tools_to_use = self.tools if self.tools else None
                response = await self.call_llm(messages, tools_to_use)
                
                if "error" in response:
                    return f"Error from LLM: {response['error']}"

                choices = response.get("choices", [])
                if not choices:
                    return "No response received from LLM."

                message = choices[0].get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                if tool_calls and isinstance(tool_calls, list):
                    # Add assistant message with tool calls
                    assistant_message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": content or ""
                    }
                    # Only add tool_calls if they exist
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    messages.append(assistant_message)

                    # Execute tool calls
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_result = await self.execute_tool_call(tool_call)
                            tool_message: Dict[str, Any] = {
                                "role": "tool",
                                "tool_call_id": str(tool_call.get("id", "unknown")),
                                "content": tool_result
                            }
                            messages.append(tool_message)
                    continue
                else:
                    # No tool calls, return final response
                    return content or "I couldn't generate a response to your query."

            except Exception as e:
                return f"Error processing request: {str(e)}"

        return "Maximum processing steps reached. Please try rephrasing your question."

    async def get_response(self, user_message: str) -> Dict[str, Any]:
        """Get a formatted response for the user"""
        try:
            response_content = await self.process_user_query(user_message)
            return {
                "status": "success",
                "response": response_content,
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error: {str(e)}",
                "timestamp": asyncio.get_event_loop().time()
            }

# Global agent instance
federal_agent = FederalRegistryAgent()

async def test_agent():
    """Test function for the agent"""
    test_queries = [
        "What are the recent executive orders from the last 7 days?",
        "Find documents about artificial intelligence from the Department of Defense",
        "Show me documents published in January 2025",
        "Search for documents about climate change regulations"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = await federal_agent.get_response(query)
        print(f"Response: {response['response']}")

if __name__ == "__main__":
    asyncio.run(test_agent())