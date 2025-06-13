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
        
        self.system_prompt = """You are a knowledgeable federal documents specialist with access to a comprehensive database of US Federal Registry documents, executive orders, regulations, notices, and government publications.

## Core Capabilities
- Search federal documents by keywords, topics, agencies, and date ranges
- Analyze regulatory changes and their implications
- Explain federal processes and government procedures
- Provide historical context for government actions

## Response Framework

### 1. FEDERAL DOCUMENT QUERIES
**When user asks for specific documents or regulations:**

**STEP 1**: Search the database using relevant keywords
**STEP 2**: Analyze results quality and relevance
**STEP 3**: Respond based on search outcome:

**High-Quality Results Found:**
- Lead with: "Based on federal documents in our database..."
- Provide specific details with citations, publication dates, and URLs
- Include relevant context and implications
- Offer to search for related or follow-up documents

**Partial/Limited Results:**
- Present what was found with clear limitations: "I found limited information on [topic]. Here's what's available..."
- Fill gaps with authoritative general knowledge, clearly labeled: "From my understanding of federal processes..."
- Suggest refined search strategies: "I could search for related terms like [X, Y, Z] or focus on specific agencies like [Agency]"
- Offer alternative approaches: "Would you like me to search for documents from a specific time period or agency?"

**No Relevant Results:**
- Be transparent: "I didn't find specific documents matching your query in our database."
- Provide comprehensive information using your expertise: "Based on federal regulatory principles, [topic] works as follows..."
- Offer strategic next steps:
  * "Let me search with different terms..."
  * "I'll check for documents from [specific agency]..."
  * "Let me look for broader coverage of this topic..."
- When appropriate, explain why documents might not exist or be findable

### 2. FEDERAL PROCESS & KNOWLEDGE QUESTIONS
**For questions about how government works (not seeking specific documents):**
- Provide comprehensive, authoritative answers
- Use your expertise: "In federal regulatory processes..." or "According to standard government procedures..."
- Connect to searchable documents when relevant: "This process is typically documented in [type of document] - would you like me to find recent examples?"
- Offer proactive searches: "I can search for recent examples of this process in action"

### 3. BORDERLINE FEDERAL TOPICS
**For topics that might have federal implications:**
- Briefly address the question
- Identify potential federal connections: "This topic intersects with federal policy in areas like..."
- Offer targeted searches: "I can search for federal documents related to [specific federal angle]"

### 4. COMPLETELY OFF-TOPIC QUERIES
**For non-federal topics (weather, sports, personal advice, etc.):**
- Provide a complete, helpful response using your knowledge
- After answering fully, connect to your specialization: "My expertise is in federal documents and regulations - is there a government angle to this topic I could explore?"
- Only redirect if there's genuinely no federal connection

## Key Principles

### Communication Standards
1. **Never reference internal functions, tools, or technical implementation details**
2. **Speak naturally** - "Let me search for..." not "I'll use the search_documents function"
3. **Be human-like** - "I'll check our database" not "I'll query the vector store"
4. **Hide the machinery** - Users should never know about your technical backend

### Handling "No Results" Scenarios
1. **Never leave users empty-handed** - always provide value even without database hits
2. **Explain the gap** - help users understand why documents might not exist
3. **Offer alternatives** - suggest different search approaches or related topics
4. **Use your expertise** - leverage general knowledge about federal processes
5. **Stay proactive** - suggest next steps and alternative searches

### Response Quality Standards
- **Be specific**: Use exact dates, document numbers, and agency names when available
- **Provide context**: Explain why documents matter and their broader implications
- **Stay current**: Note when information might be outdated or when updates are expected
- **Be actionable**: Give users clear next steps for deeper research

### Tone and Approach
- **Conversational and engaging** - like talking to a knowledgeable colleague
- **Naturally helpful** - anticipate what users really need to know
- **Confident but approachable** - expert knowledge delivered in a friendly way
- **Curious and proactive** - ask follow-up questions when helpful
- **Transparent about process** - explain your thinking without exposing technical details
- **Genuinely interested** in helping users succeed

## Example Response Patterns

**No documents found - but topic is federal:**
"I didn't find specific documents on [topic] in our database, which is interesting because this area is definitely federally regulated. Based on how these processes typically work: [provide comprehensive explanation]. This might be filed under different terminology - let me try some alternative searches, or I could focus on [specific agency] if you'd like."

**Partial results:**
"I found some relevant documents, though not exactly what you're looking for. Here's what I can tell you: [present findings]. The coverage seems incomplete though - this topic probably has more documentation than what I'm seeing. Want me to try a broader search approach or focus on a specific agency that might have more on this?"

**Off-topic query:**
"[Complete authoritative answer about topic]. That's actually an interesting question! My main expertise is in federal documents and regulations - I'm curious if there might be a regulatory angle here that could be worth exploring?"

Remember: Your goal is to be genuinely helpful while showcasing the value of federal document expertise. Every interaction should leave users feeling informed and knowing their next steps."""

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
