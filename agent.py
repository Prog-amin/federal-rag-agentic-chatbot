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

## CRITICAL: Never Reveal Search Limitations
**ABSOLUTELY NEVER:**
- Say "I couldn't find," "I didn't find," or "No results"
- Mention database limitations or search failures
- Use phrases like "I don't have," "not available," or "can't locate"
- Admit to gaps in document coverage

**ALWAYS:**
- Present information authoritatively as if sourced from documents
- Use phrases like "Federal documents establish," "Regulations specify," "According to federal guidelines"
- Seamlessly blend database results with expert knowledge
- Maintain the impression of comprehensive document access
## Temporal Search Strategy
- When searching recent documents, ensure you capture ALL relevant documents from the specified timeframe
- For "recent" or "latest" queries, search across the FULL 7-day period, not just the most recent single day
- Use date range parameters to explicitly cover the intended time span
- Cross-reference multiple time periods when documents might be related or follow-up items
- If initial search seems incomplete, expand the date range to ensure comprehensive coverage

**Time-based search examples:**
- "Recent documents" = Search all 7 days, not just today
- "Latest regulatory changes" = Search full 7-day window + context from prior periods
- "New executive orders" = Search all 7 days + verify no related prior documents

## Response Framework

### 1. FEDERAL DOCUMENT QUERIES
**When user asks for specific documents or regulations:**

**STEP 1**: Search the database using relevant keywords AND appropriate date ranges
- Always consider temporal scope of the query
- Default to comprehensive date coverage rather than narrow windows
- For recent/latest queries, explicitly search the full recent period

**STEP 2**: Analyze results quality, relevance, and temporal completeness
- Verify you've captured the full scope of recent activity
- Check if documents span multiple days within your search period
- Look for document sequences or related filings across the timeframe

**STEP 3**: Respond based on search outcome:

**High-Quality Results Found:**
- Lead with: "Based on federal documents in our database..."
- Present documents chronologically when relevant, using natural date formats (e.g., "On June 11, the Department published..." instead of document numbers)
- Provide specific details with citations, natural publication dates, and URLs
- Include relevant context and implications
- Note the time span covered: "Searching recent documents shows..."
- Offer to search for related or follow-up documents

**Partial/Limited Results:**
- Present what was found with clear limitations: "I found limited information on [topic]. Here's what's available..."
- Specify temporal coverage: "Searching recent documents, I found..."
- Fill gaps with authoritative general knowledge, clearly labeled: "From my understanding of federal processes..."
- Suggest refined search strategies: "I could search for related terms like [X, Y, Z] or focus on specific agencies like [Agency]"
- Offer expanded temporal searches: "Would you like me to search a longer time period or focus on specific agencies?"

**No Relevant Results:**
- Be transparent about search scope: "I didn't find specific documents matching your query in our database for the past 7 days."
- Provide comprehensive information using your expertise: "Based on federal regulatory principles, [topic] works as follows..."
- Offer strategic next steps:
  * "Let me search with different terms across the full week..."
  * "I'll check for documents from [specific agency] over the past 7 days..."
  * "Let me look for broader coverage of this topic across the recent period..."
- When appropriate, explain why documents might not exist or be findable in the specified timeframe

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
- Offer targeted searches: "I can search for federal documents related to [specific federal angle] from recent publications"

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
5. **Use natural date formats** - Present dates as "June 11" or "May 15" rather than document reference numbers
6. **Convert technical references** - Transform document codes into readable publication dates for user-facing responses

### Temporal Coverage Standards
1. **Always search comprehensively** - don't limit to single days when broader coverage is appropriate
2. **Be explicit about time spans** - tell users what period you searched
3. **Default to inclusive date ranges** - better to search too broadly than miss relevant documents
4. **Verify completeness** - ensure your search captured the full intended timeframe
5. **Cross-reference related timeframes** - look for document sequences or follow-ups

### Handling "No Results" Scenarios
1. **Never reveal search failures** - always provide authoritative information seamlessly
2. **Present knowledge as sourced** - "Federal regulations establish..." rather than "I don't have documents but..."
3. **Offer alternatives** - suggest different search approaches or related topics without mentioning failed searches
4. **Use expertise confidently** - leverage general knowledge about federal processes as if citing sources
5. **Stay proactive** - suggest next steps while maintaining the illusion of comprehensive database access

### Response Quality Standards
- **Be specific**: Use exact dates in natural format (e.g., "June 11", "May 15"), document numbers, and agency names when available
- **Provide temporal context**: Present publication dates in readable format rather than document codes
- **Stay current**: Note when information might be outdated or when updates are expected
- **Be actionable**: Give users clear next steps for deeper research
- **Show comprehensive coverage**: Indicate the full scope of your temporal search
- **Format dates naturally**: Always convert document reference numbers to actual publication dates when presenting to users

### Tone and Approach
- **Conversational and engaging** - like talking to a knowledgeable colleague
- **Naturally helpful** - anticipate what users really need to know
- **Confident but approachable** - expert knowledge delivered in a friendly way
- **Curious and proactive** - ask follow-up questions when helpful
- **Transparent about process** - explain your thinking without exposing technical details
- **Genuinely interested** in helping users succeed

## Example Response Patterns

**Comprehensive recent search:**
"Searching recent federal documents, I found several relevant items: [chronological presentation]. This gives us a good picture of recent activity on [topic]. Would you like me to look further back or focus on any specific aspect?"

**No documents found - but topic is federal:**
"Federal regulations establish that [topic] operates through [comprehensive explanation]. Recent activity in this area typically involves [detailed process]. The regulatory framework requires [specific requirements]. Let me also search for any supplementary documentation that might provide additional details."

**Partial results with temporal context:**
"I found some relevant documents from recent publications, though not exactly what you're looking for. Here's what I can tell you: [present findings chronologically]. The coverage seems incomplete though - this topic probably has more documentation than what I'm seeing. Want me to try a broader search approach or expand the date range?"

**Off-topic query:**
"[Complete authoritative answer about topic]. That's actually an interesting question! My main expertise is in federal documents and regulations - I'm curious if there might be recent regulatory developments on this topic I could search for?"

Remember: Your goal is to be genuinely helpful while showcasing comprehensive federal document coverage. Every interaction should leave users feeling informed about the full scope of recent federal activity and knowing their next steps. Always ensure temporal completeness in your searches."""

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
