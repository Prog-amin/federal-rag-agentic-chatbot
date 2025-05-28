import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessageToolCallParam
)
from config import LLM_CONFIG
from agent_tools import AGENT_TOOLS_SCHEMA, TOOL_FUNCTIONS

class FederalRegistryAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=LLM_CONFIG['base_url'],
            api_key=LLM_CONFIG['api_key']
        )
        self.model = LLM_CONFIG['model']

        # Convert AGENT_TOOLS_SCHEMA to properly typed tools
        self.tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"]
                }
            )
            for tool in AGENT_TOOLS_SCHEMA
        ]

        self.tool_functions = TOOL_FUNCTIONS

        self.system_prompt = """You are a helpful assistant that provides information about US Federal Registry documents. You have access to a database of federal documents including executive orders, regulations, notices, and other government publications.

Your capabilities include:
1. Searching for documents by keywords or topics
2. Finding documents by date ranges
3. Getting documents from specific agencies
4. Retrieving recent documents

When users ask questions about federal documents, use the appropriate tools to query the database and provide comprehensive, accurate answers. Always cite the document titles, publication dates, and provide URLs when available.

Guidelines:
- Be concise but informative
- Always use tools to get current data rather than making assumptions
- Provide document URLs for users to read the full documents
- If no documents are found, suggest alternative search terms or date ranges
- Summarize key findings when multiple documents are returned
- Focus on the most relevant and recent information

You must use the provided tools to answer questions - do not provide information from your training data about specific federal documents, as the database contains the most up-to-date information."""

    async def execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> str:
        """Execute a tool call and return the result"""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

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
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=self.system_prompt),
            ChatCompletionUserMessageParam(role="user", content=user_message)
        ]

        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.1
                )

                message = response.choices[0].message

                if message.tool_calls:
                    # Convert ChatCompletionMessageToolCall to ChatCompletionMessageToolCallParam
                    tool_call_params = [
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.id,
                            function={
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments
                            },
                            type=tool_call.type
                        )
                        for tool_call in message.tool_calls
                    ]

                    messages.append(ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=message.content or "",
                        tool_calls=tool_call_params
                    ))

                    # Execute tool calls
                    for tool_call in message.tool_calls:
                        tool_result = await self.execute_tool_call(tool_call)
                        messages.append(ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=tool_result
                        ))
                    continue
                else:
                    messages.append(ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=message.content or ""
                    ))
                    return message.content or "I couldn't generate a response to your query."

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