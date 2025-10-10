import os
import sys
import logging
from collections.abc import AsyncIterable
from typing import Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama.chat_models import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage, ToolMessage
from dotenv import load_dotenv


# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Determine LLM provider from environment, with fallback
llm_selected = os.getenv("AGENT_TYPE", "ollama").lower()
logger.info(f"Selected LLM provider: {llm_selected}")

def create_mcp_client():
    MCP_SERVER_URL=os.getenv('MCP_SERVER_URL')
    return MultiServerMCPClient({
        "k8s": {
            "url":f"{MCP_SERVER_URL}/sse",
            "transport": "sse"
        }
    })

async def get_available_tools():
    client = create_mcp_client()
    return await client.get_tools()



class k8sAgent:
    SYSTEM_INSTRUCTION = (
        "You are an assistant that helps users get information about Kubernetes. "
        "You can only use the provided MCP tools to answer. "
        "If you cannot help, clearly communicate it."
    )

    def __init__(self):
        self.agent = None
        self.selected_tool_objects = []

    async def setup_agent(self, selected_tools: list[str]):
        """Prepare the agent with selected tools and LLM model."""
        # Connect to MCP
        client = create_mcp_client()
        all_tools = await client.get_tools()

        self.selected_tool_objects = all_tools

        if not self.selected_tool_objects:
            raise ValueError("No valid tool selected.")

        # Select LLM with automatic fallback
        llm = None

        try:
            if llm_selected == "ollama":
                logger.info("Trying to connect to Ollama...")
                ollama_url = os.getenv('OLLAMA_URL')
                ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
                logger.info(f"Ollama URL: {ollama_url}, Model: {ollama_model}")

                llm = ChatOllama(
                    model=ollama_model,
                    base_url=ollama_url
                )
                # Quick test to verify Ollama works
                await llm.ainvoke("test")
                logger.info("✅ Ollama configured correctly")

            elif llm_selected == "oai":
                logger.info("Connecting to OpenAI...")
                llm = setup_llm(
                    deployment_name=os.getenv("OAI_DEPLOYMENT_NAME_GPT_4O"),
                    api_key=os.getenv("OAI_API_KEY"),
                    api_version=os.getenv("OAI_API_VERSION"),
                    azure_endpoint=os.getenv("OAI_AZURE_ENDPOINT")
                )

        except Exception as e:
            logger.warning(f"Error with {llm_selected}: {e}")

        if llm is None:
            logger.error("Unable to configure an LLM provider")
            raise ValueError("Unable to configure an LLM provider")
        
        memory = MemorySaver()
        # Create agent with checkpointer
        logger.info("Creating agent...")
        self.agent = create_react_agent(
            model=llm,
            tools=self.selected_tool_objects,
            prompt=self.SYSTEM_INSTRUCTION,
            checkpointer=memory
        )

    async def stream(self, user_question: str, selected_tools: list[str]) -> AsyncIterable[dict[str, Any]]:
        """Stream intermediate updates and the final response."""
        if self.agent is None:
            await self.setup_agent(selected_tools)


        prompt = {"messages": [{"role": "user", "content": user_question}]}
        config = {"configurable": {"thread_id": "k8s_session"}}

        final_response = None
        async for step in self.agent.astream(prompt, config, stream_mode="values"):
            message = step["messages"][-1]

            # Handle different message types during agent execution:
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                # The agent decided to call an MCP tool (e.g., to query k8s)
                # Provide feedback to the user that data is being retrieved
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Retrieving data from Kubernetes..."
                }
            elif isinstance(message, ToolMessage):
                # The MCP tool responded with the requested data
                # The agent is now processing the tool's response
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing Kubernetes data..."
                }
            elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                # The agent has completed processing and has a final response
                # No more tool calls, this is the final message for the user:
                final_response = message

        # At the end of the stream, send the final response
        if final_response:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": getattr(final_response, "content", "").strip()
            }
        else:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Unable to process the request at this time."
            }
