"""
Agent Core

LangChain agent that sends POST requests to Interceptor.
Agent cannot communicate with MCP directly.
"""

import logging
import os
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from opentelemetry import trace
from pydantic import BaseModel, Field

from agent.config.prompts import get_system_prompt_with_context
from sentinel_sdk import SentinelClient
from .callbacks import SentinelCallbackHandler, TelemetryCallbackHandler

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class AgentConfig(BaseModel):
    """Configuration for the Sentinel Agent."""
    
    model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="LLM model to use (OpenRouter compatible)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in response",
    )
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="Maximum agent iterations",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter API key (falls back to env var)",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class AgentResponse(BaseModel):
    """Response from the agent."""
    
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="Agent response message")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools that were invoked",
    )


class SentinelAgent:
    """
    RAG Agent that sends POST requests to Interceptor.
    
    Agent cannot communicate with MCP directly. All tool calls go through Interceptor.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tools_config: list[dict[str, Any]],
    ) -> None:
        self.config = config
        self.tools_config = tools_config
        
        api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required but not set")
        
        self._llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=api_key,
            openai_api_base=config.openrouter_base_url,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/sentinel-rag"),
                "X-Title": "Sentinel RAG Agent",
            },
        )
        
        # Map user/chat session_id -> SentinelClient
        self._sessions: dict[str, SentinelClient] = {}
        # Map user/chat session_id -> LangChain Agent
        self._agents: dict[str, Any] = {}
    
    async def _get_or_create_session(self, session_id: str) -> tuple[Any, SentinelClient]:
        """Get or create a Sentinel session and LangChain agent for the given ID."""
        if session_id in self._sessions:
            return self._agents[session_id], self._sessions[session_id]
            
        # Initialize new Sentinel Client
        sentinel_api_key = os.getenv("SENTINEL_API_KEY") # Ensure this is set in .env
        sentinel_url = os.getenv("SENTINEL_URL", "http://localhost:8000")
        
        if not sentinel_api_key:
            logger.warning("SENTINEL_API_KEY not set. Using mock/default if available or failing.")
            # Depending on strictness, we might raise error. 
            # For template usage, let's assume valid key is provided or we error out.
        
        client = SentinelClient(api_key=sentinel_api_key, base_url=sentinel_url)
        await client.start_session()
        
        # Get Tools (synchronous proxy)
        tools = await client.get_langchain_tools()
        
        logger.info("Created Sentinel session %s for user session %s with %d tools", 
                   client.session_id, session_id, len(tools))

        agent = create_react_agent(
            model=self._llm,
            tools=tools,
        )
        
        self._sessions[session_id] = client
        self._agents[session_id] = agent
        
        return agent, client
    
    async def invoke(
        self,
        message: str,
        session_id: str | None = None,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> AgentResponse:
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        with tracer.start_as_current_span("agent.invoke") as span:
            span.set_attribute("session.id", session_id)
            
            callbacks = [
                SentinelCallbackHandler(session_id),
                TelemetryCallbackHandler(session_id),
            ]
            
            if session_id is None:
                session_id = str(uuid.uuid4())

            agent, client = await self._get_or_create_session(session_id)
            
            # Use the Sentinel Session ID for logging context, but keep user session ID for tracking
            span.set_attribute("sentinel.session_id", client.session_id)
            
            messages: list[Any] = []
            
            system_prompt = get_system_prompt_with_context(session_id)
            messages.append(SystemMessage(content=system_prompt))
            
            if conversation_history:
                for hist_msg in conversation_history:
                    if hist_msg.role == "user":
                        messages.append(HumanMessage(content=hist_msg.content))
                    elif hist_msg.role == "assistant":
                        messages.append(AIMessage(content=hist_msg.content))
                    elif hist_msg.role == "system":
                        messages.append(SystemMessage(content=hist_msg.content))
            
            messages.append(HumanMessage(content=message))
            
            logger.info(
                "Agent invocation: session=%s, message_length=%d",
                session_id,
                len(message),
            )
            
            try:
                logger.info("Invoking LangGraph agent with %d messages", len(messages))
                logger.debug("Messages: %s", [f"{type(m).__name__}: {str(m.content)[:100]}" for m in messages])
                
                result = await agent.ainvoke(
                    {"messages": messages},
                    config={"callbacks": callbacks},
                )
                
                logger.info("Agent result received: type=%s, keys=%s", type(result), list(result.keys()) if isinstance(result, dict) else "N/A")
                
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict result, got {type(result)}")
                
                response_messages = result.get("messages", [])
                logger.info("Response messages count: %d", len(response_messages))
                
                # Log all message types
                for i, msg in enumerate(response_messages):
                    msg_type = type(msg).__name__
                    content_preview = str(msg.content)[:100] if hasattr(msg, 'content') and msg.content else "No content"
                    logger.info("Message %d: type=%s, content=%s", i, msg_type, content_preview)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        logger.info("  Tool calls: %s", msg.tool_calls)
                
                final_message = ""
                tool_calls: list[dict[str, Any]] = []
                
                if not response_messages:
                    logger.warning("No messages in agent result")
                    final_message = "No response generated"
                else:
                    ai_messages = [msg for msg in response_messages if isinstance(msg, AIMessage)]
                    tool_messages = [msg for msg in response_messages if isinstance(msg, ToolMessage)]
                    
                    logger.info("Found %d AIMessages, %d ToolMessages", len(ai_messages), len(tool_messages))
                    
                    last_ai_message = None
                    if ai_messages:
                        last_ai_message = ai_messages[-1]
                    
                    if last_ai_message:
                        if last_ai_message.content:
                            final_message = str(last_ai_message.content)
                        
                        for ai_msg in ai_messages:
                            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                                logger.info("Found tool_calls in AIMessage: %d calls", len(ai_msg.tool_calls))
                                for tc in ai_msg.tool_calls:
                                    tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                                    tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                    logger.info("Tool call: name=%s, args=%s", tool_name, tool_args)
                                    tool_calls.append({
                                        "name": tool_name,
                                        "args": tool_args,
                                    })
                        
                        if tool_calls and not final_message:
                            final_message = "Processing tool calls..."
                    else:
                        logger.warning("No AIMessage found in response")
                        # Log all message types for debugging
                        for i, msg in enumerate(response_messages):
                            logger.debug("Message %d: type=%s", i, type(msg).__name__)
                        for msg in reversed(response_messages):
                            if hasattr(msg, "content") and msg.content:
                                final_message = str(msg.content)
                                break
                
                logger.info(
                    "Agent completed: session=%s, tool_calls=%d",
                    session_id,
                    len(tool_calls),
                )
                
                return AgentResponse(
                    session_id=session_id,
                    message=final_message,
                    tool_calls=tool_calls,
                )
            
            except Exception as e:
                logger.error(
                    "Agent error: session=%s, error=%s",
                    session_id,
                    str(e),
                    exc_info=True,
                )
                span.record_exception(e)
                raise


def create_agent(
    tools_config: list[dict[str, Any]],
    agent_config: AgentConfig | None = None,
) -> SentinelAgent:
    if agent_config is None:
        agent_config = AgentConfig()
    
    agent = SentinelAgent(
        config=agent_config,
        tools_config=tools_config,
    )
    
    logger.info(
        "Agent created: model=%s, tools=%d",
        agent_config.model_name,
        len(tools_config),
    )
    
    return agent

