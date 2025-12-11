"""Sentinel RAG Agent - FastAPI entry point."""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

load_dotenv()

from agent.src.bootstrap import bootstrap, setup_logging
from agent.src.core import (
    SentinelAgent,
    AgentConfig,
    AgentResponse,
    ConversationMessage,
    create_agent,
)
from agent.src.api.openai_router import router as openai_router

logger = logging.getLogger(__name__)

_agent: SentinelAgent | None = None
_config: dict | None = None

# Store pending tool results by session_id + tool_name
# Removed ToolResultRequest



class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    conversation_history: list[ConversationMessage] | None = Field(None, description="Previous conversation messages")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="Agent response")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tools that were invoked")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _config
    
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("Starting Sentinel RAG Agent...")
    
    try:
        _config = bootstrap()
        
        agent_config = AgentConfig(
            model_name=os.environ.get("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "10")),
            verbose=os.environ.get("AGENT_VERBOSE", "false").lower() == "true",
        )
        
        tools_config = _config.get("tools", [])
        
        _agent = create_agent(
            tools_config=tools_config,
            agent_config=agent_config,
        )
        
        # pending results store removed

        
        # Expose agent in app state for routers (e.g. OpenAI router)
        app.state.agent = _agent
        
        # Load API Key for auth check
        app.state.api_key = os.environ.get("ANAM_API_KEY")
        
        logger.info("Agent initialized: model=%s, tools=%d", agent_config.model_name, len(tools_config))
        
        yield
        
    except Exception as e:
        logger.error("Startup failed: %s", str(e))
        raise
    finally:
        logger.info("Sentinel RAG Agent shutdown complete")


app = FastAPI(
    title="Sentinel RAG Agent",
    description="Secure RAG client implementing the Blind Courier pattern in a Zero Trust architecture.",
    version="1.0.0",
    lifespan=lifespan,
)

# API Mode Configuration
api_mode = os.environ.get("API_MODE", "legacy")
logger.info("Configured API Mode: %s", api_mode)

if api_mode.lower() == "openai":
    logger.info("Mounting OpenAI-compatible API routes at /v1")
    app.include_router(openai_router)
else:
    logger.info("Running in Legacy Mode (Standard Routes only)")

cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_session(x_session_id: str | None = Header(None, alias="X-Session-ID")) -> str | None:
    return x_session_id


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if _agent else "degraded",
        service_name=_config.get("service_name", "sentinel-rag-agent") if _config else "sentinel-rag-agent",
        version=_config.get("version", "unknown") if _config else "unknown",
    )


# Removed /tool-result and debug endpoints



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str | None = Depends(verify_session)) -> ChatResponse:
    if not _agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    effective_session_id = session_id or request.session_id or str(uuid.uuid4())
    
    logger.info("=" * 60)
    logger.info("Chat request: session=%s, message_length=%d", effective_session_id, len(request.message))
    logger.info("Message: %s", request.message)
    logger.info("=" * 60)
    
    try:
        response = await _agent.invoke(
            message=request.message,
            session_id=effective_session_id,
            conversation_history=request.conversation_history,
        )
        
        logger.info("=" * 60)
        logger.info("Chat response: session=%s, message_length=%d, tool_calls=%d", 
                   effective_session_id, len(response.message), len(response.tool_calls))
        logger.info("Response message: %s", response.message[:200])
        logger.info("Tool calls: %s", response.tool_calls)
        logger.info("=" * 60)
        
        return ChatResponse(
            session_id=response.session_id,
            response=response.message,
            tool_calls=response.tool_calls,
        )
    
    except Exception as e:
        import traceback
        logger.error("=" * 60)
        logger.error("Chat error: session=%s, error=%s", effective_session_id, str(e))
        logger.error("Traceback:\n%s", traceback.format_exc())
        logger.error("=" * 60)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentinel RAG Agent")
    parser.add_argument("--api-mode", default=os.environ.get("API_MODE", "legacy"), 
                       choices=["legacy", "openai"],
                       help="API Mode: legacy (default) or openai")
    parser.add_argument("--expose", action="store_true", 
                       help="Expose the agent via ngrok (requires pyngrok)")
    
    # Parse known args to avoid conflict with uvicorn args if any (though we run uvicorn programmatically)
    args, unknown = parser.parse_known_args()
    
    # Set env var so it persists when uvicorn re-imports the app
    os.environ["API_MODE"] = args.api_mode
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", "8001"))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    # Handle OpenAI Mode Setup (Key Gen + Tunnel)
    if args.api_mode == "openai":
        import secrets
        
        # 1. Generate API Key if not set
        api_key = os.environ.get("ANAM_API_KEY")
        if not api_key:
            api_key = secrets.token_urlsafe(32)
            os.environ["ANAM_API_KEY"] = api_key
            
        # 2. Handle Tunneling
        public_url = f"http://{host}:{port}"
        if args.expose:
            try:
                from pyngrok import ngrok
                # Connect to the port
                public_url = ngrok.connect(port).public_url
                print(f" [INFO] ngrok tunnel establishing... {public_url}")
            except ImportError:
                print(" [WARN] pyngrok not installed. Install with `pip install pyngrok`")
            except Exception as e:
                print(f" [ERROR] Failed to start ngrok: {e}")

        # 3. Print Connection Info
        print("\n" + "="*60)
        print(" SENTINEL RAG AGENT - ANAM CONNECTION INFO")
        print("="*60)
        print(f" Base URL : {public_url}")
        print(f" API URL  : {public_url}/v1")
        print(f" API Key  : {api_key}")
        print("="*60)
        print(" Copy the above API URL and API Key to Anam to connect.")
        print("="*60 + "\n")
    
    uvicorn.run(
        "agent.src.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()

