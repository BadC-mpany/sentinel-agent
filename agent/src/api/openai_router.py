import logging
import time
import uuid
from typing import Any, List, Optional, Union, Dict

from fastapi import APIRouter, HTTPException, Request, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import json

from agent.src.core import ConversationMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

# --- OpenAI Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "sentinel-rag"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class ChatCompletionChoiceMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# --- Helper Functions ---

def get_agent(request: Request):
    """Dependency to retrieve the initialized agent from app state."""
    agent = getattr(request.app.state, "agent", None)
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent

security_scheme = HTTPBearer()

def verify_api_key(request: Request, credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """Verify Bearer token against the generated API key."""
    expected_key = getattr(request.app.state, "api_key", None)
    
    # If no key is set in app state (shouldn't happen in production flow), fail open or secure?
    # Requirement is generated key, so we fail if not matching.
    if not expected_key:
        logger.warning("No API key configured in app state, rejecting request")
        raise HTTPException(status_code=500, detail="Server misconfiguration: API key not set")
        
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return credentials.credentials

# --- Endpoints ---

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    agent: Any = Depends(get_agent),
    api_key: str = Depends(verify_api_key)
):
    """
    OpenAI-compatible chat completion endpoint.
    """
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    logger.info("OpenAI Chat Request: %d messages, model=%s", len(request.messages), request.model)
    
    # 1. Convert OpenAI messages to internal ConversationMessage format
    # We assume the last message is the "user" query, and the rest are history.
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    last_message = request.messages[-1]
    if last_message.role != "user":
        # Some clients might send system prompt as last, but typically user msg is last.
        # Ideally we find the last user message.
        pass

    user_query = last_message.content
    
    # Extract history if any
    conversation_history = []
    for msg in request.messages[:-1]:
        conversation_history.append(ConversationMessage(
            role=msg.role,
            content=msg.content
        ))
    
    # Session management
    # OpenAI API is stateless, but our agent is stateful per session.
    # We can use the 'user' field as session_id if provided, otherwise generate one.
    session_id = request.user or str(uuid.uuid4())
    
    try:
        # 2. Invoke the Agent (Buffered execution for now, streaming simulation)
        response = await agent.invoke(
            message=user_query,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        if request.stream:
            async def event_generator():
                chunk_id = f"chatcmpl-{uuid.uuid4()}"
                
                # 1. Initial Role Chunk
                role_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"
                
                # 2. Content Chunks (Split by token/word approximation)
                import asyncio
                words = response.message.split(" ")
                for i, word in enumerate(words):
                    text = word + (" " if i < len(words) - 1 else "")
                    content_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n"
                    # Small sleep to ensure chunks are sent distinctively if needed, 
                    # though yield usually handles it.
                    await asyncio.sleep(0.005)
                
                # 3. Finish Chunk
                finish_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
            
        else:
            # 3. Construct Standard OpenAI Response
            choice = ChatCompletionChoice(
                index=0,
                message=ChatCompletionChoiceMessage(
                    role="assistant",
                    content=response.message
                ),
                finish_reason="stop"
            )
            
            return ChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[choice],
                usage=UsageInfo(
                    prompt_tokens=0, 
                    completion_tokens=0,
                    total_tokens=0
                )
            )
        
    except Exception as e:
        logger.error("Error processing OpenAI request: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List available models (mocked for compatibility)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "sentinel-rag",
                "object": "model",
                "created": 1677610602,
                "owned_by": "sentinel"
            }
        ]
    }
