import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from agent.src.bootstrap import bootstrap
from agent.src.core import (
    AgentConfig,
    SentinelAgent,
    create_agent,
    ConversationMessage,
)

async def main():
    print("=" * 60)
    print(" SENTINEL RAG AGENT - CLI MODE")
    print("=" * 60)
    
    # 1. Bootstrap Config
    config = bootstrap()
    
    agent_config = AgentConfig(
        model_name=os.environ.get("LLM_MODEL", "gpt-4-turbo-preview"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
    )
    
    tools_config = config.get("tools", [])
    
    # 2. Create Agent
    print("Initializing Agent...")
    try:
        agent = create_agent(
            tools_config=tools_config,
            agent_config=agent_config,
        )
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    print("Agent ready. Type 'exit' to quit.")
    print("-" * 60)
    
    conversation_history = []
    
    # 3. Chat Loop
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
                
            print("\nAgent: ...", end="", flush=True)
            
            response = await agent.invoke(
                message=user_input,
                conversation_history=conversation_history
            )
            
            # Clear "..." and print response
            print(f"\rAgent: {response.message}\n")
            
            # Print tool calls if any
            if response.tool_calls:
                print(f" [Tool Calls: {len(response.tool_calls)}]")
                for tc in response.tool_calls:
                    print(f"  - {tc['name']}: {tc['args']}")
            
            # Update history
            conversation_history.append(ConversationMessage(role="user", content=user_input))
            conversation_history.append(ConversationMessage(role="assistant", content=response.message))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

    print("\nGoodbye!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
