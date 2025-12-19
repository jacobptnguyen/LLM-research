"""
Example async client that connects to the local MCP server via stdio and
pipes the exposed tools into a LangChain agent.
"""
import asyncio
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()
ROOT = Path(__file__).resolve().parent
SERVER_PATH = ROOT / "mcp_server.py"


async def main(query: str) -> None:
    config = {
        "startup": {
            "transport": "stdio",
            "command": "python",
            "args": [str(SERVER_PATH)],
        }
    }
    ##here this is the multi server mcp client configuration for the MCP server
    client = MultiServerMCPClient(config)
    ##here this is the session creation for the MCP server
    async with client.session("startup") as session:
        tools = await load_mcp_tools(session)

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise EnvironmentError("Set GROQ_API_KEY before running the client.")

        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-8b-instant",
            temperature=0,
        )

        agent = create_react_agent(model=llm, tools=tools)
        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a startup financial analyst. Think step-by-step, cite tool outputs, and keep answers under 400 words.",
                    },
                    {"role": "user", "content": query},
                ]
            }
        )
        # Extract the final message content (AIMessage object, not dict)
        final_message = response["messages"][-1]
        output_text = final_message.content if hasattr(final_message, "content") else str(final_message)
        print(output_text)


if __name__ == "__main__":
    asyncio.run(
        main("""I am planning to start an AI-agent startup and I want to understand whether my current financial situation supports this decision.
Here are my details:
• Current savings: ₹9,80,000
• Expected monthly burn (MVP + infra + salaries): ₹1,20,000
• Expected revenue in first 6 months: ₹0
• Estimated marketing spend: ₹40,000/month
• Time I can work without salary: 10 months
• Competition level: high

Given these details, tell me whether I am financially positioned to start this AI-agent startup, how long my runway will last, 
and whether the market scope justifies the financial risk.
""")
    )

