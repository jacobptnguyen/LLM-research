import os
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rag_tools import semantic_retriever, news_lookup
from dotenv import load_dotenv
load_dotenv()
##here we are creating the fast mcp app with the name startup-analyst
app = FastMCP(name="startup-analyst")

##shared LangChain tool definitions that the agent can call
@tool
def semantic_search(query: str) -> str:
    """Search for semantic evidence about competitors, funding, and benchmarks."""
    hits = semantic_retriever(query)
    return "\n".join(hits) if hits else "No supporting passages found."

@tool
def news_search_tool(query: str) -> str:
    """Fetch recent news headlines about a startup, market, or technology."""
    stories = news_lookup(query)
    return "\n\n".join(stories) if stories else "No recent news found."

##load the Groq LLM from env credentials
try:
    GROQ_KEY = os.environ.get("GROQ_API_KEY")
except KeyError as exc:
    raise EnvironmentError("Set GROQ_API_KEY before starting the MCP server.") from exc

llm = ChatGroq(
    groq_api_key=GROQ_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0,
)
##here this is the react agent creation with the model and the tools
agent = create_react_agent(model=llm, tools=[semantic_search, news_search_tool])

##here this is the text payload function to return the text payload
##It's just a helper function to convert normal Python text into the format MCP requires.
def _text_payload(lines: list[str], empty_message: str) -> CallToolResult:
    text = "\n".join(lines) if lines else empty_message
    return CallToolResult(content=[TextContent(type="text", text=text)])

##this is the vector tool used for the MCP clients to use the semantic search tool
@app.tool(name="vector_search", description="Raw semantic matches from the Chroma store.")
def vector_search(query: str) -> CallToolResult:
    hits = semantic_retriever(query)
    return _text_payload(hits, "No supporting passages found.")

##this is the news tool used for the client to use the news search tool
@app.tool(name="news_search", description="Latest NewsAPI headlines for the topic.")
def news_search(query: str) -> CallToolResult:
    stories = news_lookup(query)
    return _text_payload(stories, "No recent news found.")


##here we are creating the startup_report tool with the name startup_report
##here this is the tool which is responsible to feed the query to the react agent and return the output
@app.tool(name="startup_report", description="Answer user questions using ReAct + tools.")
def startup_report(query: str) -> CallToolResult:
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a startup financialanalyst. Use the tools provided, cite evidence, and keep answers under 400 words.",
                },
                {"role": "user", "content": query},
            ]
        }
    )
    # Extract the final message content (AIMessage object, not dict)
    final_message = response["messages"][-1]
    # AIMessage has a .content attribute
    output_text = final_message.content if hasattr(final_message, "content") else str(final_message)
    return CallToolResult(
        content=[TextContent(type="text", text=output_text)]
    )


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http": ##here this is the streamable http transport for the MCP server
        app.run_http()
    else:
        app.run()

