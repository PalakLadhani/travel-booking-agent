"""
LangGraph agent for hotel booking with human-in-the-loop approval.
"""
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from app.tools import ALL_TOOLS

# ---- State ----
class AgentState(TypedDict):
    """
    add_messages is a reducer — it appends to the list rather than replacing.
    Without it, every node would overwrite the full message history.
    """
    messages: Annotated[list, add_messages]


# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

# SYSTEM_PROMPT = """You are a friendly hotel booking assistant.

# Your capabilities:
# - Search hotels by city using search_hotels
# - Get hotel details using get_hotel_details
# - Book a hotel using book_hotel (requires user approval — they will be asked)

# Guidelines:
# - Always confirm guest name and dates before calling book_hotel
# - Show prices clearly and the total cost (price per night × number of nights)
# - Be concise but warm
# """
SYSTEM_PROMPT = """You are a hotel booking assistant.

Your tools:
- search_hotels(city) — search hotels by city
- get_hotel_details(hotel_id) — get details for one hotel
- book_hotel(hotel_id, guest_name, check_in, check_out, number_of_guests) — book a hotel

IMPORTANT BOOKING BEHAVIOR:
- As soon as you have ALL required information (hotel_id, guest_name, check_in date, check_out date, number_of_guests), call the book_hotel tool immediately.
- DO NOT ask "shall I proceed?" or "would you like me to book?" in your text response — the system has a separate approval step that runs automatically when you call book_hotel.
- Your job is to gather the information and then call the tool. The system handles the user approval.
- Dates must be in ISO format: YYYY-MM-DD. Convert "June 1" to "2026-06-01" etc.
- If any required information is missing, ask the user for ONLY the missing pieces. Don't ask for confirmation of info you already have.

When showing search results, format them clearly with name, price, and rating.
"""

# ---- Nodes ----
def agent_node(state: AgentState) -> dict:
    """Calls the LLM with the conversation so far."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


def human_approval_node(state: AgentState) -> Command:
    """
    Pauses the graph if the LLM wants to book a hotel.
    Resumes based on the human's decision.
    """
    last_msg = state["messages"][-1]
    booking_call = next(
        (tc for tc in last_msg.tool_calls if tc["name"] == "book_hotel"),
        None,
    )

    # Defensive: shouldn't reach this node without a booking call,
    # but if we do, just continue.
    if not booking_call:
        return Command(goto="tools")

    # interrupt() pauses execution. The dict is sent to the frontend
    # so the user can see what they're approving.
    decision = interrupt({
        "type": "approval_request",
        "action": "book_hotel",
        "args": booking_call["args"],
    })

    if decision == "approve":
        return Command(goto="tools")

    # Rejection: tell the LLM and let it respond to the user
    return Command(
        goto="agent",
        update={
            "messages": [ToolMessage(
                content="Booking was cancelled by the user. Acknowledge and ask if they want something different.",
                tool_call_id=booking_call["id"],
            )]
        },
    )


# ---- Routing ----
def route_after_agent(state: AgentState) -> str:
    """Decides what runs after the LLM responds."""
    last = state["messages"][-1]
    if not last.tool_calls:
        return "end"
    if any(tc["name"] == "book_hotel" for tc in last.tool_calls):
        return "human_approval"
    return "tools"


# ---- Build graph ----
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("human_approval", human_approval_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "human_approval": "human_approval",
            "end": END,
        },
    )
    builder.add_edge("tools", "agent")
    # human_approval uses Command(goto=...) so no static edge needed

    # MemorySaver = in-memory checkpointer.
    # Replace with SqliteSaver/PostgresSaver in production.
    return builder.compile(checkpointer=MemorySaver())


# Single graph instance shared across requests
graph = build_graph()