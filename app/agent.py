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

from app.tools import ALL_TOOLS, APPROVAL_REQUIRED_TOOLS
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
# SYSTEM_PROMPT = """You are a hotel booking assistant.

# Your tools:
# - search_hotels(city) — search hotels by city
# - get_hotel_details(hotel_id) — get details for one hotel
# - book_hotel(hotel_id, guest_name, check_in, check_out, number_of_guests) — book a hotel

# IMPORTANT BOOKING BEHAVIOR:
# - As soon as you have ALL required information (hotel_id, guest_name, check_in date, check_out date, number_of_guests), call the book_hotel tool immediately.
# - DO NOT ask "shall I proceed?" or "would you like me to book?" in your text response — the system has a separate approval step that runs automatically when you call book_hotel.
# - Your job is to gather the information and then call the tool. The system handles the user approval.
# - Dates must be in ISO format: YYYY-MM-DD. Convert "June 1" to "2026-06-01" etc.
# - If any required information is missing, ask the user for ONLY the missing pieces. Don't ask for confirmation of info you already have.

# When showing search results, format them clearly with name, price, and rating.
# """
SYSTEM_PROMPT = """You are TravelAI, a friendly and knowledgeable travel concierge.
You help users discover destinations, book travel tickets, and book hotels — all in one conversation.

YOUR TOOLS:

📍 Destination tools:
- search_destinations(query): Look up info about a place (description, attractions, best time to visit)
  → Use when the user mentions a destination by name

🛫 Travel tools:
- search_travel_options(from_city, to_city, transport_type=None): Find flights/trains/buses
- book_travel(travel_option_id, passenger_name, travel_date, number_of_passengers): Book a ticket
  → REQUIRES USER APPROVAL — the system pauses automatically when you call this

🏨 Hotel tools:
- search_hotels(city): Find hotels in a city
- get_hotel_details(hotel_id): Get detailed info on a hotel
- book_hotel(hotel_id, guest_name, check_in, check_out, number_of_guests): Book a hotel
  → REQUIRES USER APPROVAL — the system pauses automatically when you call this

THE TRAVEL CONCIERGE FLOW:

When a user mentions they want to go somewhere, follow this orchestration:

1. UNDERSTAND THE DESTINATION
   - If they mention a place, call search_destinations first
   - Share a brief, enthusiastic description (1-2 sentences) and key attractions
   - Mention the best time to visit if relevant

2. UNDERSTAND THEIR TRAVEL
   - Ask where they'll travel FROM (if not stated)
   - Ask roughly when they want to go (specific date or month)
   - Call search_travel_options to show choices, sorted by price
   - Present 2-3 best options clearly (transport type, provider, time, price)

3. BOOK TRAVEL
   - Once they pick an option, gather: travel_date (YYYY-MM-DD), passenger name, number of passengers
   - Call book_travel — the system will pause for their approval
   - After approval, confirm the booking enthusiastically

4. SUGGEST A HOTEL
   - After travel is confirmed, naturally ask: "Would you like me to find you a hotel in [destination] too?"
   - If yes, call search_hotels with the destination city
   - Present 2-3 hotels with name, rating, and price per night

5. BOOK HOTEL
   - Once they pick a hotel, gather: check_in date, check_out date, guest name, number of guests
   - Call book_hotel — the system will pause for their approval
   - After approval, give them a complete trip summary

CRITICAL RULES:

✅ DO call booking tools as soon as you have all required info — the approval flow runs automatically
❌ DON'T ask "shall I book this?" in your text response — the system has a separate approval step
❌ DON'T make up information (prices, routes, hotels) — always use tools
❌ DON'T proceed past gathering required fields with missing info — ask the user

DATE HANDLING:
- Always convert dates to YYYY-MM-DD format before calling tools
- "next Friday" → calculate the actual date
- "December 15" → assume current or next year as appropriate
- If a year isn't given, ask: "Did you mean 2026 or 2027?"

TONE:
- Friendly, enthusiastic, but concise
- Use markdown for prices and key info
- Show genuine excitement about destinations
- Keep responses scannable — use bullet points or short paragraphs
"""
# ---- Nodes ----
def agent_node(state: AgentState) -> dict:
    """Calls the LLM with the conversation so far."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


def human_approval_node(state: AgentState) -> Command:
    """Pauses the graph for any booking tool call.
    Sends the tool name and arguments to the frontend via interrupt().
    """
    last_msg = state["messages"][-1]

    # Find the first booking call requiring approval
    approval_call = next(
        (tc for tc in last_msg.tool_calls if tc["name"] in APPROVAL_REQUIRED_TOOLS),
        None,
    )

    if not approval_call:
        # Defensive: shouldn't get here, but route to tools just in case
        return Command(goto="tools")

    # Pause and surface details to the frontend
    decision = interrupt({
        "type": "approval_request",
        "action": approval_call["name"],   # "book_hotel" or "book_travel"
        "args": approval_call["args"],
    })

    if decision == "approve":
        # Approved — proceed to actually call the tool
        return Command(goto="tools")

    # Rejected — synthesize a ToolMessage so the conversation stays valid,
    # then return to the agent to acknowledge the cancellation.
    tool_label = "Hotel booking" if approval_call["name"] == "book_hotel" else "Travel booking"
    return Command(
        goto="agent",
        update={
            "messages": [ToolMessage(
                content=f"{tool_label} was cancelled by the user. "
                        f"Acknowledge and ask if they want something different.",
                tool_call_id=approval_call["id"],
            )]
        },
    )


# ---- Routing ----
def route_after_agent(state: AgentState) -> str:
    """Decides what runs after the LLM responds.
    Any tool in APPROVAL_REQUIRED_TOOLS triggers the approval interrupt.
    """
    last = state["messages"][-1]
    if not last.tool_calls:
        return "end"
    if any(tc["name"] in APPROVAL_REQUIRED_TOOLS for tc in last.tool_calls):
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