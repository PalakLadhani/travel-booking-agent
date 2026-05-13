"""FastAPI app exposing the LangGraph agent over HTTP."""
import uuid
from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.agent import graph
from app.schemas import ChatRequest, ResumeRequest, ChatResponse

app = FastAPI(title="Travel Agent Service", version="0.1.0")

# Allow CAP (and later Fiori UI) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _last_ai_text(messages: list) -> str:
    """Find the most recent assistant text message."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            # content can be string or list of blocks — normalize
            if isinstance(m.content, str):
                return m.content
            return " ".join(
                block.get("text", "") for block in m.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return ""


def _build_response(thread_id: str, result: dict, config: RunnableConfig) -> ChatResponse:
    """Inspect graph state and build a structured response."""
    state = graph.get_state(config)
    interrupted = bool(state.tasks and state.tasks[0].interrupts)
    interrupt_data = None

    if interrupted:
        interrupt_data = state.tasks[0].interrupts[0].value
        reply = "Awaiting your approval to proceed."
    else:
        reply = _last_ai_text(result.get("messages", []))

    return ChatResponse(
        thread_id=thread_id,
        reply=reply,
        interrupted=interrupted,
        interrupt_data=interrupt_data,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message to the agent. Creates a new thread if none provided."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config: RunnableConfig = {
    "configurable": {"thread_id": thread_id}
            }

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return _build_response(thread_id, result, config)


@app.post("/resume", response_model=ChatResponse)
def resume(req: ResumeRequest):
    """Resume a paused thread after human approval/rejection."""
    if req.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")

    config = {"configurable": {"thread_id": req.thread_id}}

    # Verify the thread exists and is actually interrupted
    state = graph.get_state(config)
    if not state.tasks or not state.tasks[0].interrupts:
        raise HTTPException(status_code=400, detail="Thread is not awaiting approval")

    try:
        result = graph.invoke(Command(resume=req.decision), config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume error: {e}")

    return _build_response(req.thread_id, result, config)