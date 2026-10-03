import os
import asyncio
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from operator import add
from config import gemini_api_key

# LangChain & Model Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Vector & Embeddings Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector # Conceptual / Standard Import for pgvector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# LangGraph Complexities Imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt
from langgraph.checkpoint.memory import MemorySaver

# 1. Environment Initialization

# Using the stable, active model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


print("============================================================")
print("PART 1: MODEL CONTEXT PROTOCOL (MCP) CLIENT & TOOL INTEGRATION")
print("============================================================\n")

# Simulated MCP Server Tool (Model Context Protocol standardizes how external tools expose capabilities)
@tool
def query_enterprise_mcp_server(query: str) -> str:
    """Queries an isolated Model Context Protocol (MCP) data provider server."""
    print(f"   [MCP Server Network Layer] Received remote tool execution request with query: '{query}'")
    # In production, this communicates via stdio or SSE (Server-Sent Events) with a separate MCP server process
    return "MCP Data Payload: Enterprise Security Standard v4.2 enforces token encryption."

# Bind MCP tool to LLM
mcp_tools = [query_enterprise_mcp_server]
llm_with_mcp = llm.bind_tools(mcp_tools)

mcp_response = llm_with_mcp.invoke("Fetch security guidelines from the enterprise MCP server.")
print(f"MCP Tool Call Intent Detected: {bool(mcp_response.tool_calls)}")
if mcp_response.tool_calls:
    tool_call = mcp_response.tool_calls[0]
    print(f"   -> Calling Tool: {tool_call['name']} with args: {tool_call['args']}")
    tool_result = query_enterprise_mcp_server.invoke(tool_call['args'])
    print(f"   -> Tool Output Result: {tool_result}\n")


print("============================================================")
print("PART 2: RAG ARCHITECTURES (Traditional vs Vectorless vs PgVector)")
print("============================================================\n")

corpus = [
    Document(page_content="PostgreSQL with pgvector allows lightning-fast approximate nearest neighbor (ANN) searches using HNSW indexes."),
    Document(page_content="Vectorless RAG relies on BM25 sparse keyword matching, regex filters, and direct SQL full-text search without embeddings."),
    Document(page_content="Traditional RAG uses chunking, dense embedding models like sentence-transformers, and cosine similarity stores.")
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# A. Traditional RAG / In-Memory Vector Search Flow
print("--- [2.1] Traditional Dense Vector RAG Simulation ---")
query_text = "How does pgvector work in database?"
query_vector = embeddings.embed_query(query_text)
print(f"Generated Embedding Vector Length: {len(query_vector)} dimensions")
print(f"Retrieved Context Match: '{corpus[0].page_content}'\n")

# B. PgVector Production Setup Blueprint (Code Pattern for Postgres + pgvector)
print("--- [2.2] PgVector Architecture Blueprint ---")
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5432/vector_db"
print(f"Targeting Postgres Connection: {CONNECTION_STRING}")
print("   -> Schema storage: Creates a table with a vector column (vector(384))")
print("   -> Indexing configuration: CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);")
print("   -> Execution: Performs native SQL distance calculations (<=> cosine distance operator) inside Postgres.\n")


print("============================================================")
print("PART 3: ADVANCED LANGGRAPH COMPLEXITIES (State, Fan-out, Interrupt)")
print("============================================================\n")

# Complex State Definition with Reducers (Annotated with `add` for list merging)
class AdvancedGraphState(TypedDict):
    task_id: str
    raw_input: str
    subtasks: List[str]
    retrieved_contexts: Annotated[List[str], add] # Reducer merges list updates instead of overwriting
    analysis_results: Annotated[List[str], add]
    final_report: str
    approval_status: str

# Node 1: Decomposer (Splits input into parallel subtasks)
def task_decomposer_node(state: AdvancedGraphState) -> Dict[str, Any]:
    print(f"[Node: Decomposer] Breaking down input: '{state['raw_input']}'")
    subtasks = [
        "Analyze security compliance",
        "Evaluate database performance scaling"
    ]
    return {"subtasks": subtasks}

# Map/Fan-out Worker Node (Executed in parallel via LangGraph 'Send')
def parallel_worker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    subtask = state["subtask"]
    print(f"   [Parallel Worker Node] Executing subtask worker for: '{subtask}'")
    simulated_finding = f"Report for [{subtask}]: Verified optimal metrics."
    return {"analysis_results": [simulated_finding]}

# Conditional routing function that uses Send for Fan-out
def fan_out_router(state: AdvancedGraphState):
    print("[Router] Dynamic Fan-out: Spawning parallel worker tasks...")
    # Send primitive dispatches state payloads to worker nodes in parallel
    return [Send("parallel_worker_node", {"subtask": st}) for st in state["subtasks"]]

# Node 2: Aggregator & Human-in-the-Loop Interrupt Gate
def aggregation_and_gate_node(state: AdvancedGraphState) -> Dict[str, Any]:
    print("[Node: Aggregator & Gate] Compiling parallel analysis results...")
    combined_summary = "\n".join(state["analysis_results"])
    
    # --- HUMAN-IN-THE-LOOP INTERRUPT ---
    # Pauses graph execution here, checkpoints state, and waits for external resume input
    print("   [INTERRUPT TRIGGERED] Pausing graph for human review and sign-off...")
    approval = interrupt({
        "prompt": "Review compiled analytics report before final database write.",
        "data": combined_summary
    })
    # Execution resumes here when caller passes Command(resume="approved")
    print(f"   [RESUMED] Received human command response: {approval}")
    
    return {
        "final_report": f"APPROVED REPORT:\n{combined_summary}",
        "approval_status": str(approval)
    }

# Build StateGraph
workflow = StateGraph(AdvancedGraphState)

workflow.add_node("task_decomposer_node", task_decomposer_node)
workflow.add_node("parallel_worker_node", parallel_worker_node)
workflow.add_node("aggregation_and_gate_node", aggregation_and_gate_node)

workflow.add_edge(START, "task_decomposer_node")
# Conditional Edge utilizing Send for parallel execution map
workflow.add_conditional_edges("task_decomposer_node", fan_out_router, ["parallel_worker_node"])
workflow.add_edge("parallel_worker_node", "aggregation_and_gate_node")
workflow.add_edge("aggregation_and_gate_node", END)

# Compile with MemorySaver Checkpointer to handle checkpoints and interruptions safely
memory_checkpointer = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory_checkpointer)


print("============================================================")
print("PART 4: DEEP STREAMS & GRAPH EXECUTION TRACE")
print("============================================================\n")

async def run_pipeline():
    thread_config = {"configurable": {"thread_id": "production_thread_99"}}
    initial_input = {
        "task_id": "T-100",
        "raw_input": "Run full infrastructure audit on vector search and MCP integration.",
        "subtasks": [],
        "retrieved_contexts": [],
        "analysis_results": [],
        "final_report": "",
        "approval_status": "pending"
    }

    print("--- Phase 1: Initiating graph execution up to the Human Interrupt ---")
    # Stream node updates live using stream mode 'updates'
    async for event in compiled_graph.astream(initial_input, thread_config, stream_mode="updates"):
        print(f"   [Live Stream Event Node Update]: {list(event.keys())}")

    # Inspect state at the interrupt pause point
    checkpoint_state = compiled_graph.get_state(thread_config)
    print(f"\nGraph Status Check: Interrupted = {bool(checkpoint_state.next)}")
    print(f"Interrupt Payload Received by Host: {checkpoint_state.tasks[0].interrupts[0].value}\n")

    print("--- Phase 2: Resuming graph execution with human approval ---")
    from langgraph.types import Command
    
    # Resume execution by passing Command with resume payload
    resume_command = Command(resume="Approved by Senior Architect via CLI")
    
    async for event in compiled_graph.astream(resume_command, thread_config, stream_mode="updates"):
        print(f"   [Live Stream Event Post-Resume]: {list(event.keys())}")

    # Fetch final graph state output
    final_state = compiled_graph.get_state(thread_config)
    print("\n============================================================")
    print("FINAL COMPILED STATE OUTPUT:")
    print("============================================================")
    print(f"Approval Status: {final_state.values.get('approval_status')}")
    print(f"Final Report:\n{final_state.values.get('final_report')}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())