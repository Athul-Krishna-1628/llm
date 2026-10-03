import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
#llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
#llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 1. Define Agent State using TypedDict
class AgentState(TypedDict):
    input_text: str
    processed_output: str

# 2. Define Node Functions
def uppercase_node(state: AgentState) -> AgentState:
    print("--- Running Uppercase Node ---")
    text = state["input_text"]
    return {"processed_output": text.upper()}

def llm_enrichment_node(state: AgentState) -> AgentState:
    print("--- Running LLM Enrichment Node ---")
    current_text = state["processed_output"]
    response = llm.invoke([HumanMessage(content=f"Summarize or comment on this uppercase text briefly: {current_text}")])
    return {"processed_output": response.content}

# 3. Build the StateGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("uppercase_node", uppercase_node)
workflow.add_node("llm_enrichment_node", llm_enrichment_node)

# Add edges to connect nodes sequentially
workflow.add_edge(START, "uppercase_node")
workflow.add_edge("uppercase_node", "llm_enrichment_node")
workflow.add_edge("llm_enrichment_node", END)

# Compile the graph into an executable app
app = workflow.compile()

# 4. Execute the workflow
initial_state = {"input_text": "langgraph makes building multi-step AI agents intuitive and modular.", "processed_output": ""}
final_result = app.invoke(initial_state)

print("\n=== 04. LangGraph Execution Output ===")
print(final_result["processed_output"])
