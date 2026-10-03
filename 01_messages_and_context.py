import os
from config import gemini_api_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env


# Initialize Gemini Chat Model
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 1. Individual Message Types & Context Stack
conversation = [
    SystemMessage(content="You are an expert AI architecture mentor who is concise and technical."),
    HumanMessage(content="Hi! Remember that my favorite database is PostgreSQL."),
    AIMessage(content="Understood! I have noted that PostgreSQL is your preferred database."),
    HumanMessage(content="What database did I mention as my favorite?") # Testing context retention
]

response = llm.invoke(conversation)
print(conversation[0])
print("=== 01. Messages & Context Output ===")
print(f"llm_content_only_response:{response}")
print(f"llm_full_response:{response.content}")