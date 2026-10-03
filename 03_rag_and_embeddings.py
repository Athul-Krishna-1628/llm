import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# 1. Sample source text for RAG
raw_text = """
The Project LLM codebase uses uv for lightning-fast package management. 
Virtual environments are isolated in the 'llm' directory. 
Model Context Protocol (MCP) servers can be integrated with LangGraph to expand tool capabilities.
"""

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.create_documents([raw_text])

# 3. Create Local Embeddings using sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store chunks in ChromaDB vector store (in-memory)
vector_store = Chroma.from_documents(docs, embeddings)

# 5. Perform a similarity search (RAG retrieval step)
query = "How are virtual environments managed?"
results = vector_store.similarity_search(query, k=1)

print("=== 03. RAG Retrieval Output ===")
print(f"Query: {query}")
print(f"Retrieved Context: {results[0].page_content}")