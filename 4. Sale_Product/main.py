import pandas as pd
import os
import chromadb
import sys
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from readers import parse_multiple_files
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
chroma_client = chromadb.PersistentClient(path="./DB/data")
chroma_collection = chroma_client.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
KB_index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, embed_model=embed_model, show_progress=True
)
query_engine = KB_index.as_query_engine()

class KBSearchTool(BaseTool):
    name: str = "KB Search tool"
    description: str = "Trả lời các câu hỏi liên quan đến công ty"

    def _run(self, user_input: str) -> str:
        response = query_engine.query(user_input)

        return response
    
class ProductSaleTool(BaseTool):
    name: str = "Product Sale tool"
    description: str = "Trả lời các câu hỏi liên quan đến sản phẩm"

    def _run(self, user_input: str) -> str:
        response = query_engine.query(user_input)

        return response
