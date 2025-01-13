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

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# os.environ["OPENAI_API_KEY"] = ""

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
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

    def _run(self, user_input: str) -> str:
        response = query_engine.query(user_input)

        return response

sale_agent = Agent(
    role = "Sale",
    goal = "Trả lời câu hỏi của người dùng",
    backstory = """
        Bạn là một người có kiến thức về quy đổi điểm ngoại ngữ sang bậc trình độ ngoại ngữ của Việt Nam.
    """,
    llm=llm,
)

search_task = Task(
    description='Phản hồi tin nhắn của người dùng: {user_message}',
    expected_output='Một câu trả lời phù hợp với câu hỏi của người dùng.',
    agent=sale_agent,
    tools=[KBSearchTool()]
)

crew = Crew(
    agents=[sale_agent],
    tasks=[search_task],
    manager_llm=llm,
    verbose=True
)

prompt = "Trình độ ngoại ngữ"

inputs = {
    "user_message": f"{prompt}",
}

response = crew.kickoff(inputs=inputs)

print("retriever_response: ", response)