import os
import chromadb
import sys
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from readers import parse_multiple_files
from dotenv import load_dotenv

load_dotenv()

# Set the OPENAI_API_KEY environment variable to your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_data(source_dir):
    return parse_multiple_files(source_dir)


def ingest_data(data_dir="./RAG_tools/data/"):
    documents = load_data(data_dir)
    
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    
    chroma_client = chromadb.PersistentClient(path="./DB/data")
    chroma_collection = chroma_client.get_or_create_collection("test")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
    )
    
if __name__ == "__main__":
    ingest_data()