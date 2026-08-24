import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.rag_pipeline import PlantTreatmentRAG

print("🔄 Building ChromaDB...")
backend_dir = os.path.join(os.getcwd(), "backend")
rag = PlantTreatmentRAG(
    knowledge_base_path=os.path.join(backend_dir, "knowledge_base"),
    persist_directory=os.path.join(backend_dir, "chroma_db")
)
print("✅ ChromaDB ready.")
