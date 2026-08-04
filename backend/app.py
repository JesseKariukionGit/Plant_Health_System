import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from classifier import PlantDiseaseClassifier
from rag_pipeline import PlantTreatmentRAG
from database import log_diagnosis, log_treatment, get_diagnosis_history, get_treatment_by_diagnosis, delete_old_records

app = FastAPI(title="Plant Health Diagnosis and Recommendation System", version="1.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run cleanup on startup
@app.on_event("startup")
async def startup_event():
    delete_old_records(24)

print("Loading classifier...")
classifier = PlantDiseaseClassifier(
    model_path="../models/efficientnetb0_plantvillage.h5",
    class_names_path="../models/class_names.json"
)

print("Loading RAG pipeline...")
rag = PlantTreatmentRAG(
    knowledge_base_path="./knowledge_base",
    persist_directory="./chroma_db"
)

CONFIDENCE_THRESHOLD = 0.70

class DiagnosisResponse(BaseModel):
    success: bool
    disease: Optional[str] = None
    confidence: Optional[float] = None
    treatment_organic: Optional[str] = None
    treatment_chemical: Optional[str] = None
    treatment_prevention: Optional[str] = None
    message: Optional[str] = None
    disclaimer: Optional[str] = None

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name
    
    try:
        disease_name, confidence, all_probs = classifier.predict(temp_path)
        diagnosis_id = log_diagnosis(disease_name, confidence)
        
        if confidence < CONFIDENCE_THRESHOLD:
            return DiagnosisResponse(
                success=False,
                disease=disease_name,
                confidence=confidence,
                message=f"Low confidence ({confidence:.2%}). Unable to diagnose.",
                disclaimer="AI suggestion only. Not a substitute for professional advice."
            )
        
        question = f"What is the treatment for {disease_name}?"
        treatment_sections = rag.query(question)
        
        log_treatment(diagnosis_id, [treatment_sections.get('organic', '')], treatment_sections.get('organic', ''), safety_flag=0)
        
        disclaimer = "⚠️ AI suggestion only. Confirm with a local agronomist before applying any treatment, especially on edible plants."
        
        return DiagnosisResponse(
            success=True,
            disease=disease_name,
            confidence=confidence,
            treatment_organic=treatment_sections.get('organic', 'No organic treatment found.'),
            treatment_chemical=treatment_sections.get('chemical', 'No chemical treatment found.'),
            treatment_prevention=treatment_sections.get('prevention', 'No prevention information found.'),
            message="Diagnosis completed successfully.",
            disclaimer=disclaimer
        )
    
    except Exception as e:
        return DiagnosisResponse(
            success=False,
            message=f"Error processing image: {str(e)}"
        )
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/history")
def get_history(page: int = 1, limit: int = 7):
    """Get diagnosis history with pagination and auto-cleanup"""
    # Clean up old records before fetching
    delete_old_records(24)
    offset = (page - 1) * limit
    rows, total = get_diagnosis_history(limit, offset)
    history = []
    for record in rows:
        log_id, timestamp, disease_predicted, confidence, actual_disease = record
        treatment = get_treatment_by_diagnosis(log_id)
        history.append({
            'log_id': log_id,
            'timestamp': timestamp,
            'disease': disease_predicted,
            'confidence': confidence,
            'actual_disease': actual_disease,
            'treatment': treatment
        })
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    return {
        "success": True,
        "history": history,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages
    }

@app.get("/history/treatment/{diagnosis_id}")
def get_treatment(diagnosis_id: int):
    treatment = get_treatment_by_diagnosis(diagnosis_id)
    return {"success": True, "treatment": treatment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)