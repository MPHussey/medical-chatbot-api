from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.rag import ask_question
import os
from core.config import CHROMA_PATH

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
def ask(request: QuestionRequest):
    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a document first.")
    
    answer = ask_question(request.question)
    return {"answer": answer}