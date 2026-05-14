import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.ingestor import ingest_document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunk_count = ingest_document(tmp_path)
    finally:
        os.unlink(tmp_path)  # clean up temp file

    return {"message": "Document ingested successfully", "chunks": chunk_count}