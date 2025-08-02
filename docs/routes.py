from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from auth.routes import authenticate
from docs.vectorstore import load_vectorstore
import uuid

router = APIRouter()

@router.post("/upload_docs")
async def upload_docs(
    user=Depends(authenticate),
    file: UploadFile = File(...),
    role: str = Form(...)
):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admin can upload files")

    doc_id = str(uuid.uuid4())
    await load_vectorstore([file], role, doc_id)  # ✅ AWAIT here
    return {
        "message": f"{file.filename} uploaded successfully",
        "doc_id": doc_id,
        "accessible_to": role
    }
