# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import get_service

API_KEY = os.getenv("API_KEY")  # optional simple auth

app = FastAPI(title="GPT-2 Next-Word API", version="1.0")

class GenerateIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    task_name: str = Field(..., description="Task type: 'spam' for classification, 'got' for text generation")
    max_new_tokens: int = Field(32, ge=1, le=64)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    stop: list[str] = []

class GenerateOut(BaseModel):
    completion: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate", response_model=GenerateOut)
def generate(body: GenerateIn, x_api_key: str | None = None):
    # tiny auth to prevent abuse
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    service = get_service(body.task_name)
    try:
        text = service.generate(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop=body.stop or None
        )
        return GenerateOut(completion=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 