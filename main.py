from fastapi import FastAPI, Request, HTTPException
import requests
import json

app = FastAPI()

LLM_API_URL = "https://llm-model-2-p5no.onrender.com/generate/"

# Load API keys from file
with open("keys.json", "r") as f:
    VALID_KEYS = json.load(f)

@app.get("/")
def home():
    return {"message": "LLM API Proxy is running."}

@app.post("/generate/")
async def proxy_generate(request: Request):
    # Extract API key from Authorization header
    headers = request.headers
    api_key = headers.get("Authorization", "").replace("Bearer ", "")
    
    if api_key not in VALID_KEYS.values():
        raise HTTPException(status_code=403, detail="Invalid API Key")

    # Extract prompt from JSON body
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Forward to LLM model API
    try:
        response = requests.post(LLM_API_URL, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {str(e)}")
