from fastapi import FastAPI, Request, HTTPException
import requests
import json

app = FastAPI()

LLM_API_URL = "https://llm-model-2-p5no.onrender.com//generate/"

# Load API keys from local file
with open("keys.json", "r") as f:
    VALID_KEYS = json.load(f)

@app.post("/generate/")
async def proxy_generate(request: Request):
    headers = request.headers
    api_key = headers.get("Authorization", "").replace("Bearer ", "")
    if api_key not in VALID_KEYS.values():
        raise HTTPException(status_code=403, detail="Invalid API Key")

    data = await request.json()
    prompt = data.get("prompt", "")

    response = requests.post(LLM_API_URL, json={"prompt": prompt})
    return response.json()
