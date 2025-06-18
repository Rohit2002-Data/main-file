from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch, os, json

# Load environment variables
load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Initialize FastAPI app
app = FastAPI()

# ✅ Use small model for Render (under 512 MB)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cpu")
model.eval()

# Load API keys from keys.json
with open("keys.json", "r") as f:
    VALID_KEYS = json.load(f)

@app.get("/")
async def root():
    return {"message": "LLM API is running."}

@app.post("/generate/")
async def generate(request: Request):
    headers = request.headers
    api_key = headers.get("Authorization", "").replace("Bearer ", "")
    if api_key not in VALID_KEYS.values():
        raise HTTPException(status_code=403, detail="Invalid API Key")

    data = await request.json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return {"response": "Prompt is empty."}

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": result}
