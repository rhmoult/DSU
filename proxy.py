from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
import httpx
import uuid
from datetime import datetime
import os

app = FastAPI()
ollama_url = "http://localhost:11434/api/generate"

# Generate timestamped filename for timing logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"prompt_timings_{timestamp}.csv"

# Write header once
with open(log_filename, "w") as f:
    f.write("uuid,start_time,end_time,duration_seconds,prompt\n")

@app.post("/generate")
async def proxy_generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    prompt_id = str(uuid.uuid4())
    start_time = time.time()

    timeout_seconds = 120.0
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        ollama_response = await client.post(ollama_url, json=data)

    end_time = time.time()
    duration = end_time - start_time

    with open(log_filename, "a") as f:
        f.write(f"{prompt_id},{start_time},{end_time},{duration:.4f},\"{prompt.strip()}\"\n")

    return JSONResponse(content=ollama_response.json())

