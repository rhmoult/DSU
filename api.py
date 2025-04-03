from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 1024

app = FastAPI()

model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

@app.post("/generate")
async def generate_text(request: TextRequest):
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512  # Limit input length
    ).to("cuda")
    
    # More aggressive generation parameters
    output = model.generate(
        **inputs,
        max_new_tokens=256,    # Generate up to 256 new tokens
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,        # Higher temperature for faster (though less precise) generation
        max_time=15.0,         # Reduced time limit
        num_beams=1,           # Greedy search
        do_sample=True,        # Enable sampling
        top_k=50,              # Limit vocabulary choices
        top_p=0.9,             # Nucleus sampling
        early_stopping=True    # Stop when complete
    )
    
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000

