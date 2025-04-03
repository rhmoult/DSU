from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

class TextRequest(BaseModel):
    prompt: str
    max_length: int = 1024

app = FastAPI()

# Load generation model for /generate and for RAG
model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Shared pipeline for both endpoints
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Load RAG components for /rag
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_pii_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# /generate endpoint for standard text generation
@app.post("/generate")
async def generate_text(request: TextRequest):
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        max_time=15.0,
        num_beams=1,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        early_stopping=True
    )

    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}

# /rag endpoint for retrieval-augmented generation
@app.post("/rag")
async def rag_search(request: Request):
    data = await request.json()
    query = data.get("prompt")
    docs = vectorstore.similarity_search(query, k=5)
    result = qa_chain.run(input_documents=docs, question=query)
    return {"response": result}

# Note: /generate and /rag are completely separate. Data submitted to /generate is NOT retained or accessible by /rag.
# Only documents pre-loaded into the RAG vector store (e.g., from faiss_pii_index) are used in /rag responses.
# If data leakage is observed in /rag responses, it is due to the content retrieved from the RAG index and NOT because DeepSeek R1 memorized or leaked the data.

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000

