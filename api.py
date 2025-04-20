import logging
import os
import time
import warnings
from datetime import datetime

import torch
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# FastAPI setup
app = FastAPI()

# Set up logging
os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/latency_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Request model
class TextRequest(BaseModel):
    prompt: str
    enable_rag: bool = False  # Ignored due to ABAC


# ABAC Policy function
def abac_policy(role: str, resource: str, action: str) -> bool:
    if role == "admin" and resource == "/rag" and action == "post":
        return True
    return False


# Load generation model
model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)

if hasattr(model.config, "use_sliding_window"):
    model.config.use_sliding_window = False

# Build pipeline
qa_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Load FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_pii_index", embeddings=embedding_model, allow_dangerous_deserialization=True
)

# Load QA chain
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qa_chain = load_qa_chain(llm, chain_type="stuff")

SIMILARITY_THRESHOLD = 0.7


# Endpoint
@app.post("/rag")
async def rag_smart_response(
    request: TextRequest, role: str = Header(default="unverified")
):
    query = request.prompt
    start_time = time.time()
    source = "unknown"

    try:
        enable_rag = abac_policy(role, "/rag", "post")

        if enable_rag:
            results = vectorstore.similarity_search_with_score(query, k=5)
            docs = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]
            if docs:
                result = qa_chain.run(input_documents=docs, question=query)
                source = "retrieved_docs"
                return JSONResponse(
                    content={"response": result, "source": source, "mode": role}
                )

        # Fallback to LLM
        inputs = tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512
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
            early_stopping=False,
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        source = "llm_only" if not enable_rag else "llm_fallback"
        return JSONResponse(
            content={"response": generated_text, "source": source, "mode": role}
        )

    finally:
        latency = round(time.time() - start_time, 4)
        logging.info(
            f"Role: {role} | Source: {source} | Latency: {latency}s | Prompt: {query[:50]}..."
        )
