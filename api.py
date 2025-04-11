from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import warnings
import time
import logging
from datetime import datetime
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

from nemoguardrails import LLMRails

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate a new log file with current date and time
log_filename = datetime.now().strftime("logs/latency_%Y%m%d_%H%M%S.log")

# Set up logging to the new file
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# FastAPI setup
app = FastAPI()

# Request model
class TextRequest(BaseModel):
    prompt: str
    enable_rag: bool = False  # RAG is disabled by default

# Global guardrails instance
rails = None

# Load generation model
model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Prevent "Sliding Window Attention is enabled" warning
if hasattr(model.config, "use_sliding_window"):
    model.config.use_sliding_window = False

# Build pipeline for HuggingFace
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Load FAISS vector store and embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_pii_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Load LangChain QA chain with deprecation warning suppressed
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qa_chain = load_qa_chain(llm, chain_type="stuff")

SIMILARITY_THRESHOLD = 0.7

# Load guardrails on startup
@app.on_event("startup")
async def startup_event():
    global rails
    rails = LLMRails.from_config("./guardrails")
    await rails.app_config.load()

# /rag endpoint: RAG + Guardrails
@app.post("/rag")
async def rag_smart_response(request: TextRequest):
    global rails
    query = request.prompt
    use_rag = request.enable_rag
    source = "unknown"
    start_time = time.time()

    try:
        # Guardrails input filter
        input_guard = await rails.guard_input(prompt=query)
        if input_guard.is_blocked:
            source = "guardrails_input_block"
            return JSONResponse(content={
                "response": input_guard.response,
                "source": source
            })

        # Attempt RAG if enabled
        if use_rag:
            results = vectorstore.similarity_search_with_score(query, k=5)
            docs = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]
            if docs:
                result = qa_chain.run(input_documents=docs, question=query)
                output_guard = await rails.guard_output(prompt=query, response=result)
                source = "retrieved_docs_guarded"
                return JSONResponse(content={
                    "response": output_guard.response,
                    "source": source
                })

        # Fallback to direct LLM generation
        inputs = tokenizer(
            query,
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
            early_stopping=False
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Guardrails output filter
        output_guard = await rails.guard_output(prompt=query, response=generated_text)
        source = "llm_only_guarded" if not use_rag else "llm_fallback_guarded"

        return JSONResponse(content={
            "response": output_guard.response,
            "source": source
        })

    finally:
        end_time = time.time()
        latency = round(end_time - start_time, 4)
        logging.info(f"Source: {source} | Latency: {latency} sec | Prompt: {query[:50]}...")

