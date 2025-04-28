import logging
import os
import time
import warnings
from datetime import datetime

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from nemoguardrails import LLMRails, RailsConfig
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up logging
os.makedirs("logs", exist_ok=True)
log_filename = datetime.now().strftime("logs/latency_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# FastAPI setup
app = FastAPI()

# Request model
class TextRequest(BaseModel):
    prompt: str
    enable_rag: bool = False  # Optional flag, not used now

# Global variables
rails = None

# Load generation model
model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)

# Prevent "Sliding Window Attention is enabled" warning
if hasattr(model.config, "use_sliding_window"):
    model.config.use_sliding_window = False

# Build pipeline for HuggingFace
qa_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Load FAISS vector store and embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_pii_index", embeddings=embedding_model, allow_dangerous_deserialization=True
)

# Load LangChain QA chain
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qa_chain = load_qa_chain(llm, chain_type="stuff")

SIMILARITY_THRESHOLD = 0.7

# --- Guardrails Startup ---
@app.on_event("startup")
async def startup_event():
    global rails

    # --- Inline Guardrails config ---
    yaml_content = """
instructions:
  - type: general
    content: |
      You are a helpful assistant. Answer the user's questions thoughtfully.

models:
  - type: langchain
    engine: huggingface_pipeline
    model: HuggingFacePipeline
"""

    colang_content = """
define user express greeting
    "hello"
    "hi"
    "what's up?"

define user ask question
    "Can you help me with *?"
    "I have a question about *"
    "Tell me about *"
    "Explain *"
    "What is *"

define flow greet_user
    user express greeting
    assistant express greeting

define flow answer_question
    user ask question
    assistant call action run_rag
"""

    # Load guardrails config
    config = RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content
    )

    rails = LLMRails(config)

    # Register RAG as a Guardrails action
    @rails.action()
    async def run_rag(query: str) -> str:
        """Custom RAG retrieval and fallback generation."""
        results = vectorstore.similarity_search_with_score(query, k=5)
        docs = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

        if docs:
            return qa_chain.run(input_documents=docs, question=query)
        else:
            # Fallback to LLM-only generation with fallback message
            fallback_message = (
                "I couldn't find relevant documents to directly answer your question. "
                "However, based on my general knowledge:\n\n"
            )

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

            return fallback_message + generated_text

# --- FastAPI Endpoint ---
@app.post("/rag")
async def rag_smart_response(request: TextRequest):
    start_time = time.time()
    query = request.prompt
    source = "guardrails"

    try:
        # Use Guardrails to generate a response
        guardrails_response = await rails.generate(prompt=query)

        return JSONResponse(
            content={"response": guardrails_response, "source": source}
        )

    finally:
        end_time = time.time()
        latency = round(end_time - start_time, 4)
        logging.info(
            f"Source: {source} | Latency: {latency} sec | Prompt: {query[:50]}..."
        )

