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

# Presidio
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
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
    enable_rag: bool = True  # RAG is enabled by default


# Presidio engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

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

# Load LangChain QA chain with deprecation warning suppressed
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qa_chain = load_qa_chain(llm, chain_type="stuff")

SIMILARITY_THRESHOLD = 0.7


@app.post("/rag")
async def rag_smart_response(request: TextRequest):
    start_time = time.time()
    query = request.prompt
    use_rag = request.enable_rag
    source = "unknown"

    try:
        if use_rag:
            results = vectorstore.similarity_search_with_score(query, k=5)
            docs = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

            # Redact PII (including SSNs) from document content
            redacted_docs = []
            for doc in docs:
                text = doc.page_content
                findings = analyzer.analyze(text=text, language="en")
                if findings:
                    text = anonymizer.anonymize(
                        text=text, analyzer_results=findings
                    ).text
                doc.page_content = text
                redacted_docs.append(doc)

            if redacted_docs:
                result = qa_chain.run(input_documents=redacted_docs, question=query)
                source = "retrieved_docs"
                return JSONResponse(
                    content={"response": result, "source": source, "pii_redacted": True}
                )

        # Fallback to LLM-only generation
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
            early_stopping=False,  # no effect with num_beams=1, suppresses warning
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        source = "llm_only" if not use_rag else "llm_fallback"

        return JSONResponse(
            content={
                "response": generated_text,
                "source": source,
                "pii_redacted": False,
            }
        )

    finally:
        end_time = time.time()
        latency = round(end_time - start_time, 4)
        logging.info(
            f"Source: {source} | Latency: {latency} sec | Prompt: {query[:50]}..."
        )
