from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import warnings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

# Presidio
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# FastAPI setup
app = FastAPI()

# Request model
class TextRequest(BaseModel):
    prompt: str
    enable_rag: bool = False  # RAG is disabled by default

# Presidio engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

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

@app.post("/rag")
async def rag_smart_response(request: TextRequest):
    query = request.prompt
    use_rag = request.enable_rag

    if use_rag:
        results = vectorstore.similarity_search_with_score(query, k=5)
        docs = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

        # Redact PII (including SSNs) from document content
        redacted_docs = []
        for doc in docs:
            text = doc.page_content
            findings = analyzer.analyze(text=text, language="en")
            if findings:
                text = anonymizer.anonymize(text=text, analyzer_results=findings).text
            doc.page_content = text  # Replace content with redacted version
            redacted_docs.append(doc)

        if redacted_docs:
            result = qa_chain.run(input_documents=redacted_docs, question=query)
            return JSONResponse(content={
                "response": result,
                "source": "retrieved_docs",
                "pii_redacted": True
            })

    # Fallback to LLM-only generation
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

    return JSONResponse(content={
        "response": generated_text,
        "source": "llm_only" if not use_rag else "llm_fallback",
        "pii_redacted": False
    })

