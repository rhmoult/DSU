from langchain.schema.runnable import RunnableLambda
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Load DeepSeek model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-1.5b")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-r1-distill-qwen-1.5b", torch_dtype=torch.float16, device_map="auto"
)

# LangChain-compatible LLM
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Embedding and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_pii_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# QA chain for retrieved documents
qa_chain = load_qa_chain(llm, chain_type="stuff")
SIMILARITY_THRESHOLD = 0.7

# RAG retrieval logic
def retrieve_docs(query):
    results = vectorstore.similarity_search_with_score(query, k=5)
    return [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]

# RAG-based generation logic
def rag_or_fallback(query):
    docs = retrieve_docs(query)

    if docs:
        return qa_chain.invoke({"input_documents": docs, "question": query})
    else:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to("cuda")
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
        return tokenizer.decode(output[0], skip_special_tokens=True)

# Create LangChain Runnable
rag_chain = RunnableLambda(rag_or_fallback)

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Minimal Guardrails YAML configuration
yaml_content = """
instructions:
  - type: general
    content: |
      You are a helpful bot. Answer the user's questions thoughtfully.
      - If you don't know the answer, say: "I'm not sure about that."
      - Never make up information (hallucinate).
      - Never include your personal opinions (like "I think" or "I believe").
      - Always be concise and avoid rambling.

settings:
  passthrough: false

output_checks:
  - name: no_hallucination
    type: regex
    pattern: ".*I'm not sure.*"
    on_fail: "If unsure, clearly say 'I'm not sure about that.' Do not fabricate answers."

  - name: no_personal_opinion
    type: contains
    keywords: ["I think", "I believe", "In my opinion"]
    on_fail: "Do not include personal opinions. Stick to factual information."

  - name: no_ssn_rambling
    type: contains
    keywords: ["social security number", "SSN", "social security numbers"]
    on_fail: "Avoid discussing social security numbers unless absolutely necessary, and be concise."

  - name: no_ssn_format
    type: regex
    pattern: '\d{3}-\d{2}-\d{4}'
    on_fail: "Detected a pattern resembling a Social Security Number. Do not output SSNs."

  - name: limit_response_length
    type: length
    max_length: 500
    on_fail: "Keep responses concise and under 500 characters unless truly necessary."
"""

# Initialize RunnableRails and build the full chain
@app.on_event("startup")
async def startup_event():
    global chain

    config = RailsConfig.from_content(yaml_content=yaml_content)
    guardrails = RunnableRails(config=config, runnable=rag_chain)

    chain = guardrails | llm

@app.post("/rag")
async def rag_endpoint(request: QueryRequest):
    response = await chain.ainvoke(request.query)
    return {"response": response["output_text"]}
