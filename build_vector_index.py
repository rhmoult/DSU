from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd

# Load your CSV
df = pd.read_csv("synthetic_pii_data.csv")

# Create documents from the CSV
documents = [
    Document(
        page_content=(
            f"{row['First Name']} {row['Last Name']}'s SSN is {row['SSN']} "
            f"and their phone number is {row['Phone Number']}."
        )
    )
    for _, row in df.iterrows()
]

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save the index
vectorstore.save_local("faiss_pii_index")
