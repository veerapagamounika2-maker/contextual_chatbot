from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------- Step 1: Load embeddings model --------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- Step 2: Load FAISS vector DB --------
vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully")

# -------- Step 3: Ask a question --------
query = "What is the return policy?"

# -------- Step 4: Retrieve relevant chunks --------
docs = vector_db.similarity_search(query, k=2)

print("\nUser Question:")
print(query)

print("\nRetrieved Context:\n")
for i, doc in enumerate(docs):
    print(f"--- RESULT {i+1} ---")
    print(doc.page_content)
