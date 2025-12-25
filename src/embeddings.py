from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# -------- Step 1: Load documents --------
data_folder = "data"
documents = []

for file_name in sorted(os.listdir(data_folder)):
    if file_name.endswith(".txt"):
        file_path = os.path.join(data_folder, file_name)
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

print(f"Documents loaded: {len(documents)}")

# -------- Step 2: Chunk documents --------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")

# -------- Step 3: Create embeddings (LOCAL) --------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- Step 4: Store in FAISS --------
vector_db = FAISS.from_documents(chunks, embeddings)

# Save FAISS index locally
vector_db.save_local("faiss_index")

print("FAISS vector database created successfully")
