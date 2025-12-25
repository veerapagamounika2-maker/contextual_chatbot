from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# -------- Step 1: Load documents --------
data_folder = "data"
documents = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(data_folder, file_name)
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

print(f"Documents loaded: {len(documents)}")

# -------- Step 2: Create chunks --------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,     # size of each chunk
    chunk_overlap=50    # overlap between chunks
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# -------- Step 3: Preview chunks --------
print("\n--- SAMPLE CHUNK 1 ---\n")
print(chunks[0].page_content)

print("\n--- SAMPLE CHUNK 2 ---\n")
if len(chunks) > 1:
    print(chunks[1].page_content)

print("\n ---sample chunk 3----\n")
if len(chunks) > 2:
    print(chunks[2].page_content)
