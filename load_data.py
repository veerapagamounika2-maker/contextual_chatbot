from langchain_community.document_loaders import TextLoader
import os

data_folder = "data"
documents = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(data_folder, file_name)
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

print(f"Total documents loaded: {len(documents)}")
