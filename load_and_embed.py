import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Folder where your .txt files live
DOCS_DIR = "docs"

# Load all text files as LangChain Documents
def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            loader = TextLoader(path, encoding="utf-8")
            doc = loader.load()
            docs.extend(doc)
    return docs

# Split documents into chunks
def split_documents(documents, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

if __name__ == "__main__":
    print("📄 Loading documents...")
    docs = load_documents(DOCS_DIR)
    print(f"✅ Loaded {len(docs)} documents")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    # Optional preview
    print("\n🔹 Sample chunk:\n", chunks[0].page_content[:500])
