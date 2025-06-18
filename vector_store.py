import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load and split documents again
def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            path = os.path.join(directory, filename)
            loader = TextLoader(path, encoding="utf-8")
            doc = loader.load()
            docs.extend(doc)
    return docs

def split_documents(documents, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

if __name__ == "__main__":
    print("📄 Loading documents...")
    documents = load_documents("docs")
    print(f"✅ Loaded {len(documents)} documents")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔍 Generating embeddings and creating FAISS store...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(chunks, embeddings)

    # Save the vector database locally
    db.save_local("vectorstore")
    print("✅ Vector store saved to ./vectorstore/")
