import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the saved FAISS vector store
print("📦 Loading vector store...")
db = FAISS.load_local(
    "vectorstore",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)

# Set up the language model and QA chain
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# Command-line interface
print("\n✅ TorchBOT is ready. Ask any question about Torchbox.")
print("Type 'exit' to quit.\n")

while True:
    question = input("❓ Your question: ")
    if question.lower() in ("exit", "quit"):
        break

    result = qa_chain({"query": question})
    answer = result["result"]
    print(f"\n💬 Answer:\n{answer}\n")

    # Optional: show where the answer came from
    print("📚 Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata["source"])
    print("-" * 50)
