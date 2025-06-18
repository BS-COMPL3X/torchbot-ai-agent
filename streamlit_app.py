import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image

# Adding custom prompt structure
prompt_template = """
You are a helpful assistant answering internal questions about Torchbox, a digital agency.
Use only the provided documents to answer.
Be concise, factual, and professional.
If the answer is not found in the documents, say "I don’t know based on the available information, but do not hesitate to contact us if you need more information."

Question: {question}
Context: {context}
Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"]
)

# Add logo
logo = Image.open("assets/torchbox_logo.png")
st.image(logo, width=150)

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load vector store
@st.cache_resource
def load_qa_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    return RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

qa_chain = load_qa_chain()

# UI
st.title("TorchBOT – Internal AI FAQ")
st.write("Ask me anything basic about Torchbox's values, work, team, or structure.")

query = st.text_input("❓ Your question:")

if query:
    result = qa_chain({"query": query})
    st.markdown("### 💬 Answer:")
    st.write(result["result"])

    st.markdown("### 📚 Sources:")
    for doc in result["source_documents"]:
        st.code(doc.metadata.get("source", "Unknown"))
