import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ OPENAI_API_KEY not found in .env file!")
else:
    print("✅ API key loaded. Testing OpenAI...")

    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    response = chat.predict("What is Torchbox?")
    print("🔹 Response:", response)

