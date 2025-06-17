# TorchBOT – Internal AI Assistant for Torchbox

**TorchBOT** is a retrieval-augmented AI assistant built to help answer internal questions about Torchbox using real content from their public website. It's a focused demo aligned with the [AI Agent Engineer role](https://torchbox.com), built to show hands-on skills in automation, LLMs, and agent workflows.

---

## 🔍 What It Does

- Uses a custom GPT-4 agent to answer questions about:
  - Company structure, values, and culture
  - Services, clients, team members, and work
- Trained on public Torchbox pages scraped and chunked into vector embeddings
- Sources answers from real content with transparent citations

---

## 🚀 Live Demo (via Streamlit Cloud)

🔗 [Try the app here](https://YOUR-APP-NAME.streamlit.app)

Ask questions like:
- “What services does Torchbox offer?”
- “What is employee ownership?”
- “Where is the office located?”
- “How does Torchbox approach AI?”

---

## 🧠 Technologies Used

| Tech            | Purpose                         |
|-----------------|----------------------------------|
| Python          | Backend scripting               |
| Streamlit       | Lightweight web UI              |
| OpenAI (GPT-4)  | LLM engine                      |
| LangChain       | RAG orchestration & chain setup |
| FAISS           | Vector DB for semantic search   |
| tiktoken        | Token-aware chunking            |

---

## 💻 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/torchbot-ai-agent.git
cd torchbot-ai-agent
pip install -r requirements.txt
streamlit run streamlit_app.py

➡️ Add your OpenAI key to a .env file:

OPENAI_API_KEY=sk-xxxxx

Or store it in Streamlit Cloud's Secrets Manager.

This project was built in response to Torchbox’s draft AI Agent Engineer role. It demonstrates:

Real-world implementation of AI agent principles
Focus on utility, performance, and clarity
Comfort across both prototyping and deployment

Thanks for reading.

– Juan Luis Barrales
