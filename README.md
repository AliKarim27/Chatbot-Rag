# 🤖 Webpage RAG Chatbot

A chatbot that can answer any question about the content of a webpage you provide. Built with **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Google Gemini** (free tier).

## How It Works

```
URL ──▶ Scrape Text ──▶ Split into Chunks ──▶ Embed (HuggingFace) ──▶ Store in FAISS
                                                                          │
User Question ──▶ Retrieve relevant chunks ──▶ Gemini LLM ──▶ Answer ◀───┘
```

1. **Scrape** — Extracts clean text from any webpage  
2. **Chunk** — Splits text into overlapping 1000-char chunks  
3. **Embed** — Uses `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)  
4. **Store** — Saves embeddings in a FAISS vector store  
5. **Query** — Uses Gemini 2.0 Flash to generate answers from retrieved chunks  

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Framework | LangChain | Free |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free (local) |
| Vector Store | FAISS | Free (local) |
| LLM | Google Gemini 2.0 Flash | Free tier |
| UI | Streamlit | Free |

## Setup

### 1. Clone & install dependencies

```bash
cd Chatbot-Rag
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get a free Gemini API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)  
2. Sign in with your Google account  
3. Click **"Create API Key"**  
4. Copy the key

### 3. Configure the API key

Edit the `.env` file and replace the placeholder:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

1. Paste any webpage URL in the sidebar  
2. Click **Load & Index** (takes a few seconds)  
3. Ask questions in the chat input  
4. Expand **"Source snippets"** to see which chunks were used  

## Project Structure

```
Chatbot-Rag/
├── app.py              # Streamlit chatbot UI
├── rag_core.py         # RAG pipeline (scrape, embed, query)
├── requirements.txt    # Python dependencies
├── .env                # API key config (not committed)
├── .env.example        # Template for .env
├── .gitignore
└── README.md
```

## Notes

- The **embeddings model** downloads once (~80 MB) and runs locally — no API needed.  
- **Gemini free tier** allows ~15 requests/minute and 1,500/day — plenty for personal use.  
- The FAISS index is kept in memory by default. Call `save_vectorstore()` / `load_vectorstore()` in `rag_core.py` to persist it to disk.
