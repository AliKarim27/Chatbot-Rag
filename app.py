"""
Streamlit Chatbot UI for RAG over any webpage.
The webpage must be ingested first by running:  python ingest.py
Then launch:  streamlit run app.py
"""

import streamlit as st
from rag_core import load_vectorstore, build_rag_chain, vectorstore_exists
from logger_setup import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    layout="centered",
)

st.title("RAG Assistant")
st.caption("Ask questions about the indexed webpage — powered by LangChain, ChromaDB, and Ollama")

# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ask_fn" not in st.session_state:
    st.session_state.ask_fn = None

# ── Load vector store from ChromaDB on first run ────────────────────────────
if st.session_state.ask_fn is None:
    if vectorstore_exists():
        with st.spinner("Loading vector database..."):
            logger.info("Vectorstore found — loading")
            vectorstore = load_vectorstore()
            st.session_state.ask_fn = build_rag_chain(vectorstore)
    else:
        st.error(
            "No vector database found. "
            "Please run `python ingest.py` first to scrape and index a webpage."
        )
        st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.markdown(
        "This assistant answers questions using content retrieved "
        "from a previously indexed webpage.\n\n"
        "**Usage**\n"
        "1. Set the target URL in `ingest.py`\n"
        "2. Run `python ingest.py`\n"
        "3. Launch `streamlit run app.py`\n"
    )
    st.divider()
    st.markdown(
        "**Technology**\n\n"
        "- Embeddings: HuggingFace (local)\n"
        "- Vector DB: ChromaDB\n"
        "- LLM: Ollama llama3.2:1b (local)"
    )
    st.divider()
    if st.button("Reload Database"):
        with st.spinner("Reloading..."):
            logger.info("Reloading vectorstore on user request")
            vectorstore = load_vectorstore()
            st.session_state.ask_fn = build_rag_chain(vectorstore)
            st.session_state.messages = []
        st.success("Database reloaded successfully.")

# ── Chat area ────────────────────────────────────────────────────────────────
# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                answer, source_docs = st.session_state.ask_fn(
                    prompt, st.session_state.messages[:-1]
                )

                # Show source snippets in an expander
                if source_docs:
                    with st.expander("View source snippets"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Chunk {i}**")
                            st.text(doc.page_content[:500])
                            st.divider()

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                logger.exception("Error generating response for prompt: %s", prompt)
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
