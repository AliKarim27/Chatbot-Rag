"""
RAG (Retrieval-Augmented Generation) Core Module
Handles: webpage loading, chunking, embedding, ChromaDB vector storage, and Q&A.
Uses modern LangChain LCEL (LangChain Expression Language).
"""

import os
import re
import requests
from bs4 import BeautifulSoup

from logger_setup import get_logger

logger = get_logger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Path where ChromaDB stores its data persistently
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


# ---------------------------------------------------------------------------
# 1. Web scraping – extract clean text from a URL
# ---------------------------------------------------------------------------

def scrape_webpage(url: str) -> str:
    """Fetch the URL and return the visible text content."""
    logger.info("Fetching URL: %s", url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.SSLError:
        # Fallback: skip SSL verification for sites with certificate issues
        logger.warning("SSL error fetching %s — retrying without verification", url)
        response = requests.get(url, headers=headers, timeout=30, verify=False)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, nav, footer, header tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "title"]):
        tag.decompose()

    # Remove site-specific clutter from ul.edu.lb pages
    for el in soup.select(".navbarcontent"):
        el.decompose()
    for el in soup.select("section.addit-header.faculties-add-header.hide-mob"):
        el.decompose()
    for el in soup.select("nav.navbar.navbar-expand-lg.navbar-light.bg-light.nav-first-top"):
        el.decompose()
    for el in soup.select("p.prof-speech.before-pink-main.cairoreg.font15"):
        el.decompose()
    for el in soup.select(".col-md-3.ph-right.right-list-type2"):
        el.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    logger.debug("Fetched %d characters from %s", len(text), url)
    return text


# ---------------------------------------------------------------------------
# 2. Split text into chunks
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split raw text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([text])
    return docs


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------

def get_embeddings():
    """Return a HuggingFace embeddings model (runs locally, 100 % free)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ---------------------------------------------------------------------------
# 4. ChromaDB vector store (persistent)
# ---------------------------------------------------------------------------

def build_vectorstore(docs: list[Document], collection_name: str = "webpage"):
    """Create / overwrite a ChromaDB collection from documents."""
    embeddings = get_embeddings()
    logger.info("Building vectorstore (collection=%s) with %d documents", collection_name, len(docs))
    # Delete existing collection to avoid duplicates on re-ingest
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name,
    )
    return vectorstore


def load_vectorstore(collection_name: str = "webpage"):
    """Load an existing ChromaDB collection from disk."""
    embeddings = get_embeddings()
    logger.info("Loading vectorstore (collection=%s) from %s", collection_name, CHROMA_DB_DIR)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectorstore


def vectorstore_exists() -> bool:
    """Check whether the ChromaDB directory exists and has data."""
    exists = os.path.isdir(CHROMA_DB_DIR) and len(os.listdir(CHROMA_DB_DIR)) > 0
    logger.debug("vectorstore_exists=%s (dir=%s)", exists, CHROMA_DB_DIR)
    return exists


# ---------------------------------------------------------------------------
# 5. Build the LLM (Ollama - runs locally, 100% free)
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2:3b"


def get_llm():
    """
    Return a local Ollama model.
    Requires Ollama to be installed and running.
    Pull the model first:  ollama pull llama3.2:3b
    """
    logger.info("Instantiating Ollama model: %s", OLLAMA_MODEL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# 6. RAG Q&A function (LCEL-based, with chat history)
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable assistant for the Faculty of Engineering at the Lebanese University.
You answer questions based ONLY on the provided context, which is scraped from official faculty webpages.
If the answer is not in the context, say "I don't have enough information from the faculty webpages to answer that."


Rules:
- Be concise, accurate, and well-structured.
- If the user asks in Arabic or French, respond in the same language.
- When listing items (departments, names, etc.), use bullet points.
- When answering, specify which branch the information relates to if applicable.
- Always cite which part of the context your answer comes from.

Context from faculty webpages:
{context}
"""


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single string for the prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    """
    Build a RAG chain using LCEL.
    Returns a callable: ask(question, chat_history) -> (answer, source_docs)
    Uses hybrid retrieval: keyword filter first, then semantic fallback.
    """
    K = 9

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    def _extract_keyword_filter(question: str):
        """
        Detect specific article/section references in the question
        and return a ChromaDB where_document filter.
        """
        # Match patterns like "article 1", "Article 12", "article no. 3"
        match = re.search(r'article\s*(?:no\.?\s*)?(\d+)', question, re.IGNORECASE)
        if match:
            article_num = match.group(1)
            return {"$or": [
                {"$contains": f"Article {article_num}"},
                {"$contains": f"article {article_num}"},
                {"$contains": f"Article {article_num} "},
                {"$contains": f"Article {article_num}:"},
            ]}
        return None

    def _extract_metadata_filter(question: str):
        """
        Detect branch references in the question and return a
        ChromaDB metadata filter to narrow results by title.
        """
        q = question.lower()
        # Detect branch number
        branch_match = re.search(r'branch\s*(\d+)', q)
        if branch_match:
            branch_num = branch_match.group(1)
            branch_prefix = f"Branch {branch_num}"
            return {"title": {"$contains": branch_prefix}}
        return None

    def _retrieve(question: str):
        """
        Hybrid retrieval:
        1. Try metadata filter (branch) + semantic search
        2. Try keyword filter (article) in document content
        3. Fall back to pure semantic search
        """
        metadata_filter = _extract_metadata_filter(question)
        keyword_filter = _extract_keyword_filter(question)

        # Try metadata-filtered search (e.g., branch-specific)
        if metadata_filter:
            try:
                filtered_docs = vectorstore.similarity_search(
                    question, k=K, filter=metadata_filter,
                    where_document=keyword_filter  # can be None, that's fine
                )
                if filtered_docs:
                    return filtered_docs
            except Exception:
                pass

        # Try keyword-only filter (e.g., article number)
        if keyword_filter:
            try:
                filtered_docs = vectorstore.similarity_search(
                    question, k=K, filter=None,
                    where_document=keyword_filter
                )
                if filtered_docs:
                    return filtered_docs
            except Exception:
                pass

        # Fallback: pure semantic search
        retriever = vectorstore.as_retriever(search_kwargs={"k": K})
        logger.debug("Performing semantic fallback retrieval for question: %s", question)
        return retriever.invoke(question)

    def ask(question: str, chat_history: list = None):
        """
        Ask a question with optional chat history.
        Returns (answer_text, source_documents).
        """
        if chat_history is None:
            chat_history = []

        # Retrieve relevant documents (hybrid: keyword + semantic)
        logger.info("Retrieving documents for question: %s", question)
        source_docs = _retrieve(question)
        context = format_docs(source_docs)

        # Build chat history messages
        history_messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            else:
                history_messages.append(AIMessage(content=msg["content"]))

        try:
            answer = chain.invoke({
                "context": context,
                "chat_history": history_messages,
                "question": question,
            })
        except Exception:
            logger.exception("LLM chain failed for question: %s", question)
            raise

        return answer, source_docs

    return ask


# ---------------------------------------------------------------------------
# 7. High-level helpers
# ---------------------------------------------------------------------------

def ingest_url(url: str, title: str = "", collection_name: str = "webpage"):
    """
    End-to-end ingestion pipeline for a single URL.
    Returns (vectorstore, stats).
    """
    return ingest_urls([(title or url, url)], collection_name)


def ingest_urls(urls, collection_name: str = "webpage", progress_callback=None):
    """
    End-to-end ingestion pipeline for one or more URLs.
    urls: list of (title, url, description) tuples, (title, url) tuples, or plain url strings.
    Returns (vectorstore, stats) where stats is a list of per-URL dicts.
    progress_callback(i, total, title, status) is called after each URL.
    """
    all_docs = []
    stats = []  # {title, url, description, status, chars, chunks}

    # Normalize: accept plain strings, 2-tuples, or 3-tuples
    normalized = []
    for item in urls:
        if isinstance(item, str):
            normalized.append((item, item, ""))
        elif len(item) == 2:
            normalized.append((item[0], item[1], ""))
        else:
            normalized.append((item[0], item[1], item[2]))

    for i, (title, url, description) in enumerate(normalized, 1):
        try:
            text = scrape_webpage(url)
            if not text.strip():
                stats.append({"title": title, "url": url, "description": description, "status": "empty", "chars": 0, "chunks": 0})
                if progress_callback:
                    progress_callback(i, len(normalized), title, "empty")
                continue

            docs = split_text(text)
            # Prepend title & description into each chunk's content
            # so they become searchable by semantic + keyword search
            prefix = f"[{title}]"
            if description:
                prefix += f" {description}"
            prefix += "\n\n"
            for doc in docs:
                doc.page_content = prefix + doc.page_content
                doc.metadata["source"] = url
                doc.metadata["title"] = title
                doc.metadata["description"] = description
            all_docs.extend(docs)
            stats.append({"title": title, "url": url, "description": description, "status": "ok", "chars": len(text), "chunks": len(docs)})
            if progress_callback:
                progress_callback(i, len(normalized), title, "ok")
        except Exception as e:
            stats.append({"title": title, "url": url, "description": description, "status": f"error: {e}", "chars": 0, "chunks": 0})
            if progress_callback:
                progress_callback(i, len(normalized), title, "error")
            continue

    if not all_docs:
        raise ValueError("No content could be extracted from any of the provided URLs.")

    vectorstore = build_vectorstore(all_docs, collection_name)
    return vectorstore, stats
