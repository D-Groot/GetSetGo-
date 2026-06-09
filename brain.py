# brain.py

import chromadb                          # our vector database
from sentence_transformers import SentenceTransformer  # converts text → numbers
from groq import Groq                    # the LLM that writes answers
from datetime import datetime            # for timestamps
from dotenv import load_dotenv           # loads our .env file
import os

load_dotenv()  # reads the .env file and makes the API key available

# ── Load the embedding model ──────────────────────────────────────────────────
# This runs locally on your machine. First run downloads ~90MB. After that it's instant.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Connect to ChromaDB (local, no server needed) ─────────────────────────────
# PersistentClient means data is saved to disk — it survives restarts!
client = chromadb.PersistentClient(path="./my_brain_db")

# A "collection" is like a table in a normal database
collection = client.get_or_create_collection(name="memories")

# ── Connect to Groq (the LLM) ────────────────────────────────────────────────
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ════════════════════════════════════════════════════════════════════
#  STORE  — Data ingestion pipeline
# ════════════════════════════════════════════════════════════════════

def store_text(text: str) -> str:
    """
    Takes user text, converts it to a vector, and saves it in ChromaDB.
    Returns a status message.
    """
    # Step 1: Chunking
    # For now, we treat the whole input as one chunk.
    # (Future: split long texts into overlapping paragraphs)
    chunks = [text.strip()]

    # Step 2: Embed each chunk → get a list of numbers
    vectors = embedder.encode(chunks).tolist()

    # Step 3: Create a unique ID and timestamp
    timestamp = datetime.now().isoformat()          # e.g. "2025-05-24T14:30:00"
    doc_id = f"mem_{datetime.now().timestamp()}"    # unique ID

    # Step 4: Store in ChromaDB
    # documents = the raw text (so we can retrieve it later)
    # embeddings = the vector (so we can search by meaning)
    # metadatas = extra info like timestamp
    collection.add(
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"timestamp": timestamp, "source": "user_input"}],
        ids=[doc_id]
    )

    return f"✅ Stored! Saved at {timestamp}"


# ════════════════════════════════════════════════════════════════════
#  ASK  — Retrieval + Generation pipeline
# ════════════════════════════════════════════════════════════════════

def ask_brain(query: str) -> str:
    """
    Takes a question, finds relevant stored memories, sends them to LLM,
    and returns a grounded answer with citations.
    """
    # Step 1: Embed the query (same model = same number space)
    query_vector = embedder.encode([query]).tolist()

    # Step 2: Similarity search — find top 5 most relevant stored memories
    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(5, collection.count())   # don't ask for more than we have
    )

    # If nothing is stored yet, tell the user
    if not results["documents"][0]:
        return "🤔 No memories found yet. Switch to STORE mode and add some information first!"

    # Step 3: Build context from retrieved chunks (with timestamps)
    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        ts = meta.get("timestamp", "unknown time")
        context_parts.append(f"[Stored on {ts}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Send to LLM with instructions
    prompt = f"""You are a personal memory assistant. The user has stored the following notes:

{context}

Based ONLY on the above stored notes, answer this question: {query}

If the answer is not in the notes, say so clearly. Always mention when the relevant note was stored."""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",   # free Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3           # low = more factual, less creative
    )

    answer = response.choices[0].message.content

    # Step 5: Add citations
    citations = "\n\n---\n**📌 Sources used:**\n"
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        ts = meta.get("timestamp", "?")
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        citations += f"\n{i+1}. *{ts}* — \"{preview}\""

    return answer + citations


def count_memories() -> int:
    """Returns how many items are stored."""
    return collection.count()