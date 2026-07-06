import logging
from pathlib import Path
import sys
import time

# Add parent directory to path to import retrieval module
# For standalone deployment, the retrieval.py is in the same directory
from retreival import hybrid_search
from app_config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, DEFAULT_TOP_K, DEFAULT_TEMPERATURE

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================
# LLM Configuration
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """Du bist ein hilfreicher Support-Spezialist für die FIO ACCOUNT Software (Mietkautionsverwaltung).

**Regeln:**
1. Verwende NUR Informationen aus den bereitgestellten Kontext-Dokumenten.
2. Wenn die Dokumente die Antwort nicht enthalten, sage klar: "Diese Information ist in den verfügbaren Dokumenten nicht enthalten."
3. Gib präzise, handlungsorientierte Antworten in Deutsch.
4. Verwende Bullet Points für Schritt-für-Schritt-Anleitungen.
5. Verweise auf die Quelle (FAQ oder Support-Ticket) wenn möglich.

**Kontext-Dokumente:**
{context}

**Benutzer-Frage:**
{query}

**Antwort:**
"""

# ============================================================
# RAG Generation Function
# ============================================================

def generate_rag_answer(query: str, retrieved_chunks: list) -> tuple:
    if not retrieved_chunks:
        return "Es wurden keine relevanten Informationen gefunden.", []

    context_parts = []
    sources = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        doc_text = chunk.get('document', '')
        metadata = chunk.get('metadata', {})

        if not doc_text:
            q = metadata.get('question', '')
            a = metadata.get('answer', '')
            doc_text = f"Frage: {q}\nAntwort: {a}"

        doc_id = metadata.get('doc_id', chunk.get('doc_id', f'chunk_{i}'))
        source_type = metadata.get('source_type', 'support_ticket')
        authority = metadata.get('authority', 'resolved_ticket')

        sources.append({
            'doc_id': doc_id,
            'source_type': source_type,
            'authority': authority,
            'question': metadata.get('question', ''),
            'excerpt': doc_text[:300] + '...' if len(doc_text) > 300 else doc_text
        })

        context_parts.append(f"[Dokument {i}] (ID: {doc_id})\n{doc_text}")

    context = "\n\n---\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Support-Spezialist."},
                {"role": "user", "content": SYSTEM_PROMPT.format(context=context, query=query)}
            ],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return answer, sources

    except Exception as exc:
        log.error(f"RAG generation failed: {exc}")
        return f"Fehler bei der Antwortgenerierung: {exc}", sources

# ============================================================
# Main RAG Pipeline Function
# ============================================================

def get_chatbot_response(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    dense_top_k: int = 20,
    sparse_top_k: int = 20
) -> dict:
    start_time = time.time()

    try:
        retrieved_chunks = hybrid_search(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k
        )

        answer, sources = generate_rag_answer(query, retrieved_chunks)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "error": None,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

    except Exception as exc:
        log.error(f"Chatbot response failed: {exc}")
        return {
            "answer": f"Es ist ein Fehler aufgetreten: {exc}",
            "sources": [],
            "retrieved_chunks": [],
            "error": str(exc),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }