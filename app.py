"""
FIO ACCOUNT Support Chatbot
Main Streamlit application for answering user questions about FIO ACCOUNT software.
"""

import streamlit as st
import time
from rag_pipeline import get_chatbot_response
from app_config import DEFAULT_TOP_K, DEFAULT_TEMPERATURE

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="FIO ACCOUNT Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    .main > div {
        max-width: 1000px;
        margin: 0 auto;
    }

    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        float: right;
        max-width: 80%;
        clear: both;
    }

    .assistant-message {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        float: left;
        max-width: 80%;
        clear: both;
    }

    .source-badge {
        background-color: #e0e0e0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        display: inline-block;
        margin-right: 4px;
    }

    .source-faq {
        background-color: #c8e6c9;
    }

    .source-ticket {
        background-color: #bbdefb;
    }

    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_queries" not in st.session_state:
        st.session_state.processed_queries = 0
    if "total_time_ms" not in st.session_state:
        st.session_state.total_time_ms = 0
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

init_session_state()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    new_top_k = st.slider(
        "Retrieved chunks (k)",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="Number of document chunks to retrieve for each query"
    )
    if new_top_k != st.session_state.top_k:
        st.session_state.top_k = new_top_k

    new_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1
    )
    if new_temperature != st.session_state.temperature:
        st.session_state.temperature = new_temperature

    st.markdown("---")

    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.processed_queries)
    with col2:
        avg_time = st.session_state.total_time_ms / st.session_state.processed_queries if st.session_state.processed_queries > 0 else 0
        st.metric("Avg Response", f"{avg_time:.0f}ms")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about FIO ACCOUNT software.

        **Sources:**
        - 📄 Official FAQs
        - 🎫 Support tickets

        **Data:** Training data from 2024-2025
        """
    )

# ============================================================
# Main Chat Interface
# ============================================================

st.title("🤖 FIO ACCOUNT Support Chatbot")
st.caption("Ask me anything about FIO ACCOUNT software – I'll search through FAQs and support tickets to find the answer.")

chat_container = st.container()

with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                    <div style="background-color: #e3f2fd; padding: 10px 14px; border-radius: 16px; max-width: 80%;">
                        <strong>🧑 You</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            sources_html = ""
            if "sources" in message and message["sources"]:
                source_badges = []
                for src in message["sources"]:
                    badge_class = "source-faq" if src["source_type"] == "faq_document" else "source-ticket"
                    authority_icon = "📄" if src["authority"] == "official_documentation" else "🎫"
                    source_badges.append(
                        f'<span class="source-badge {badge_class}">{authority_icon} {src["doc_id"][:20]}...</span>'
                    )
                sources_html = f"<div style='margin-top: 8px; font-size: 0.8rem;'>📚 Sources: {' '.join(source_badges)}</div>"

            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                    <div style="background-color: #f5f5f5; padding: 10px 14px; border-radius: 16px; max-width: 80%;">
                        <strong>🤖 Assistant</strong><br>
                        {message["content"]}
                        {sources_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================================
# Input Area
# ============================================================

st.markdown("---")

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([10, 1])

    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="Type your question here...",
            key="user_input",
            label_visibility="collapsed"
        )

    with col2:
        submit_button = st.form_submit_button("Send", type="primary", use_container_width=True)

# ============================================================
# Process User Input
# ============================================================

if submit_button and user_input and user_input.strip():
    query = user_input.strip()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🤔 Thinking..."):
        response = get_chatbot_response(
            query=query,
            top_k=st.session_state.top_k
        )

        st.session_state.processed_queries += 1
        st.session_state.total_time_ms += response["processing_time_ms"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "retrieved_chunks": response["retrieved_chunks"]
        })

    st.rerun()

# ============================================================
# Welcome Message
# ============================================================

if not st.session_state.messages:
    st.info("""
    👋 **Welcome to the FIO ACCOUNT Support Chatbot!**

    I can help you with questions about:
    - 🔐 Account management (login, passwords, permissions)
    - 📊 Interest calculations and reports
    - 🏦 Security deposits (Kautionen)
    - 📄 Document management
    - ⚙️ System configuration
    - ❓ And more...

    **Try asking:**
    - *"Wie kann ich ein virtuelles Konto anlegen?"*
    - *"Was muss ich tun, wenn der Jahresabschluss nicht durchgeführt wurde?"*
    - *"Wie kann ich mein Passwort zurücksetzen?"*
    """)