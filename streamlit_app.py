"""Streamlit chat interface for the Gemini + Vanna promotion assistant."""

import asyncio
import html
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import importlib
import pandas as pd
import streamlit as st


# Ensure local vanna package (vanna_repo/src) is importable when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent
VANNA_SRC = PROJECT_ROOT / "vanna_repo" / "src"
if VANNA_SRC.exists() and str(VANNA_SRC) not in sys.path:
    sys.path.insert(0, str(VANNA_SRC))

importlib.invalidate_caches()

from vanna.core.user import RequestContext
from vanna.components.rich.data.dataframe import DataFrameComponent
from vanna.examples.gemini_excel_schema_example import (
    create_excel_schema_agent,
    collect_agent_response,
)


# Note: Module reload removed to preserve persistent Snowflake connection
# The @lru_cache decorator on _get_agent() ensures a single SnowflakeRunner instance
# is created and reused throughout the Streamlit session, avoiding repeated browser auth


def _build_snowflake_config() -> Optional[Dict[str, str]]:
    """Build Snowflake configuration from environment variables or hardcoded values."""

    # First try environment variables
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    # If not in env vars, use hardcoded values (same as example script)
    if not account or not user or not warehouse:
        config = {
            "account": "cvs-cvsretailprod",
            "user": "bharati.peddinti@CVSHealth.com",
            "role": "GRP-CN-SCAI-ANALYTICS",
            "warehouse": "WH_SCAI_ANALYTICS_L_QUERY_01",
            "authenticator": "externalbrowser",
        }
        return config

    # Build from environment variables
    config = {
        "account": account,
        "user": user,
        "warehouse": warehouse,
    }

    role = os.getenv("SNOWFLAKE_ROLE")
    if role:
        config["role"] = role

    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")
    if authenticator:
        config["authenticator"] = authenticator

    password = os.getenv("SNOWFLAKE_PASSWORD")
    if password:
        config["password"] = password

    return config


@lru_cache(maxsize=1)
def _get_agent():
    excel_path = os.getenv("EXCEL_PATH", "Promo Semantic Data_20250519.xlsx")
    snowflake_config = _build_snowflake_config()

    agent = create_excel_schema_agent(
        excel_path=excel_path,
        snowflake_config=snowflake_config,
    )
    return agent, snowflake_config is not None


def _clean_response_text(text: str) -> str:
    """Remove technical details and tool call information from response text."""
    # Remove search schema results
    if "Search results for" in text and "searched across ALL tables and columns" in text:
        return None
    
    # Remove tool call confirmations
    if "Query executed successfully" in text and "No rows returned" in text:
        return None
    
    # Remove file save messages
    if "Results saved to file:" in text or "FOR VISUALIZE_DATA USE FILENAME:" in text:
        return None
    
    # Remove technical SQL preview messages
    if "Top" in text and "rows shown below" in text and "Full results saved to file:" in text:
        # Extract just the data preview part
        lines = text.split("\n")
        cleaned_lines = []
        skip_next = False
        for line in lines:
            if "Top" in line and "rows shown below" in line:
                cleaned_lines.append(line)
                skip_next = False
            elif "Full results saved to file:" in line or "FOR VISUALIZE_DATA" in line:
                skip_next = True
            elif not skip_next and line.strip():
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines) if cleaned_lines else None
    
    # Remove empty or very short technical messages
    if len(text.strip()) < 10:
        return None
    
    return text


async def _ask_agent(prompt: str) -> Dict[str, List]:
    agent, snowflake_enabled = _get_agent()

    request_context = RequestContext(
        cookies={"vanna_email": os.getenv("CHAT_USER_EMAIL", "demo@example.com")},
        remote_addr="127.0.0.1",
    )

    response = await collect_agent_response(
        agent=agent,
        request_context=request_context,
        message=prompt,
        conversation_id=None,
    )

    # Clean response texts - filter out technical details
    cleaned_texts = []
    for text in response.texts:
        cleaned = _clean_response_text(text)
        if cleaned:
            cleaned_texts.append(cleaned)
    
    # If no cleaned text but we have tables, provide a default message
    if not cleaned_texts and response.tables:
        cleaned_texts = ["Here are the results:"]
    elif not cleaned_texts:
        cleaned_texts = ["I couldn't find any data for your query. Please try rephrasing your question."]

    return {"texts": cleaned_texts, "tables": response.tables, "snowflake_enabled": snowflake_enabled}


# Custom CSS for chat interface
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Promo Analytics Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛍️ Promo Analytics Assistant</h1>
    <p style="margin: 0; font-size: 1.1rem;">Ask questions about promotions, pricing, and store performance</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    # Show Snowflake status
    _, snowflake_enabled = _get_agent()
    if snowflake_enabled:
        st.success("✅ **Connected to Snowflake**")
    else:
        st.warning("⚠️ **Snowflake Disabled**")
        st.caption("SQL queries will be logged only")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    if st.session_state.history:
        for entry in st.session_state.history:
            if entry["role"] == "user":
                # User message - right aligned
                col1, col2 = st.columns([0.7, 0.3])
                with col2:
                    st.markdown(
                        f'<div style="background-color: #E3F2FD; padding: 0.75rem; border-radius: 1rem; margin-bottom: 1rem;">'
                        f'<strong>👤 You</strong><br>{html.escape(entry["text"])}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                # Assistant message - left aligned with header
                st.markdown(
                    '<div style="background-color: #F5F5F5; padding: 0.75rem; border-radius: 1rem; margin-bottom: 0.5rem;">'
                    '<strong>🤖 Assistant</strong>'
                    '</div>',
                    unsafe_allow_html=True
                )
                # Render message content as markdown (supports formatting)
                st.markdown(entry['text'])
                
                # Display tables if available
                for df in entry.get("tables", []):
                    st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown("")  # Add spacing
    else:
        st.info("👋 **Welcome!** Start a conversation by asking a question about promotional campaigns, sales performance, or store analytics.\n\n*Example: \"What is the performance of the promotional campaign with AD_BLK_NBR 45737?\"*")

# Input area at the bottom
st.markdown("---")

# Use form to prevent rerun on every keystroke
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Type your question here...",
            key="user_query",
            placeholder="e.g., What is the average sale price during promotions for SKU 98765?",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button("Send", type="primary", use_container_width=True)

col3, col4 = st.columns([1, 1])
with col3:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_query = None
        st.rerun()

# Process query only when form is submitted
if submit_button and user_input.strip():
    # Check if this is a new query (prevent infinite loop)
    if st.session_state.last_query != user_input.strip():
        st.session_state.last_query = user_input.strip()
        
        # Add user message to history
        st.session_state.history.append({"role": "user", "text": user_input.strip()})
        
        # Show thinking indicator
        with st.spinner("🤔 Analyzing your question..."):
            result = asyncio.run(_ask_agent(user_input.strip()))

        # Combine all text responses into one clean message
        combined_text = "\n\n".join(result["texts"]) if result["texts"] else "I'm here to help! Please ask me a question about promotional data."
        
        # Add assistant response to history
        st.session_state.history.append({
            "role": "assistant",
            "text": combined_text,
            "tables": result["tables"],
        })
        
        # Rerun to show new messages
        st.rerun()

