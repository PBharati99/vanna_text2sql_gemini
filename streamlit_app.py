"""Streamlit chat interface for the Gemini + Vanna promotion assistant."""

import asyncio
import os
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from vanna.core.user import RequestContext
from vanna.components.rich.data.dataframe import DataFrameComponent
from vanna.examples.gemini_excel_schema_example import create_excel_schema_agent


def _build_snowflake_config() -> Optional[Dict[str, str]]:
    """Build Snowflake configuration from environment variables if provided."""

    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")

    if not account or not user or not warehouse:
        return None

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
    return agent


async def _ask_agent(prompt: str) -> Dict[str, List]:
    agent = _get_agent()

    request_context = RequestContext(
        cookies={"vanna_email": os.getenv("CHAT_USER_EMAIL", "demo@example.com")},
        remote_addr="127.0.0.1",
    )

    texts: List[str] = []
    tables: List[pd.DataFrame] = []

    async for component in agent.send_message(
        request_context=request_context,
        message=prompt,
        conversation_id=None,
    ):
        # Plain text content
        if component.simple_component and component.simple_component.text:
            texts.append(component.simple_component.text)

        # Rich DataFrame components
        rich = getattr(component, "rich_component", None)
        if isinstance(rich, DataFrameComponent) and rich.rows:
            tables.append(pd.DataFrame(rich.rows))

        # Fallback generic content
        if getattr(component, "content", None):
            texts.append(component.content)

    return {"texts": texts, "tables": tables}


st.set_page_config(page_title="Promo Analytics Assistant", page_icon="🛍️", layout="wide")
st.title("🛍️ Promo Analytics Assistant")

st.markdown(
    "Ask questions about promotions, pricing, and store performance."
    "\nConfigure credentials with environment variables like `SNOWFLAKE_ACCOUNT` if you want live SQL execution."
)

if "history" not in st.session_state:
    st.session_state.history = []  # List[Dict[str, str]]

user_input = st.text_area("Your question:", height=120, key="user_query")

col1, col2 = st.columns([1, 1])
with col1:
    run_clicked = st.button("Run", type="primary")
with col2:
    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.experimental_rerun()

if run_clicked and user_input.strip():
    st.session_state.history.append({"role": "user", "text": user_input.strip()})
    with st.spinner("Thinking..."):
        result = asyncio.run(_ask_agent(user_input.strip()))

    # Append assistant response for display
    st.session_state.history.append({
        "role": "assistant",
        "text": "\n\n".join(result["texts"]) if result["texts"] else "(no textual response)",
        "tables": result["tables"],
    })

if st.session_state.history:
    st.markdown("---")
    st.subheader("Conversation")
    for entry in st.session_state.history:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['text']}")
        else:
            st.markdown(f"**Assistant:** {entry['text']}")
            for df in entry.get("tables", []):
                st.dataframe(df)

