"""
Gemini example using the SQL query tool with the Chinook database.

This example demonstrates using the RunSqlTool with SqliteRunner and Google Gemini AI
to intelligently query and analyze the Chinook database.

Requirements:
- GOOGLE_API_KEY or GEMINI_API_KEY environment variable or .env file
- google-generativeai package: pip install 'vanna[gemini]'

Usage:
  PYTHONPATH=. python vanna/examples/gemini_sqlite_example.py
"""

import asyncio
import importlib.util
import logging
import os
import sqlite3
import sys
from typing import TYPE_CHECKING, List, Optional

# Enable debug logging for Gemini integration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set specific loggers to DEBUG
logging.getLogger('vanna.integrations.gemini').setLevel(logging.DEBUG)

if TYPE_CHECKING:
    from vanna import Agent
    from vanna.core.system_prompt import SystemPromptBuilder
    from vanna.core.tool.models import ToolSchema
    from vanna.core.user.models import User


class SchemaAwareSystemPromptBuilder:
    """System prompt builder that includes SQLite database schema information.
    
    This builder extends the default system prompt with database schema information,
    including table names and column names, to help the LLM use correct table names.
    """
    
    def __init__(self, database_path: str):
        """Initialize with database path.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self._schema_info: Optional[str] = None
    
    def _get_schema_info(self) -> str:
        """Get schema information from SQLite database."""
        if self._schema_info:
            return self._schema_info
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_parts = ["DATABASE SCHEMA:", "=" * 60]
            schema_parts.append(f"Available tables: {', '.join(tables)}")
            schema_parts.append("")
            schema_parts.append("IMPORTANT: Use the EXACT table names shown above (case-sensitive).")
            schema_parts.append("")
            schema_parts.append("Table details:")
            
            # Get columns for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                schema_parts.append(f"  - {table}: {', '.join(column_names)}")
            
            conn.close()
            
            self._schema_info = "\n".join(schema_parts)
            return self._schema_info
        except Exception as e:
            return f"[Schema info unavailable: {e}]"
    
    async def build_system_prompt(
        self, user: "User", tools: List["ToolSchema"]
    ) -> Optional[str]:
        """Build system prompt with schema information."""
        from vanna.core.system_prompt import DefaultSystemPromptBuilder
        
        # Get base prompt from default builder
        default_builder = DefaultSystemPromptBuilder()
        base_prompt = await default_builder.build_system_prompt(user, tools)
        
        # Add schema information
        schema_info = self._get_schema_info()
        
        if base_prompt:
            return f"{base_prompt}\n\n{schema_info}"
        else:
            return schema_info


def ensure_env() -> None:
    """Ensure environment is properly configured."""
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv

        # Load from local .env without overriding existing env
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
    else:
        print(
            "[warn] python-dotenv not installed; skipping .env load. Install with: pip install python-dotenv"
        )

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "[error] GOOGLE_API_KEY or GEMINI_API_KEY is not set. Add it to your environment or .env file."
        )
        sys.exit(1)


async def main() -> None:
    """Main function to run the Gemini SQLite example."""
    ensure_env()

    try:
        from vanna.integrations.gemini import GeminiLlmService
    except ImportError:
        print(
            "[error] gemini integration not available. Make sure vanna/integrations/gemini/llm.py exists."
        )
        raise

    from vanna import AgentConfig, Agent
    from vanna.core.registry import ToolRegistry
    from vanna.core.user import CookieEmailUserResolver, RequestContext
    from vanna.integrations.sqlite import SqliteRunner
    from vanna.tools import (
        RunSqlTool,
        LocalFileSystem,
    )

    # Get the path to the Chinook database
    database_path = os.path.join(os.path.dirname(__file__), "..", "..", "Chinook.sqlite")
    database_path = os.path.abspath(database_path)
    
    # Also check if it's in src/vanna directory
    if not os.path.exists(database_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "Chinook.sqlite")
        if os.path.exists(alt_path):
            database_path = os.path.abspath(alt_path)

    if not os.path.exists(database_path):
        print(f"[error] Chinook database not found at {database_path}")
        print("Please download it with: curl -o Chinook.sqlite https://vanna.ai/Chinook.sqlite")
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"Using Gemini model: {model}")
    print(f"Using database: {database_path}")

    # Create Gemini LLM service
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    llm = GeminiLlmService(model=model, api_key=api_key)

    # Create shared FileSystem for tools
    file_system = LocalFileSystem(working_directory="./gemini_data")

    # Create tool registry and register the SQL tool with SQLite runner
    tool_registry = ToolRegistry()
    sqlite_runner = SqliteRunner(database_path=database_path)
    sql_tool = RunSqlTool(sql_runner=sqlite_runner, file_system=file_system)
    tool_registry.register(sql_tool)

    user_resolver = CookieEmailUserResolver()

    # Create schema-aware system prompt builder
    system_prompt_builder = SchemaAwareSystemPromptBuilder(database_path=database_path)
    
    agent = Agent(
        llm_service=llm,
        config=AgentConfig(
            stream_responses=False,
            max_tool_iterations=20  # Increase limit to allow more tool calls if needed
        ),
        tool_registry=tool_registry,
        user_resolver=user_resolver,
        system_prompt_builder=system_prompt_builder,
    )

    # Simulate a logged-in demo user via cookie-based resolver
    request_context = RequestContext(
        cookies={user_resolver.cookie_name: "demo-user@example.com"},
        metadata={"demo": True},
        remote_addr="127.0.0.1",
    )
    conversation_id = "gemini-sqlite-demo"

    # Sample queries to demonstrate different capabilities
    sample_questions = [
        "What tables are in this database?",
        "Show me the first 5 customers with their names",
        "What's the total number of tracks in the database?",
        "Find the top 5 artists by number of albums",
        "What's the average invoice total?",
        "Find the lowest 5 albums by price",
        "Which artist has the least number of albums?",
        "I want to know the total revenue for each genre",
    ]

    print("\n" + "="*60)
    print("Gemini SQLite Database Assistant Demo")
    print("="*60)
    print("This demo shows Gemini querying the Chinook music database.")
    print("Gemini will intelligently construct SQL queries to answer questions.")
    print()

    for i, question in enumerate(sample_questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        # Use a unique conversation ID for each question to avoid context pollution
        current_conversation_id = f"{conversation_id}-{i}"

        async for component in agent.send_message(
            request_context=request_context,
            message=question,
            conversation_id=current_conversation_id,
        ):
            # Handle different component types
            if hasattr(component, "simple_component") and component.simple_component:
                if hasattr(component.simple_component, "text"):
                    print("Assistant:", component.simple_component.text)
            elif hasattr(component, "rich_component") and component.rich_component:
                if hasattr(component.rich_component, "content") and component.rich_component.content:
                    print("Assistant:", component.rich_component.content)
            elif hasattr(component, "content") and component.content:
                print("Assistant:", component.content)

        print()  # Add spacing between questions

    print("\n" + "="*60)
    print("Demo complete! Gemini successfully queried the database.")
    print("="*60)


def create_demo_agent() -> "Agent":
    """Create a demo agent with Gemini and SQLite query tool.

    This function is called by the vanna server framework.

    Returns:
        Configured Agent with Gemini LLM and SQLite tool
    """
    ensure_env()

    try:
        from vanna.integrations.gemini import GeminiLlmService
    except ImportError:
        print(
            "[error] gemini integration not available. Make sure vanna/integrations/gemini/llm.py exists."
        )
        raise

    from vanna import AgentConfig, Agent
    from vanna.core.registry import ToolRegistry
    from vanna.core.user import CookieEmailUserResolver
    from vanna.integrations.sqlite import SqliteRunner
    from vanna.tools import (
        RunSqlTool,
        LocalFileSystem,
    )

    # Get the path to the Chinook database
    database_path = os.path.join(os.path.dirname(__file__), "..", "..", "Chinook.sqlite")
    database_path = os.path.abspath(database_path)
    
    # Also check if it's in src/vanna directory
    if not os.path.exists(database_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "Chinook.sqlite")
        if os.path.exists(alt_path):
            database_path = os.path.abspath(alt_path)

    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Chinook database not found at {database_path}. Please download it from https://vanna.ai/Chinook.sqlite")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    llm = GeminiLlmService(model=model, api_key=api_key)

    # Create shared FileSystem for tools
    file_system = LocalFileSystem(working_directory="./gemini_data")

    # Create tool registry and register the SQL tool with SQLite runner
    tool_registry = ToolRegistry()
    sqlite_runner = SqliteRunner(database_path=database_path)
    sql_tool = RunSqlTool(sql_runner=sqlite_runner, file_system=file_system)
    tool_registry.register(sql_tool)

    user_resolver = CookieEmailUserResolver()

    # Create schema-aware system prompt builder
    system_prompt_builder = SchemaAwareSystemPromptBuilder(database_path=database_path)

    return Agent(
        llm_service=llm,
        config=AgentConfig(stream_responses=True),  # Enable streaming for web interface
        tool_registry=tool_registry,
        user_resolver=user_resolver,
        system_prompt_builder=system_prompt_builder,
    )


if __name__ == "__main__":
    asyncio.run(main())

