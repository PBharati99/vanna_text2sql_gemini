"""
Example: Gemini agent with Excel-backed schema RAG tools and Snowflake SQL execution.

This example shows how to:
1. Load schema from Excel knowledge base
2. Register RAG tools (search_schema, table_info, column_info, relationships, resolve_term)
3. Register run_sql tool with Snowflake integration
4. Use schema-aware prompt builder
5. Run queries with Gemini that can discover schema and execute SQL

Requirements:
- GOOGLE_API_KEY or GEMINI_API_KEY environment variable
- pandas, openpyxl: pip install pandas openpyxl
- snowflake-connector-python: pip install 'vanna[snowflake]'
- Excel knowledge base file
- Snowflake credentials (configured in the script)

Usage:
  PYTHONPATH=src python vanna/examples/gemini_excel_schema_example.py \
    "./Promo Semantic Data_20250519.xlsx"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from vanna import Agent
    from vanna.core.user.models import User

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("vanna.integrations.gemini").setLevel(logging.DEBUG)
logging.getLogger("vanna.tools.run_sql").setLevel(logging.DEBUG)  # Enable SQL query logging

try:
    from vanna import Agent
    from vanna.core.agent.config import AgentConfig
    from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
    from vanna.core.registry import ToolRegistry
    from vanna.core.system_prompt import SystemPromptBuilder
    from vanna.core.tool.models import ToolSchema
    from vanna.core.user import CookieEmailUserResolver, RequestContext
    from vanna.core.user.models import User
    from vanna.integrations.gemini import GeminiLlmService
    from vanna.integrations.snowflake import SnowflakeRunner
    from vanna.schema.provider import ExcelSchemaProvider
    from vanna.schema.prompt_builder import ExcelSchemaPromptBuilder
    from vanna.schema.tools import (
        ColumnInfoTool,
        RelationshipsTool,
        ResolveTermTool,
        SearchSchemaTool,
        TableInfoTool,
    )
    from vanna.tools.run_sql import RunSqlTool
except ImportError as e:
    print(f"[error] Import failed: {e}")
    print("Ensure PYTHONPATH includes 'src' and dependencies are installed.")
    sys.exit(1)


class ExcelSchemaSystemPromptBuilder(SystemPromptBuilder):
    """System prompt builder that uses ExcelSchemaPromptBuilder."""

    def __init__(self, prompt_builder: ExcelSchemaPromptBuilder):
        self.prompt_builder = prompt_builder

    async def build_system_prompt(
        self, user: User, tools: List[ToolSchema]
    ) -> Optional[str]:
        # Check if run_sql tool is available
        tool_names = [tool.name for tool in tools]
        has_sql_tool = "run_sql" in tool_names

        # Build base prompt with SQL-aware instructions
        base_prompt = self.prompt_builder.build_system_prompt(
            allow_sql_execution=has_sql_tool
        )

        # Add additional instructions if SQL tool is available
        if has_sql_tool:
            sql_instructions = """

================================================================================
SQL EXECUTION (AVAILABLE):
================================================================================
After discovering the schema, you MUST:
1. Generate a SQL SELECT query to answer the user's question
2. Call the run_sql tool with the generated SQL query
3. Present the results to the user

CRITICAL SQL FORMATTING RULES:
- The SQL string passed to run_sql must be PURE SQL ONLY - no markdown, no comments, no explanations
- Do NOT wrap SQL in ```sql ... ``` code blocks
- Do NOT include SQL comments (-- or /* */) in the query
- Do NOT include explanatory text before or after the SQL
- The sql parameter should contain ONLY the executable SQL query

Example CORRECT format:
  run_sql(sql="SELECT STORE_NBR, COUNT(*) FROM APP_PROMOFCST.PROMO_FORECAST.NML_AD_STR_SKU_HIST GROUP BY STORE_NBR")

Example WRONG format (DO NOT DO THIS):
  run_sql(sql="```sql
  -- This query counts promotions by store
  SELECT STORE_NBR, COUNT(*) FROM ...
  ```")

IMPORTANT:
- Generate complete, executable SELECT queries
- Use fully-qualified table names (DATABASE.SCHEMA.TABLE) from the schema
- Include proper JOINs based on discovered relationships
- Add appropriate WHERE clauses, GROUP BY, ORDER BY as needed
- DO NOT just document the schema - you must execute SQL to answer the question!
"""
            return base_prompt + sql_instructions

        return base_prompt


class LoggingSqlRunner(SqlRunner):
    """SqlRunner that logs SQL instead of executing it."""

    async def run_sql(self, args: RunSqlToolArgs, context) -> pd.DataFrame:  # type: ignore[override]
        print("\n================ LLM GENERATED SQL ================")
        print(args.sql)
        print("================ END LLM SQL =====================\n")
        return pd.DataFrame()


def create_excel_schema_agent(
    excel_path: str,
    gemini_api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash-exp",
    max_tool_iterations: int = 20,
    snowflake_config: Optional[dict] = None,
) -> Agent:
    """Create a Gemini agent with Excel-backed schema RAG tools and optional Snowflake SQL execution.

    Args:
        excel_path: Path to Excel knowledge base file
        gemini_api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        model_name: Gemini model name
        max_tool_iterations: Maximum tool call iterations
        snowflake_config: Optional Snowflake connection config dict with keys:
            - account: Snowflake account identifier
            - user: Database user
            - role: Snowflake role (optional)
            - warehouse: Snowflake warehouse (optional)
            - authenticator: Authentication method (e.g., 'externalbrowser')
            - database: Database name (optional)
            - password: Password (optional, not needed for external browser auth)

    Returns:
        Configured Agent instance
    """
    # Load schema provider
    provider = ExcelSchemaProvider(excel_path)

    # Create prompt builder
    prompt_builder = ExcelSchemaPromptBuilder(
        provider=provider,
        max_tables=25,
        max_columns_per_table=20,
        include_relationships=True,
        include_business_terms=True,
    )

    # Get Gemini API key
    api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )

    # Create LLM service
    llm_service = GeminiLlmService(api_key=api_key, model_name=model_name)

    # Create tool registry and register RAG tools
    registry = ToolRegistry()
    registry.register(SearchSchemaTool(provider))
    registry.register(TableInfoTool(provider))
    registry.register(ColumnInfoTool(provider))
    registry.register(RelationshipsTool(provider))
    registry.register(ResolveTermTool(provider))

    # Register run_sql tool
    if snowflake_config:
        try:
            # Create Snowflake runner
            snowflake_runner = SnowflakeRunner(
                account=snowflake_config['account'],
                username=snowflake_config['user'],
                role=snowflake_config.get('role'),
                warehouse=snowflake_config.get('warehouse'),
                authenticator=snowflake_config.get('authenticator', 'externalbrowser'),
                database=snowflake_config.get('database', ''),
                password=snowflake_config.get('password'),
            )
            
            # Register run_sql tool
            run_sql_tool = RunSqlTool(sql_runner=snowflake_runner)
            registry.register(run_sql_tool)
            print("[info] Snowflake SQL execution enabled")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Snowflake: {e}")
            print("[ERROR] SQL execution will NOT be available. To fix this:")
            print("  1. Install Snowflake connector: pip install snowflake-connector-python")
            print("  2. Or install with extras: pip install 'vanna[snowflake]'")
            print("[info] Falling back to logging-only SQL runner.")
            logging_runner = LoggingSqlRunner()
            registry.register(RunSqlTool(sql_runner=logging_runner))
    else:
        logging_runner = LoggingSqlRunner()
        registry.register(RunSqlTool(sql_runner=logging_runner))
        print("[info] Snowflake disabled; logging SQL queries instead of executing them.")

    # Create system prompt builder (will detect SQL tool from available tools)
    system_prompt_builder = ExcelSchemaSystemPromptBuilder(prompt_builder)

    # Create user resolver
    user_resolver = CookieEmailUserResolver()

    # Create agent config
    config = AgentConfig(
        max_tool_iterations=max_tool_iterations,
        stream_responses=False,  # Disable streaming for simpler output
    )

    # Create agent
    agent = Agent(
        llm_service=llm_service,
        tool_registry=registry,
        user_resolver=user_resolver,
        config=config,
        system_prompt_builder=system_prompt_builder,
    )

    return agent


async def main():
    """Example usage of Excel schema agent with Snowflake integration."""
    if len(sys.argv) < 2:
        print("Usage: python vanna/examples/gemini_excel_schema_example.py <excel-path>")
        print("Example: python vanna/examples/gemini_excel_schema_example.py './Promo Semantic Data_20250519.xlsx'")
        sys.exit(1)

    excel_path = sys.argv[1]
    if not os.path.exists(excel_path):
        print(f"[error] Excel file not found: {excel_path}")
        sys.exit(1)

    # Snowflake configuration (set to None to disable execution and log SQL instead)
    # snowflake_config = None
    snowflake_config = {
        'account': 'cvs-cvsretailprod',
        'user': 'bharati.peddinti@CVSHealth.com',
        'role': 'GRP-CN-SCAI-ANALYTICS',
        'warehouse': 'WH_SCAI_ANALYTICS_L_QUERY_01',
        'authenticator': 'externalbrowser',
    }

    # Create agent
    print(f"\n[info] Loading schema from: {excel_path}")
    agent = create_excel_schema_agent(excel_path, snowflake_config=snowflake_config)

    # Create request context with test user email
    request_context = RequestContext(
        cookies={"vanna_email": "test@example.com"},
        metadata={"demo": True},
        remote_addr="127.0.0.1",
    )

    # Example queries
    queries = [
        # "How did promotions vary across different stores?",
        # "Which promotion types are most common for store 123?",
        # "List the top 5 stores by number of promotions in Q1 2025",
        "What is the average sale price during promotions for SKU 98765?",
        # "Show promotional activity for stores in the Northeast region",
    ]

    conversation_id = "excel-schema-test"

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")

        current_conversation_id = f"{conversation_id}-{i}"

        try:
            print("\nResponse:")
            async for component in agent.send_message(
                request_context=request_context,
                message=query,
                conversation_id=current_conversation_id,
            ):
                # Print component content
                if hasattr(component, "simple_component") and component.simple_component:
                    if hasattr(component.simple_component, "text"):
                        print(component.simple_component.text)
                elif hasattr(component, "rich_component") and component.rich_component:
                    if hasattr(component.rich_component, "content") and component.rich_component.content:
                        print(component.rich_component.content)
                elif hasattr(component, "content") and component.content:
                    print(component.content)

        except Exception as e:
            print(f"\n[error] Query failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

