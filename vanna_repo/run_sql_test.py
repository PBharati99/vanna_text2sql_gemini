"""Quick sanity check for RunSqlTool using in-memory SQLite."""

import asyncio
import logging
import os
import tempfile

from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.core.tool import ToolContext
from vanna.core.user.models import User
from vanna.integrations.sqlite import SqliteRunner
from vanna.tools.run_sql import RunSqlTool


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Use a temporary on-disk SQLite database so the connection persists across calls
    temp_db_path = os.path.join(tempfile.gettempdir(), "run_sql_test.db")
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

    runner = SqliteRunner(temp_db_path)
    tool = RunSqlTool(sql_runner=runner)

    # Minimal tool context (what Agent normally provides)
    context = ToolContext(
        user=User(id="test-user", username="test"),
        conversation_id="test-conversation",
        request_id="test-request",
    )

    # Create table and seed data directly via runner (validator blocks DDL)
    await runner.run_sql(
        RunSqlToolArgs(
            sql="""
            CREATE TABLE stores (
                store_nbr INTEGER,
                ad_dt TEXT
            )
            """
        ),
        context,
    )

    await runner.run_sql(
        RunSqlToolArgs(
            sql="""
            INSERT INTO stores (store_nbr, ad_dt)
            VALUES
                (1, '2025-01-01'),
                (1, '2025-01-02'),
                (2, '2025-01-03'),
                (2, '2025-01-04'),
                (3, '2025-01-05')
            """
        ),
        context,
    )

    # Execute SELECT via RunSqlTool (passes through validator + cleaning)
    query = (
        "SELECT store_nbr, COUNT(*) AS num_promos "
        "FROM stores "
        "GROUP BY store_nbr "
        "ORDER BY num_promos DESC"
    )

    result = await tool.execute(context, RunSqlToolArgs(sql=query))

    print("Result for LLM:")
    print(result.result_for_llm)

    if result.ui_component and result.ui_component.simple_component:
        print("\nSimple component text:")
        print(result.ui_component.simple_component.text)


    # Clean up temp DB file
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)


if __name__ == "__main__":
    asyncio.run(main())

