from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd

from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs

class GetSampleDataArgs(BaseModel):
    table_name: str = Field(description="Fully qualified table name")
    column_name: str = Field(description="Column name to sample")

class SampleDataTool(Tool[GetSampleDataArgs]):
    """
    Tool to fetch actual data samples from the database to understand values/formats.
    Useful for checking categorical values (e.g. 'Active' vs 'A') or date formats.
    """
    def __init__(self, sql_runner: SqlRunner):
        self.sql_runner = sql_runner

    @property
    def name(self) -> str:
        return "get_sample_data"

    @property
    def description(self) -> str:
        return "Fetch 5 distinct non-null sample values for a specific column to understand its content."

    def get_args_schema(self) -> Type[GetSampleDataArgs]:
        return GetSampleDataArgs

    async def execute(self, context: ToolContext, args: GetSampleDataArgs) -> ToolResult:
        # Construct safe sampling query
        # Use TRY_CAST or simple selection. 
        # We want DISTINCT values to be useful.
        
        # Validate input strictly to prevent SQL injection (though args come from LLM)
        # Basic cleanup
        tbl = args.table_name.replace(";", "").replace("--", "")
        col = args.column_name.replace(";", "").replace("--", "")
        
        sql = f'SELECT DISTINCT "{col}" FROM {tbl} WHERE "{col}" IS NOT NULL LIMIT 5'
        # Note: Quoting identifiers to be safe with Snowflake case sensitivity if needed, 
        # or assume LLM provides correct case. Snowflake is usually case-insensitive unless quoted.
        # Let's try without quotes first, or use the runner's quote method if exposed.
        # sql_runner doesn't expose quote method in interface.
        
        # Simpler SQL without quotes (Snowflake handles standard names fine)
        sql = f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT 5"

        try:
            # We need to wrap this in RunSqlToolArgs
            run_args = RunSqlToolArgs(sql=sql)
            # run_sql is async
            df = await self.sql_runner.run_sql(run_args, context)
            
            if df.empty:
                return ToolResult(success=True, result_for_llm="No non-null values found.")
            
            # Format as string list
            values = df[col].tolist() if col in df.columns else df.iloc[:, 0].tolist()
            return ToolResult(
                success=True,
                result_for_llm=f"Sample values for {col}: {str(values)}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result_for_llm=f"Failed to fetch samples: {str(e)}",
                error=str(e)
            )

