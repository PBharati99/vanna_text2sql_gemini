"""Generic SQL query execution tool with dependency injection."""
from typing import Optional, Type
import uuid
import logging
from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.components import UiComponent, DataFrameComponent, NotificationComponent, ComponentType, SimpleTextComponent
from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
from vanna.capabilities.file_system import FileSystem
from vanna.integrations.local import LocalFileSystem
from vanna.schema.sql_validator import validate_sql_readonly, clean_sql_query

logger = logging.getLogger(__name__)


class RunSqlTool(Tool[RunSqlToolArgs]):
    """Tool that executes SQL queries using an injected SqlRunner implementation."""

    def __init__(
        self,
        sql_runner: SqlRunner,
        file_system: Optional[FileSystem] = None,
        custom_tool_name: Optional[str] = None,
        custom_tool_description: Optional[str] = None
    ):
        """Initialize the tool with a SqlRunner implementation.

        Args:
            sql_runner: SqlRunner implementation that handles actual query execution
            file_system: FileSystem implementation for saving results (defaults to LocalFileSystem)
            custom_tool_name: Optional custom name for the tool (overrides default "run_sql")
            custom_tool_description: Optional custom description for the tool (overrides default description)
        """
        self.sql_runner = sql_runner
        self.file_system = file_system or LocalFileSystem()
        self._custom_name = custom_tool_name
        self._custom_description = custom_tool_description

    @property
    def name(self) -> str:
        return self._custom_name if self._custom_name else "run_sql"

    @property
    def description(self) -> str:
        return self._custom_description if self._custom_description else "Execute SQL queries against the configured database"

    def get_args_schema(self) -> Type[RunSqlToolArgs]:
        return RunSqlToolArgs

    async def execute(self, context: ToolContext, args: RunSqlToolArgs) -> ToolResult:
        """Execute a SQL query using the injected SqlRunner."""
        try:
            # Clean the SQL query (remove markdown, comments, etc.)
            cleaned_sql = clean_sql_query(args.sql)
            
            # Log the SQL being executed for debugging
            logger.info(f"[run_sql] Executing SQL query (original length: {len(args.sql)} chars, cleaned length: {len(cleaned_sql)} chars)")
            logger.debug(f"[run_sql] Original SQL:\n{args.sql}")
            logger.debug(f"[run_sql] Cleaned SQL:\n{cleaned_sql}")
            logger.debug(f"[run_sql] Cleaned SQL (repr): {cleaned_sql!r}")
            codepoints = ' '.join(f"{ord(ch)}" for ch in cleaned_sql[:50])
            logger.debug(f"[run_sql] Cleaned SQL first 50 chars codepoints: {codepoints}")
            
            # Validate SQL is read-only (SELECT only) - validation also cleans the SQL
            is_valid, error_msg = validate_sql_readonly(cleaned_sql)
            if not is_valid:
                error_message = f"SQL validation failed: {error_msg}"
                return ToolResult(
                    success=False,
                    result_for_llm=error_message,
                    ui_component=UiComponent(
                        rich_component=NotificationComponent(
                            type=ComponentType.NOTIFICATION,
                            level="error",
                            message=error_message
                        ),
                        simple_component=SimpleTextComponent(text=error_message)
                    ),
                    error=error_msg,
                    metadata={"error_type": "sql_validation_error", "query_blocked": True}
                )
            
            # Create a new args object with cleaned SQL
            from vanna.capabilities.sql_runner import RunSqlToolArgs as RunSqlToolArgsType
            cleaned_args = RunSqlToolArgsType(sql=cleaned_sql)
            
            # Use the injected SqlRunner to execute the cleaned query
            df = await self.sql_runner.run_sql(cleaned_args, context)

            # Determine query type (use cleaned SQL)
            query_type = cleaned_sql.strip().upper().split()[0] if cleaned_sql.strip() else "SELECT"

            if query_type == "SELECT":
                # Handle SELECT queries with results
                if df.empty:
                    result = "Query executed successfully. No rows returned."
                    ui_component = UiComponent(
                        rich_component=DataFrameComponent(
                            rows=[],
                            columns=[],
                            title="Query Results",
                            description="No rows returned"
                        ),
                        simple_component=SimpleTextComponent(text=result)
                    )
                    metadata = {
                        "row_count": 0,
                        "columns": [],
                        "query_type": query_type,
                        "results": []
                    }
                else:
                    # Convert DataFrame to records
                    results_data = df.to_dict('records')
                    columns = df.columns.tolist()
                    row_count = len(df)

                    # Write DataFrame to CSV file for downstream tools
                    file_id = str(uuid.uuid4())[:8]
                    filename = f"query_results_{file_id}.csv"
                    csv_content = df.to_csv(index=False)
                    await self.file_system.write_file(filename, csv_content, context, overwrite=True)

                    # Create result text for LLM with truncated results
                    results_preview = csv_content
                    if len(results_preview) > 1000:
                        results_preview = results_preview[:1000] + "\n(Results truncated to 1000 characters. FOR LARGE RESULTS YOU DO NOT NEED TO SUMMARIZE THESE RESULTS OR PROVIDE OBSERVATIONS. THE NEXT STEP SHOULD BE A VISUALIZE_DATA CALL)"

                    result = f"{results_preview}\n\nResults saved to file: {filename}\n\n**IMPORTANT: FOR VISUALIZE_DATA USE FILENAME: {filename}**"

                    # Create DataFrame component for UI
                    dataframe_component = DataFrameComponent.from_records(
                        records=results_data,
                        title="Query Results",
                        description=f"SQL query returned {row_count} rows with {len(columns)} columns"
                    )

                    ui_component = UiComponent(
                        rich_component=dataframe_component,
                        simple_component=SimpleTextComponent(text=result)
                    )

                    metadata = {
                        "row_count": row_count,
                        "columns": columns,
                        "query_type": query_type,
                        "results": results_data,
                        "output_file": filename
                    }
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                # The SqlRunner should return a DataFrame with affected row count
                rows_affected = len(df) if not df.empty else 0
                result = f"Query executed successfully. {rows_affected} row(s) affected."

                metadata = {
                    "rows_affected": rows_affected,
                    "query_type": query_type
                }
                ui_component = UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="success",
                        message=result
                    ),
                    simple_component=SimpleTextComponent(text=result)
                )

            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=ui_component,
                metadata=metadata
            )

        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            logger.error(f"[run_sql] SQL execution failed: {error_message}")
            logger.error(f"[run_sql] Failed SQL query:\n{args.sql}")
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message
                    ),
                    simple_component=SimpleTextComponent(text=error_message)
                ),
                error=str(e),
                metadata={"error_type": "sql_error"}
            )
