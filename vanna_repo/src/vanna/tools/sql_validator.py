from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp

from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.schema.provider import ExcelSchemaProvider

class ValidateSqlArgs(BaseModel):
    sql: str = Field(description="The SQL query to validate")

class ValidateSqlTool(Tool[ValidateSqlArgs]):
    """
    Tool to validate SQL queries against the schema before execution.
    Checks if tables and columns actually exist.
    """
    def __init__(self, provider: ExcelSchemaProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "validate_sql"

    @property
    def description(self) -> str:
        return "Validate a SQL query against the known schema to check for missing tables or columns."

    def get_args_schema(self) -> Type[ValidateSqlArgs]:
        return ValidateSqlArgs

    def execute(self, context: ToolContext, args: ValidateSqlArgs) -> ToolResult:
        try:
            # Parse SQL
            parsed = sqlglot.parse_one(args.sql, read="snowflake")
        except Exception as e:
            return ToolResult(
                success=False,
                result_for_llm=f"SQL Parsing Error: {str(e)}",
                error=str(e)
            )

        errors = []
        
        # 1. Extract and Validate Tables
        # sqlglot finds tables as exp.Table
        tables_found = []
        for table in parsed.find_all(exp.Table):
            # Normalize table name to FQN if possible, or just name
            # sqlglot table parts: catalog (db), db (schema), this (table name)
            # Note: attributes might be None
            catalog = table.catalog
            schema = table.db
            name = table.this
            
            def _get_name(item):
                if hasattr(item, 'name'): return item.name
                return str(item)

            parts = [_get_name(p) for p in [catalog, schema, name] if p]
            if not parts:
                continue
                
            # Construct FQN candidates to check
            # Case 1: Full FQN in SQL (db.schema.table)
            if len(parts) == 3:
                fqn = f"{parts[0]}.{parts[1]}.{parts[2]}"
                if not self.provider.get_table(fqn):
                    # Try case-insensitive lookup if direct lookup fails
                    found = False
                    for known_fqn in self.provider.schema.tables:
                        if known_fqn.lower() == fqn.lower():
                            found = True
                            break
                    if not found:
                        errors.append(f"Table not found in schema: {fqn}")
                else:
                    tables_found.append(fqn)
            
            # Case 2: Schema.Table
            elif len(parts) == 2:
                # Ambiguous, check if any known table ends with schema.table
                partial = f"{parts[0]}.{parts[1]}"
                matches = [t for t in self.provider.schema.tables if t.endswith(partial) or t.lower().endswith(partial.lower())]
                if not matches:
                    errors.append(f"Table not found in schema: {partial}")
                else:
                    tables_found.extend(matches)
            
            # Case 3: Table Name only
            elif len(parts) == 1:
                tname = parts[0]
                matches = [t for t in self.provider.schema.tables if t.split('.')[-1].lower() == tname.lower()]
                if not matches:
                    errors.append(f"Table not found in schema: {tname}")
                else:
                    tables_found.extend(matches)

        # 2. Extract and Validate Columns
        # This is harder because we need to know which table a column belongs to (alias resolution).
        # For a simple validation, we can check if the column exists in ANY of the referenced tables.
        # If it exists in none, it's definitely an error.
        
        # Get all valid columns from the identified tables
        valid_columns = set()
        for table_fqn in tables_found:
            cols = self.provider.list_columns(table_fqn)
            for c in cols:
                valid_columns.add(c.lower())
        
        for col in parsed.find_all(exp.Column):
            col_name = col.name.lower()
            if col_name == '*': 
                continue
                
            # If column has a table alias, we should ideally resolve it, but that's complex.
            # Simple check: Is this column name present in ANY of the tables we found?
            if col_name not in valid_columns:
                # Check if it's a known alias or derived column?
                # sqlglot might identify aliases as columns in some contexts, strictly speaking 
                # we should look at SELECT expressions vs WHERE/GROUP BY.
                # For safety, let's check if it matches any business term too?
                # Or just be strict.
                
                # IGNORE validation for now if list of tables is empty (cascading error)
                if not tables_found:
                    continue
                    
                # Relaxed check: maybe it's not in the valid_columns set because logic above failed?
                # Let's double check against ALL schema columns if we want to be loose, 
                # OR stick to strict check against found tables. Strict is better for preventing hallucinations.
                
                errors.append(f"Column potentially invalid: '{col.name}'. Not found in tables: {', '.join(tables_found)}")

        if errors:
            # Deduplicate errors
            unique_errors = list(set(errors))
            return ToolResult(
                success=False,
                result_for_llm=f"SQL Validation Failed:\n- " + "\n- ".join(unique_errors) + "\nPlease correct the table/column names.",
                error="\n".join(unique_errors)
            )

        return ToolResult(
            success=True,
            result_for_llm="SQL is valid.",
            ui_component=None
        )

