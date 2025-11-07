"""
RAG tools for schema discovery and lookup.

These tools allow Gemini to query the Excel-backed schema knowledge base
on-demand to find tables, columns, relationships, and business terms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.components import (
    ComponentType,
    NotificationComponent,
    SimpleTextComponent,
    UiComponent,
)

from .models import ColumnDefinition
from .provider import ExcelSchemaProvider


class SearchSchemaArgs(BaseModel):
    """Arguments for search_schema tool."""

    query: str = Field(description="Search text to find relevant tables and columns")


class TableInfoArgs(BaseModel):
    """Arguments for table_info tool."""

    table_fqn: str = Field(description="Fully-qualified table name (e.g., DATABASE.SCHEMA.TABLE)")


class ColumnInfoArgs(BaseModel):
    """Arguments for column_info tool."""

    table_fqn: str = Field(description="Fully-qualified table name")
    column_name: str = Field(description="Column name")


class RelationshipsArgs(BaseModel):
    """Arguments for relationships tool."""

    table_fqn: Optional[str] = Field(
        default=None, description="Table to find relationships for (optional)"
    )
    left_table_fqn: Optional[str] = Field(
        default=None, description="Left table for relationship lookup (optional)"
    )
    right_table_fqn: Optional[str] = Field(
        default=None, description="Right table for relationship lookup (optional)"
    )


class ResolveTermArgs(BaseModel):
    """Arguments for resolve_term tool."""

    term: str = Field(description="Business term or synonym to resolve to physical columns")


class SearchSchemaTool(Tool[SearchSchemaArgs]):
    """Search for tables and columns matching a query string."""

    def __init__(self, provider: ExcelSchemaProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "search_schema"

    @property
    def description(self) -> str:
        return "Search ACROSS ALL tables and columns in the schema by name, description, business terms, or metadata. This tool searches comprehensively and returns ranked results from all tables. ALWAYS use this FIRST when exploring schema - it will show you all relevant tables and columns that match your query, even if column names appear in multiple tables. Returns results sorted by relevance."

    def get_args_schema(self) -> Type[SearchSchemaArgs]:
        return SearchSchemaArgs

    async def execute(self, context: ToolContext, args: SearchSchemaArgs) -> ToolResult:
        try:
            tables = self.provider.search_tables(args.query, limit=10)
            columns = self.provider.search_columns(args.query, limit=20)

            lines: List[str] = []
            lines.append(f"Search results for '{args.query}' (searched across ALL tables and columns):")
            lines.append("")
            
            if tables:
                lines.append(f"Relevant Tables ({len(tables)} found):")
                for t in tables:
                    desc = t.description or ""
                    lines.append(f"  - {t.table_fqn}: {desc}")
                lines.append("")
            
            if columns:
                lines.append(f"Relevant Columns ({len(columns)} found):")
                # Group columns by table for better readability
                by_table: Dict[str, List[ColumnDefinition]] = {}
                for table_fqn, c in columns:
                    by_table.setdefault(table_fqn, []).append(c)
                
                for table_fqn, cols in by_table.items():
                    lines.append(f"  Table: {table_fqn}")
                    for c in cols:
                        desc = c.description or ""
                        dtype = f" ({c.data_type})" if c.data_type else ""
                        lines.append(f"    - {c.column_name}{dtype}: {desc}")
                    lines.append("")

            if not tables and not columns:
                result = f"No tables or columns found matching '{args.query}' across the entire schema. Try different keywords or use resolve_term for business terms."
            else:
                result = "\n".join(lines).strip()

            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="info",
                        message=f"Found {len(tables)} tables and {len(columns)} columns",
                    ),
                    simple_component=SimpleTextComponent(text=result),
                ),
                metadata={"tables_found": len(tables), "columns_found": len(columns)},
            )
        except Exception as e:
            error_msg = f"Error searching schema: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_msg,
                error=str(e),
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION, level="error", message=error_msg
                    ),
                    simple_component=SimpleTextComponent(text=error_msg),
                ),
            )


class TableInfoTool(Tool[TableInfoArgs]):
    """Get detailed information about a specific table including all columns."""

    def __init__(self, provider: ExcelSchemaProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "table_info"

    @property
    def description(self) -> str:
        return "Get complete information about a table including all columns, primary keys, and descriptions. Use this when you need to see all columns in a table."

    def get_args_schema(self) -> Type[TableInfoArgs]:
        return TableInfoArgs

    async def execute(self, context: ToolContext, args: TableInfoArgs) -> ToolResult:
        try:
            table = self.provider.get_table(args.table_fqn)
            if not table:
                result = f"Table '{args.table_fqn}' not found"
                return ToolResult(
                    success=False,
                    result_for_llm=result,
                    error=result,
                    ui_component=UiComponent(
                        rich_component=NotificationComponent(
                            type=ComponentType.NOTIFICATION, level="error", message=result
                        ),
                        simple_component=SimpleTextComponent(text=result),
                    ),
                )

            lines: List[str] = []
            lines.append(f"Table: {table.table_fqn}")
            if table.description:
                lines.append(f"Description: {table.description}")
            if table.business_terms:
                lines.append(f"Business terms: {', '.join(table.business_terms)}")
            if table.primary_keys:
                lines.append(f"Primary keys: {', '.join(table.primary_keys)}")
            if table.columns:
                lines.append("\nColumns:")
                for c in table.columns:
                    dtype = f" ({c.data_type})" if c.data_type else ""
                    desc = c.description or ""
                    flags = []
                    if c.is_primary_key:
                        flags.append("PK")
                    if c.is_foreign_key:
                        flags.append("FK")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    lines.append(f"  - {c.column_name}{dtype}{flag_str}: {desc}")
                    if c.business_terms:
                        lines.append(f"    Business terms: {', '.join(c.business_terms)}")
                    if c.referenced_table_fqn:
                        lines.append(
                            f"    References: {c.referenced_table_fqn}.{c.referenced_column_name or ''}"
                        )

            result = "\n".join(lines)
            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="info",
                        message=f"Table info for {args.table_fqn}",
                    ),
                    simple_component=SimpleTextComponent(text=result),
                ),
                metadata={"table_fqn": args.table_fqn, "column_count": len(table.columns)},
            )
        except Exception as e:
            error_msg = f"Error getting table info: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_msg,
                error=str(e),
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION, level="error", message=error_msg
                    ),
                    simple_component=SimpleTextComponent(text=error_msg),
                ),
            )


class ColumnInfoTool(Tool[ColumnInfoArgs]):
    """Get detailed information about a specific column."""

    def __init__(self, provider: ExcelSchemaProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "column_info"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific column including data type, description, business terms, and relationships. IMPORTANT: If a column name might exist in multiple tables, use search_schema first to find all matches, then choose the correct table based on context before calling this tool."

    def get_args_schema(self) -> Type[ColumnInfoArgs]:
        return ColumnInfoArgs

    async def execute(self, context: ToolContext, args: ColumnInfoArgs) -> ToolResult:
        try:
            column = self.provider.get_column(args.table_fqn, args.column_name)
            if not column:
                result = f"Column '{args.table_fqn}.{args.column_name}' not found"
                return ToolResult(
                    success=False,
                    result_for_llm=result,
                    error=result,
                    ui_component=UiComponent(
                        rich_component=NotificationComponent(
                            type=ComponentType.NOTIFICATION, level="error", message=result
                        ),
                        simple_component=SimpleTextComponent(text=result),
                    ),
                )

            lines: List[str] = []
            lines.append(f"Column: {args.table_fqn}.{args.column_name}")
            if column.data_type:
                lines.append(f"Data type: {column.data_type}")
            if column.description:
                lines.append(f"Description: {column.description}")
            if column.business_terms:
                lines.append(f"Business terms: {', '.join(column.business_terms)}")
            flags = []
            if column.is_primary_key:
                flags.append("Primary Key")
            if column.is_foreign_key:
                flags.append("Foreign Key")
            if flags:
                lines.append(f"Flags: {', '.join(flags)}")
            if column.referenced_table_fqn:
                lines.append(
                    f"References: {column.referenced_table_fqn}.{column.referenced_column_name or ''}"
                )
            if column.metadata:
                lines.append("Metadata:")
                for k, v in sorted(column.metadata.items()):
                    lines.append(f"  - {k}: {v}")

            result = "\n".join(lines)
            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="info",
                        message=f"Column info for {args.table_fqn}.{args.column_name}",
                    ),
                    simple_component=SimpleTextComponent(text=result),
                ),
            )
        except Exception as e:
            error_msg = f"Error getting column info: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_msg,
                error=str(e),
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION, level="error", message=error_msg
                    ),
                    simple_component=SimpleTextComponent(text=error_msg),
                ),
            )


class RelationshipsTool(Tool[RelationshipsArgs]):
    """Find relationships (joins) between tables."""

    def __init__(self, provider: ExcelSchemaProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "relationships"

    @property
    def description(self) -> str:
        return "Find relationships (join paths) between tables. Use this to discover how to join tables together."

    def get_args_schema(self) -> Type[RelationshipsArgs]:
        return RelationshipsArgs

    async def execute(self, context: ToolContext, args: RelationshipsArgs) -> ToolResult:
        try:
            rels: List = []
            if args.left_table_fqn and args.right_table_fqn:
                rels = self.provider.find_relationship_between(
                    args.left_table_fqn, args.right_table_fqn
                )
            elif args.table_fqn:
                rels = self.provider.find_relationships(args.table_fqn)
            else:
                result = "Please provide either 'table_fqn' or both 'left_table_fqn' and 'right_table_fqn'"
                return ToolResult(
                    success=False,
                    result_for_llm=result,
                    error=result,
                    ui_component=UiComponent(
                        rich_component=NotificationComponent(
                            type=ComponentType.NOTIFICATION, level="error", message=result
                        ),
                        simple_component=SimpleTextComponent(text=result),
                    ),
                )

            if not rels:
                result = "No relationships found"
                if args.left_table_fqn and args.right_table_fqn:
                    result = f"No relationships found between {args.left_table_fqn} and {args.right_table_fqn}"
                elif args.table_fqn:
                    result = f"No relationships found for {args.table_fqn}"
            else:
                lines: List[str] = []
                for r in rels:
                    line = f"{r.left_table_fqn}.{r.left_column} = {r.right_table_fqn}.{r.right_column}"
                    if r.cardinality:
                        line += f" [{r.cardinality}]"
                    if r.predicate:
                        line += f" :: {r.predicate}"
                    if r.description:
                        line += f" ({r.description})"
                    lines.append(line)
                result = "\n".join(lines)

            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="info",
                        message=f"Found {len(rels)} relationship(s)",
                    ),
                    simple_component=SimpleTextComponent(text=result),
                ),
                metadata={"relationship_count": len(rels)},
            )
        except Exception as e:
            error_msg = f"Error finding relationships: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_msg,
                error=str(e),
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION, level="error", message=error_msg
                    ),
                    simple_component=SimpleTextComponent(text=error_msg),
                ),
            )


class ResolveTermTool(Tool[ResolveTermArgs]):
    """Resolve a business term or synonym to physical column locations."""

    def __init__(self, provider: ExcelSchemaProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "resolve_term"

    @property
    def description(self) -> str:
        return "Resolve a business term or synonym to physical table.column locations. Use this when the user mentions a business term that might map to specific columns."

    def get_args_schema(self) -> Type[ResolveTermArgs]:
        return ResolveTermArgs

    async def execute(self, context: ToolContext, args: ResolveTermArgs) -> ToolResult:
        try:
            candidates = self.provider.resolve_term(args.term, limit=20)
            if not candidates:
                result = f"No columns found for business term '{args.term}'"
            else:
                lines: List[str] = []
                lines.append(f"Business term '{args.term}' maps to:")
                for table_fqn, col_name in candidates:
                    c = self.provider.get_column(table_fqn, col_name)
                    if c:
                        desc = c.description or ""
                        dtype = f" ({c.data_type})" if c.data_type else ""
                        lines.append(f"- {table_fqn}.{col_name}{dtype}: {desc}")
                    else:
                        lines.append(f"- {table_fqn}.{col_name}")
                result = "\n".join(lines)

            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="info",
                        message=f"Resolved '{args.term}' to {len(candidates)} column(s)",
                    ),
                    simple_component=SimpleTextComponent(text=result),
                ),
                metadata={"term": args.term, "candidate_count": len(candidates)},
            )
        except Exception as e:
            error_msg = f"Error resolving term: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_msg,
                error=str(e),
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION, level="error", message=error_msg
                    ),
                    simple_component=SimpleTextComponent(text=error_msg),
                ),
            )

