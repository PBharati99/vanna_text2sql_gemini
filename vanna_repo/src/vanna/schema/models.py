"""
Data models for ingesting a business glossary into a normalized schema.

These models capture:
- Schemas, tables, columns (physical names)
- Business terms/aliases
- Relationships (join recommendations)
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ColumnDefinition(BaseModel):
    table_fqn: str = Field(
        description="Fully-qualified table name: DATABASE.SCHEMA.TABLE (or SCHEMA.TABLE if no database)"
    )
    column_name: str = Field(description="Physical column name as used in SQL")
    data_type: Optional[str] = Field(default=None, description="Column data type")
    description: Optional[str] = Field(default=None, description="Business/technical description")
    business_terms: List[str] = Field(default_factory=list, description="Aliases/business terms for this column")
    is_primary_key: Optional[bool] = Field(default=None)
    is_foreign_key: Optional[bool] = Field(default=None)
    referenced_table_fqn: Optional[str] = Field(default=None, description="If FK, referenced table FQN")
    referenced_column_name: Optional[str] = Field(default=None, description="If FK, referenced column name")
    semantic_type: Optional[str] = Field(default=None, description="Semantic type: Metric, Identifier, Time, Attribute")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Arbitrary extra attributes from glossary")


class TableDefinition(BaseModel):
    table_fqn: str = Field(description="Fully-qualified table name: DATABASE.SCHEMA.TABLE (or SCHEMA.TABLE)")
    database: Optional[str] = Field(default=None)
    schema: Optional[str] = Field(default=None)
    table: str = Field(description="Physical table name")
    role: Optional[str] = Field(default=None, description="Table role: FACT or DIM")
    description: Optional[str] = Field(default=None)
    business_terms: List[str] = Field(default_factory=list, description="Aliases/business terms for this table")
    primary_keys: List[str] = Field(default_factory=list)
    columns: List[ColumnDefinition] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict, description="Arbitrary extra attributes from glossary")


class RelationshipDefinition(BaseModel):
    """Represents a recommended join between two tables/columns.

    Examples:
        left_fqn = SALES_DB.RPT.CUSTOMER, left_column = CUSTOMER_ID
        right_fqn = SALES_DB.FACT.INVOICE, right_column = CUSTOMER_ID
        cardinality = "1:N"
        predicate = "CUSTOMER.CUSTOMER_ID = INVOICE.CUSTOMER_ID"
    """

    left_table_fqn: str
    left_column: str
    right_table_fqn: str
    right_column: str
    cardinality: Optional[str] = Field(default=None, description="e.g., 1:1, 1:N, N:1, N:N")
    description: Optional[str] = Field(default=None)
    predicate: Optional[str] = Field(default=None, description="Suggested ON clause text")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Arbitrary extra attributes from glossary")


class BusinessTerm(BaseModel):
    """Maps business-friendly terms to physical objects."""

    term: str
    applies_to: str = Field(description="'table' or 'column'")
    table_fqn: Optional[str] = None
    column_name: Optional[str] = None
    description: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict, description="Arbitrary extra attributes from glossary")


class DatabaseSchemaModel(BaseModel):
    """Top-level container of ingested knowledge base."""

    tables: Dict[str, TableDefinition] = Field(
        default_factory=dict, description="Keyed by table FQN"
    )
    columns: Dict[str, ColumnDefinition] = Field(
        default_factory=dict, description="Keyed by table_fqn.column_name"
    )
    relationships: List[RelationshipDefinition] = Field(default_factory=list)
    business_terms: List[BusinessTerm] = Field(default_factory=list)

    def get_table(self, table_fqn: str) -> Optional[TableDefinition]:
        return self.tables.get(table_fqn)

    def get_column(self, table_fqn: str, column_name: str) -> Optional[ColumnDefinition]:
        key = f"{table_fqn}.{column_name}"
        return self.columns.get(key)

    def list_tables(self) -> List[str]:
        return sorted(self.tables.keys())

    def list_columns(self, table_fqn: str) -> List[str]:
        tbl = self.tables.get(table_fqn)
        if not tbl:
            return []
        return [c.column_name for c in tbl.columns]


