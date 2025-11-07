"""
SchemaProvider backed by the Excel-ingested DatabaseSchemaModel.

Provides lookup/search utilities for:
- Tables and columns
- Relationships
- Business term resolution (map terms/synonyms to physical columns)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .excel_loader import ExcelSchemaLoader
from .models import (
    BusinessTerm,
    ColumnDefinition,
    DatabaseSchemaModel,
    RelationshipDefinition,
    TableDefinition,
)


class ExcelSchemaProvider:
    """Loads an Excel workbook and provides schema lookup APIs."""

    def __init__(self, excel_path: str) -> None:
        self.excel_path = excel_path
        self.schema: DatabaseSchemaModel = ExcelSchemaLoader(excel_path).load()

        # Build quick indices
        self._term_to_columns: Dict[str, List[Tuple[str, str]]] = {}
        self._index_terms()

    def _index_terms(self) -> None:
        # Map normalized term -> list of (table_fqn, column_name)
        for table_fqn, tdef in self.schema.tables.items():
            for c in tdef.columns:
                terms = [c.column_name] + c.business_terms
                for term in terms:
                    key = term.strip().lower()
                    if not key:
                        continue
                    self._term_to_columns.setdefault(key, []).append((table_fqn, c.column_name))
        for bt in self.schema.business_terms:
            key = bt.term.strip().lower()
            if key:
                # If scoped to a column, attach to that; else keep as general
                if bt.table_fqn and bt.column_name:
                    self._term_to_columns.setdefault(key, []).append((bt.table_fqn, bt.column_name))
            for s in bt.synonyms:
                k = s.strip().lower()
                if not k:
                    continue
                if bt.table_fqn and bt.column_name:
                    self._term_to_columns.setdefault(k, []).append((bt.table_fqn, bt.column_name))

    # --- Table APIs ---
    def list_tables(self) -> List[str]:
        return self.schema.list_tables()
    
    def list_tables_by_relevance(self, limit: int = 25) -> List[str]:
        """List tables ordered by relevance (most connected/important first).
        
        Relevance is determined by:
        1. Number of relationships (tables with more joins are more important)
        2. Number of columns (larger tables are often more important)
        3. Whether table is referenced by other tables (FK targets)
        
        Args:
            limit: Maximum number of tables to return
            
        Returns:
            List of table FQNs ordered by relevance
        """
        # Count relationships per table
        rel_counts: Dict[str, int] = {}
        referenced_tables: set[str] = set()
        
        for rel in self.schema.relationships:
            rel_counts[rel.left_table_fqn] = rel_counts.get(rel.left_table_fqn, 0) + 1
            rel_counts[rel.right_table_fqn] = rel_counts.get(rel.right_table_fqn, 0) + 1
            # Track which tables are referenced (FK targets)
            referenced_tables.add(rel.right_table_fqn)
        
        # Score tables: relationships count + column count + referenced bonus
        scored: List[Tuple[int, str]] = []
        for table_fqn, table_def in self.schema.tables.items():
            score = 0
            # Relationship count (weighted heavily)
            score += rel_counts.get(table_fqn, 0) * 10
            # Column count
            score += len(table_def.columns)
            # Bonus if referenced by other tables (important hub tables)
            if table_fqn in referenced_tables:
                score += 50
            
            scored.append((score, table_fqn))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [table_fqn for _, table_fqn in scored[:limit]]

    def get_table(self, table_fqn: str) -> Optional[TableDefinition]:
        return self.schema.get_table(table_fqn)

    def search_tables(self, text: str, limit: int = 20) -> List[TableDefinition]:
        """Search tables with word-based matching and relevance scoring."""
        q = text.strip().lower()
        if not q:
            return []
        
        # Split query into words (remove common stop words)
        query_words = [w for w in q.split() if len(w) > 2]  # Ignore very short words
        if not query_words:
            query_words = [q]  # Fallback to whole query if all words too short
        
        scored_results: List[Tuple[int, TableDefinition]] = []
        
        for t in self.schema.tables.values():
            hay = " ".join([
                t.table_fqn,
                t.table,
                " ".join(t.business_terms or []),
                t.description or "",
                " ".join([f"{k}:{v}" for k, v in t.metadata.items()]),
            ]).lower()
            
            # Score: count how many query words appear
            score = 0
            for word in query_words:
                if word in hay:
                    score += 1
                    # Bonus for exact phrase match
                    if q in hay:
                        score += 2
                    # Bonus for word in table name or FQN
                    if word in t.table.lower() or word in t.table_fqn.lower():
                        score += 1
            
            if score > 0:
                scored_results.append((score, t))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_results[:limit]]

    # --- Column APIs ---
    def list_columns(self, table_fqn: str) -> List[str]:
        return self.schema.list_columns(table_fqn)

    def get_column(self, table_fqn: str, column_name: str) -> Optional[ColumnDefinition]:
        return self.schema.get_column(table_fqn, column_name)

    def search_columns(self, text: str, limit: int = 50) -> List[Tuple[str, ColumnDefinition]]:
        """Search columns with word-based matching and relevance scoring."""
        q = text.strip().lower()
        if not q:
            return []
        
        # Split query into words (remove common stop words)
        query_words = [w for w in q.split() if len(w) > 2]  # Ignore very short words
        if not query_words:
            query_words = [q]  # Fallback to whole query if all words too short
        
        scored_results: List[Tuple[int, str, ColumnDefinition]] = []
        
        for t in self.schema.tables.values():
            for c in t.columns:
                hay = " ".join([
                    t.table_fqn,
                    c.column_name,
                    c.data_type or "",
                    " ".join(c.business_terms or []),
                    c.description or "",
                    " ".join([f"{k}:{v}" for k, v in c.metadata.items()]),
                ]).lower()
                
                # Score: count how many query words appear
                score = 0
                for word in query_words:
                    if word in hay:
                        score += 1
                        # Bonus for exact phrase match
                        if q in hay:
                            score += 2
                        # Bonus for word in column name
                        if word in c.column_name.lower():
                            score += 2
                        # Bonus for word in business terms
                        if any(word in bt.lower() for bt in c.business_terms):
                            score += 1
                
                if score > 0:
                    scored_results.append((score, t.table_fqn, c))
        
        # Sort by score (descending) and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [(table_fqn, c) for _, table_fqn, c in scored_results[:limit]]

    # --- Relationship APIs ---
    def find_relationships(self, table_fqn: str) -> List[RelationshipDefinition]:
        return [r for r in self.schema.relationships if r.left_table_fqn == table_fqn or r.right_table_fqn == table_fqn]

    def find_relationship_between(self, left_table_fqn: str, right_table_fqn: str) -> List[RelationshipDefinition]:
        return [
            r
            for r in self.schema.relationships
            if (r.left_table_fqn == left_table_fqn and r.right_table_fqn == right_table_fqn)
            or (r.left_table_fqn == right_table_fqn and r.right_table_fqn == left_table_fqn)
        ]

    # --- Term resolution ---
    def resolve_term(self, term: str, limit: int = 20) -> List[Tuple[str, str]]:
        """Resolve a business term or synonym to candidate (table_fqn, column) pairs."""
        return self._term_to_columns.get(term.strip().lower(), [])[:limit]


