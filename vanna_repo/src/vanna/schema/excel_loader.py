"""
Excel-backed schema loader for business glossary ingestion.

Expected workbook content (sheet names are flexible; matched by keywords):
- Schema/Tab/Table glossary: tables with descriptions and physical names
- Column glossary: columns with data types and descriptions
- Business relationships: recommended joins and cardinalities

Dependencies:
    pip install pandas openpyxl
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import (
    BusinessTerm,
    ColumnDefinition,
    DatabaseSchemaModel,
    RelationshipDefinition,
    TableDefinition,
)


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _to_fqn(database: Optional[str], schema: Optional[str], table: str) -> str:
    table = _norm(table)
    schema = _norm(schema)
    database = _norm(database)
    if database and schema:
        return f"{database}.{schema}.{table}"
    if schema:
        return f"{schema}.{table}"
    return table


def _merge_text(existing: Optional[str], incoming: Optional[str]) -> Optional[str]:
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming in existing:
        return existing
    return f"{existing} | {incoming}"


def _collect_metadata(row: pd.Series, df_columns: Dict[str, str], used_keys: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    used_set = {k.lower() for k in used_keys}
    for low_name, original in df_columns.items():
        if low_name in used_set:
            continue
        val = row.get(original)
        sval = _norm(str(val)) if val is not None else ""
        if sval:
            meta[low_name] = sval
    return meta


class ExcelSchemaLoader:
    """Loads a business glossary workbook into DatabaseSchemaModel."""

    def __init__(self, excel_path: str) -> None:
        self.excel_path = excel_path
        self._sheets: Dict[str, pd.DataFrame] = {}

    def load(self) -> DatabaseSchemaModel:
        # Read workbook (all sheets)
        xls = pd.read_excel(self.excel_path, sheet_name=None)
        # Normalize sheet keys
        self._sheets = {k.strip(): v for k, v in xls.items()}

        schema = DatabaseSchemaModel()

        # Locate sheets by heuristic
        table_sheet = self._pick_sheet(["table", "tab", "schema"], min_columns=2)
        column_sheet = self._pick_sheet(["column"], min_columns=2)
        rel_sheet = self._pick_sheet(["relationship", "join"], min_columns=2)
        term_sheet = self._pick_sheet(["glossary", "term", "dictionary"], min_columns=1, exclude=[table_sheet, column_sheet, rel_sheet])

        if table_sheet is not None:
            self._ingest_tables(schema, self._sheets[table_sheet])
        if column_sheet is not None:
            self._ingest_columns(schema, self._sheets[column_sheet])
        if rel_sheet is not None:
            self._ingest_relationships(schema, self._sheets[rel_sheet])
        if term_sheet is not None:
            self._ingest_terms(schema, self._sheets[term_sheet])

        return schema

    def _pick_sheet(self, keywords: List[str], min_columns: int = 1, exclude: List[Optional[str]] | None = None) -> Optional[str]:
        exclude = exclude or []
        for name, df in self._sheets.items():
            if name in exclude:
                continue
            key = name.lower()
            if any(k in key for k in keywords) and df.shape[1] >= min_columns:
                return name
        return None

    def _ingest_tables(self, schema: DatabaseSchemaModel, df: pd.DataFrame) -> None:
        cols = {c.lower().strip(): c for c in df.columns}
        # Heuristics for expected column names
        db_col = self._first_match(cols, ["database", "db"])
        schema_col = self._first_match(cols, ["schema", "schema_name"])
        table_col = self._first_match(cols, ["table", "table_name", "tab_name"])
        desc_col = self._first_match(cols, ["description", "table_description", "desc"])  # optional
        business_name_col = self._first_match(cols, ["business_name", "display_name", "alias"])  # optional

        for _, row in df.iterrows():
            database = _norm(row.get(db_col)) if db_col else None
            schema_name = _norm(row.get(schema_col)) if schema_col else None
            table_name = _norm(row.get(table_col)) if table_col else None
            if not table_name:
                continue
            table_fqn = _to_fqn(database, schema_name, table_name)

            tdef = schema.tables.get(table_fqn) or TableDefinition(
                table_fqn=table_fqn,
                database=database or None,
                schema=schema_name or None,
                table=table_name,
                description=None,
                business_terms=[],
                primary_keys=[],
                columns=[],
            )

            if desc_col:
                new_desc = _norm(row.get(desc_col))
                tdef.description = _merge_text(tdef.description, new_desc)
            if business_name_col:
                bn = _norm(row.get(business_name_col))
                if bn and bn not in tdef.business_terms:
                    tdef.business_terms.append(bn)

            # capture any additional fields into metadata
            used = [
                (db_col or ""),
                (schema_col or ""),
                (table_col or ""),
                (desc_col or ""),
                (business_name_col or ""),
            ]
            row_meta = _collect_metadata(row, {c.lower().strip(): c for c in df.columns}, used)
            for k, v in row_meta.items():
                tdef.metadata[k] = _merge_text(tdef.metadata.get(k), v) or v

            schema.tables[table_fqn] = tdef

    def _ingest_columns(self, schema: DatabaseSchemaModel, df: pd.DataFrame) -> None:
        cols = {c.lower().strip(): c for c in df.columns}
        db_col = self._first_match(cols, ["database", "db"])  # optional
        schema_col = self._first_match(cols, ["schema", "schema_name"])  # optional
        table_col = self._first_match(cols, ["table", "table_name", "tab_name"])
        column_col = self._first_match(cols, ["column", "column_name", "col_name"])
        dtype_col = self._first_match(cols, ["data_type", "datatype", "type"])  # optional
        desc_col = self._first_match(cols, ["description", "column_description", "desc"])  # optional
        business_name_col = self._first_match(cols, ["business_name", "display_name", "alias"])  # optional
        pk_flag_col = self._first_match(cols, ["is_pk", "primary_key", "is_primary_key"])  # optional
        fk_flag_col = self._first_match(cols, ["is_fk", "foreign_key", "is_foreign_key"])  # optional
        ref_table_col = self._first_match(cols, ["ref_table", "referenced_table", "fk_table"])  # optional
        ref_col_col = self._first_match(cols, ["ref_column", "referenced_column", "fk_column"])  # optional

        for _, row in df.iterrows():
            table_name = _norm(row.get(table_col)) if table_col else None
            column_name = _norm(row.get(column_col)) if column_col else None
            if not table_name or not column_name:
                continue

            database = _norm(row.get(db_col)) if db_col else None
            schema_name = _norm(row.get(schema_col)) if schema_col else None
            table_fqn = _to_fqn(database, schema_name, table_name)

            # ensure table exists
            tdef = schema.tables.get(table_fqn) or TableDefinition(
                table_fqn=table_fqn,
                database=database or None,
                schema=schema_name or None,
                table=table_name,
            )
            schema.tables[table_fqn] = tdef

            key = f"{table_fqn}.{column_name}"
            existing = schema.columns.get(key)

            # Build incoming values
            in_dtype = _norm(row.get(dtype_col)) if dtype_col else None
            in_desc = _norm(row.get(desc_col)) if desc_col else None
            in_bn = _norm(row.get(business_name_col)) if business_name_col else None
            in_is_pk = bool(row.get(pk_flag_col)) if pk_flag_col else None
            in_is_fk = bool(row.get(fk_flag_col)) if fk_flag_col else None
            in_ref_table = _norm(row.get(ref_table_col)) if ref_table_col else None
            in_ref_col = _norm(row.get(ref_col_col)) if ref_col_col else None

            if existing:
                # Merge descriptions (preserve all unique notes)
                existing.description = _merge_text(existing.description, in_desc)
                # Prefer first non-empty dtype; if different later, keep original (no override)
                if not existing.data_type and in_dtype:
                    existing.data_type = in_dtype
                # Merge business terms
                if in_bn and in_bn not in existing.business_terms:
                    existing.business_terms.append(in_bn)
                # Merge FK/PK hints conservatively
                if existing.is_primary_key is None and in_is_pk is not None:
                    existing.is_primary_key = in_is_pk
                if existing.is_foreign_key is None and in_is_fk is not None:
                    existing.is_foreign_key = in_is_fk
                if not existing.referenced_table_fqn and in_ref_table:
                    existing.referenced_table_fqn = in_ref_table
                if not existing.referenced_column_name and in_ref_col:
                    existing.referenced_column_name = in_ref_col
                # Merge metadata
                used = [
                    (db_col or ""),
                    (schema_col or ""),
                    (table_col or ""),
                    (column_col or ""),
                    (dtype_col or ""),
                    (desc_col or ""),
                    (business_name_col or ""),
                    (pk_flag_col or ""),
                    (fk_flag_col or ""),
                    (ref_table_col or ""),
                    (ref_col_col or ""),
                ]
                row_meta = _collect_metadata(row, {c.lower().strip(): c for c in df.columns}, used)
                for k, v in row_meta.items():
                    existing.metadata[k] = _merge_text(existing.metadata.get(k), v) or v
                # Ensure table.primary_keys reflects any pk flag
                if existing.is_primary_key and column_name not in tdef.primary_keys:
                    tdef.primary_keys.append(column_name)
                # Ensure table.columns list has unique column once
                if not any(c.column_name == column_name for c in tdef.columns):
                    tdef.columns.append(existing)
            else:
                cdef = ColumnDefinition(
                    table_fqn=table_fqn,
                    column_name=column_name,
                    data_type=in_dtype,
                    description=in_desc,
                    business_terms=[in_bn] if in_bn else [],
                    is_primary_key=in_is_pk,
                    is_foreign_key=in_is_fk,
                    referenced_table_fqn=in_ref_table,
                    referenced_column_name=in_ref_col,
                    metadata={},
                )
                # attach new
                schema.columns[key] = cdef
                if not any(c.column_name == column_name for c in tdef.columns):
                    tdef.columns.append(cdef)
                if cdef.is_primary_key and column_name not in tdef.primary_keys:
                    tdef.primary_keys.append(column_name)
                # capture metadata for new
                used = [
                    (db_col or ""),
                    (schema_col or ""),
                    (table_col or ""),
                    (column_col or ""),
                    (dtype_col or ""),
                    (desc_col or ""),
                    (business_name_col or ""),
                    (pk_flag_col or ""),
                    (fk_flag_col or ""),
                    (ref_table_col or ""),
                    (ref_col_col or ""),
                ]
                row_meta = _collect_metadata(row, {c.lower().strip(): c for c in df.columns}, used)
                for k, v in row_meta.items():
                    cdef.metadata[k] = v

    def _ingest_relationships(self, schema: DatabaseSchemaModel, df: pd.DataFrame) -> None:
        cols = {c.lower().strip(): c for c in df.columns}
        left_table_col = self._first_match(cols, ["left_table", "table_a", "from_table"])  # required
        left_col_col = self._first_match(cols, ["left_column", "column_a", "from_column"])  # required
        right_table_col = self._first_match(cols, ["right_table", "table_b", "to_table"])  # required
        right_col_col = self._first_match(cols, ["right_column", "column_b", "to_column"])  # required
        card_col = self._first_match(cols, ["cardinality", "card"])  # optional
        desc_col = self._first_match(cols, ["description", "desc"])  # optional
        pred_col = self._first_match(cols, ["predicate", "on_clause", "join_on"])  # optional

        for _, row in df.iterrows():
            lt = _norm(row.get(left_table_col)) if left_table_col else None
            lc = _norm(row.get(left_col_col)) if left_col_col else None
            rt = _norm(row.get(right_table_col)) if right_table_col else None
            rc = _norm(row.get(right_col_col)) if right_col_col else None
            if not (lt and lc and rt and rc):
                continue
            rel = RelationshipDefinition(
                left_table_fqn=lt,
                left_column=lc,
                right_table_fqn=rt,
                right_column=rc,
                cardinality=_norm(row.get(card_col)) if card_col else None,
                description=_norm(row.get(desc_col)) if desc_col else None,
                predicate=_norm(row.get(pred_col)) if pred_col else None,
                metadata={},
            )
            # Deduplicate exact duplicates
            duplicate = next(
                (
                    r
                    for r in schema.relationships
                    if r.left_table_fqn == rel.left_table_fqn
                    and r.left_column == rel.left_column
                    and r.right_table_fqn == rel.right_table_fqn
                    and r.right_column == rel.right_column
                    and (r.predicate or "") == (rel.predicate or "")
                ),
                None,
            )
            if not duplicate:
                # attach metadata
                used = [
                    (left_table_col or ""),
                    (left_col_col or ""),
                    (right_table_col or ""),
                    (right_col_col or ""),
                    (card_col or ""),
                    (desc_col or ""),
                    (pred_col or ""),
                ]
                row_meta = _collect_metadata(row, {c.lower().strip(): c for c in df.columns}, used)
                for k, v in row_meta.items():
                    rel.metadata[k] = v
                schema.relationships.append(rel)

    def _ingest_terms(self, schema: DatabaseSchemaModel, df: pd.DataFrame) -> None:
        cols = {c.lower().strip(): c for c in df.columns}
        term_col = self._first_match(cols, ["term", "business_term", "name"])  # required
        applies_to_col = self._first_match(cols, ["applies_to", "type"])  # optional
        table_col = self._first_match(cols, ["table", "table_fqn"])  # optional
        column_col = self._first_match(cols, ["column", "column_name"])  # optional
        desc_col = self._first_match(cols, ["description", "desc"])  # optional
        syn_col = self._first_match(cols, ["synonyms", "aliases"])  # optional

        for _, row in df.iterrows():
            term = _norm(row.get(term_col)) if term_col else None
            if not term:
                continue
            applies_to = _norm(row.get(applies_to_col)) if applies_to_col else ""
            table_fqn = _norm(row.get(table_col)) if table_col else None
            column_name = _norm(row.get(column_col)) if column_col else None
            desc = _norm(row.get(desc_col)) if desc_col else None
            synonyms: List[str] = []
            if syn_col:
                raw_syn = row.get(syn_col)
                if isinstance(raw_syn, str) and raw_syn.strip():
                    synonyms = [s.strip() for s in re.split(r"[,;]\s*", raw_syn) if s.strip()]

            # Merge/dedupe by identity of term scope
            existing = next(
                (
                    b
                    for b in schema.business_terms
                    if b.term == term
                    and (b.applies_to or "") == (applies_to or "")
                    and (b.table_fqn or "") == (table_fqn or "")
                    and (b.column_name or "") == (column_name or "")
                ),
                None,
            )
            if existing:
                existing.description = _merge_text(existing.description, desc)
                for s in synonyms:
                    if s not in existing.synonyms:
                        existing.synonyms.append(s)
                # merge metadata
                used = [
                    (term_col or ""),
                    (applies_to_col or ""),
                    (table_col or ""),
                    (column_col or ""),
                    (desc_col or ""),
                    (syn_col or ""),
                ]
                row_meta = _collect_metadata(row, {c.lower().strip(): c for c in df.columns}, used)
                for k, v in row_meta.items():
                    existing.metadata[k] = _merge_text(existing.metadata.get(k), v) or v
            else:
                bt = BusinessTerm(
                    term=term,
                    applies_to=applies_to or "",
                    table_fqn=table_fqn,
                    column_name=column_name,
                    description=desc,
                    synonyms=synonyms,
                    metadata={},
                )
                # attach metadata for new term
                used = [
                    (term_col or ""),
                    (applies_to_col or ""),
                    (table_col or ""),
                    (column_col or ""),
                    (desc_col or ""),
                    (syn_col or ""),
                ]
                row_meta = _collect_metadata(row, {c.lower().strip(): c for c in df.columns}, used)
                for k, v in row_meta.items():
                    bt.metadata[k] = v
                schema.business_terms.append(bt)

    @staticmethod
    def _first_match(col_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
        for key, original in col_map.items():
            if any(c in key for c in candidates):
                return original
        return None


