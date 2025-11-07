"""
Preview script to validate Excel knowledge base ingestion.

Usage:
    PYTHONPATH=src python vanna/examples/excel_schema_preview.py "./Promo Semantic Data_20250519.xlsx"

Options:
    --full            Print full detailed dump of tables, columns, relationships, and terms
    --json-out PATH   Save the entire ingested model to a JSON file

Outputs:
    - Counts of tables, columns, relationships, terms
    - Sample tables and columns (by default)
    - Full detailed dump when --full is provided
    - Optional JSON export when --json-out is provided
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

try:
    from vanna.schema.excel_loader import ExcelSchemaLoader
except Exception:
    print("[error] Cannot import ExcelSchemaLoader. Ensure PYTHONPATH includes 'src'.")
    raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Excel knowledge base ingestion")
    parser.add_argument("excel_path", help="Path to Excel file (e.g., Promo Semantic Data_20250519.xlsx)")
    parser.add_argument("--full", action="store_true", help="Print full detailed dump")
    parser.add_argument("--json-out", dest="json_out", default=None, help="Path to write JSON output")
    args = parser.parse_args()

    excel_path = args.excel_path
    if not os.path.exists(excel_path):
        print(f"[error] Excel file not found at {excel_path}")
        sys.exit(1)

    loader = ExcelSchemaLoader(excel_path)
    schema = loader.load()

    print("\n=== Ingestion Summary ===")
    print(f"Tables: {len(schema.tables)}")
    print(f"Columns: {len(schema.columns)}")
    print(f"Relationships: {len(schema.relationships)}")
    print(f"Business Terms: {len(schema.business_terms)}")

    if args.full:
        print("\n=== Tables ===")
        for t in schema.list_tables():
            tdef = schema.tables[t]
            print(f"\n[{tdef.table_fqn}] :: {tdef.description or ''}")
            if tdef.business_terms:
                print(f"  business_terms: {', '.join(tdef.business_terms)}")
            if tdef.metadata:
                print("  metadata:")
                for k, v in sorted(tdef.metadata.items()):
                    print(f"    - {k}: {v}")
            if tdef.primary_keys:
                print(f"  primary_keys: {', '.join(tdef.primary_keys)}")
            if tdef.columns:
                print("  columns:")
                for c in tdef.columns:
                    print(
                        f"    - {c.column_name} :: {c.data_type or ''} :: {c.description or ''}"
                    )
                    if c.business_terms:
                        print(f"      business_terms: {', '.join(c.business_terms)}")
                    if c.is_primary_key:
                        print("      is_primary_key: True")
                    if c.is_foreign_key:
                        print("      is_foreign_key: True")
                    if c.referenced_table_fqn or c.referenced_column_name:
                        print(
                            f"      references: {c.referenced_table_fqn or ''}.{c.referenced_column_name or ''}"
                        )
                    if c.metadata:
                        print("      metadata:")
                        for mk, mv in sorted(c.metadata.items()):
                            print(f"        - {mk}: {mv}")

        print("\n=== Relationships ===")
        for rel in schema.relationships:
            print(
                f"- {rel.left_table_fqn}.{rel.left_column} = {rel.right_table_fqn}.{rel.right_column} "
                f"[{rel.cardinality or ''}] :: {rel.predicate or ''}"
            )
            if rel.metadata:
                print("  metadata:")
                for rk, rv in sorted(rel.metadata.items()):
                    print(f"    - {rk}: {rv}")

        print("\n=== Business Terms ===")
        for bt in schema.business_terms:
            loc = (
                f" ({bt.applies_to} -> {bt.table_fqn or ''}.{bt.column_name or ''})".rstrip(".")
                if bt.applies_to
                else ""
            )
            syn = f" [synonyms: {', '.join(bt.synonyms)}]" if bt.synonyms else ""
            print(f"- {bt.term}{loc}{syn} :: {bt.description or ''}")
            if bt.metadata:
                print("  metadata:")
                for tk, tv in sorted(bt.metadata.items()):
                    print(f"    - {tk}: {tv}")
    else:
        # Show sample tables
        sample_tables: List[str] = schema.list_tables()[:5]
        print("\nSample tables (up to 5):")
        for t in sample_tables:
            tdef = schema.tables[t]
            print(f"- {t} :: {tdef.description or ''}")

        # Show sample columns
        print("\nSample columns (first 10):")
        i = 0
        for key, cdef in schema.columns.items():
            print(f"- {key} :: {cdef.data_type or ''} :: {cdef.description or ''}")
            i += 1
            if i >= 10:
                break

        # Show sample relationships
        print("\nSample relationships (first 5):")
        for rel in schema.relationships[:5]:
            print(
                f"- {rel.left_table_fqn}.{rel.left_column} = {rel.right_table_fqn}.{rel.right_column} "
                f"[{rel.cardinality or ''}] :: {rel.predicate or ''}"
            )

        # Show sample business terms
        print("\nSample business terms (first 10):")
        for bt in schema.business_terms[:10]:
            loc = (
                f" ({bt.applies_to} -> {bt.table_fqn or ''}.{bt.column_name or ''})".rstrip(".")
                if bt.applies_to
                else ""
            )
            syn = f" [synonyms: {', '.join(bt.synonyms)}]" if bt.synonyms else ""
            print(f"- {bt.term}{loc}{syn} :: {bt.description or ''}")

    if args.json_out:
        # Pydantic v2: model_dump; ensure plain JSON compatible
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(schema.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON to: {args.json_out}")


if __name__ == "__main__":
    main()


