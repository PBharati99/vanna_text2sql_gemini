"""
Quick preview for ExcelSchemaProvider lookups.

Usage:
  PYTHONPATH=src python vanna/examples/excel_schema_provider_preview.py \
      "./Promo Semantic Data_20250519.xlsx" --query "customer" --limit 5
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from vanna.schema.provider import ExcelSchemaProvider
except Exception:
    print("[error] Cannot import ExcelSchemaProvider. Ensure PYTHONPATH includes 'src'.")
    raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview ExcelSchemaProvider lookups")
    parser.add_argument("excel_path", help="Path to Excel file")
    parser.add_argument("--query", default="", help="Search text for tables/columns")
    parser.add_argument("--term", default="", help="Resolve a business term")
    parser.add_argument("--table", default="", help="Print info for a specific table FQN")
    parser.add_argument("--limit", type=int, default=10, help="Result limit")
    args = parser.parse_args()

    if not os.path.exists(args.excel_path):
        print(f"[error] Excel file not found at {args.excel_path}")
        sys.exit(1)

    sp = ExcelSchemaProvider(args.excel_path)

    if args.table:
        t = sp.get_table(args.table)
        if not t:
            print(f"[info] Table not found: {args.table}")
        else:
            print(f"\n[{t.table_fqn}] :: {t.description or ''}")
            if t.primary_keys:
                print(f"  primary_keys: {', '.join(t.primary_keys)}")
            print("  columns:")
            for c in t.columns:
                print(f"    - {c.column_name} :: {c.data_type or ''} :: {c.description or ''}")

    if args.query:
        print(f"\nTable search for '{args.query}':")
        for t in sp.search_tables(args.query, limit=args.limit):
            print(f"- {t.table_fqn} :: {t.description or ''}")

        print(f"\nColumn search for '{args.query}':")
        for table_fqn, c in sp.search_columns(args.query, limit=args.limit):
            print(f"- {table_fqn}.{c.column_name} :: {c.data_type or ''} :: {c.description or ''}")

    if args.term:
        print(f"\nResolve term '{args.term}':")
        for table_fqn, col in sp.resolve_term(args.term, limit=args.limit):
            print(f"- {table_fqn}.{col}")


if __name__ == "__main__":
    main()


