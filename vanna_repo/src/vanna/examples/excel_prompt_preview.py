"""
Build and print the schema-aware system prompt from the Excel knowledge base.

Usage:
  PYTHONPATH=src python vanna/examples/excel_prompt_preview.py \
    "./Promo Semantic Data_20250519.xlsx" --max-tables 15 --max-cols 15
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from vanna.schema.provider import ExcelSchemaProvider
    from vanna.schema.prompt_builder import ExcelSchemaPromptBuilder
except Exception:
    print("[error] Ensure PYTHONPATH includes 'src'.")
    raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the schema-aware system prompt")
    parser.add_argument("excel_path", help="Path to Excel knowledge base")
    parser.add_argument("--max-tables", type=int, default=25)
    parser.add_argument("--max-cols", type=int, default=20)
    parser.add_argument("--no-relationships", action="store_true")
    parser.add_argument("--no-terms", action="store_true")
    parser.add_argument("--extra", default=None, help="Extra instructions to include")
    args = parser.parse_args()

    if not os.path.exists(args.excel_path):
        print(f"[error] Excel file not found at {args.excel_path}")
        sys.exit(1)

    sp = ExcelSchemaProvider(args.excel_path)
    pb = ExcelSchemaPromptBuilder(
        provider=sp,
        max_tables=args.max_tables,
        max_columns_per_table=args.max_cols,
        include_relationships=not args.no_relationships,
        include_business_terms=not args.no_terms,
    )
    prompt = pb.build_system_prompt(extra_instructions=args.extra)
    print("\n=== SYSTEM PROMPT ===\n")
    print(prompt)


if __name__ == "__main__":
    main()


