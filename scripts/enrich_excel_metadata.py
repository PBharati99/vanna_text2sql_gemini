import pandas as pd
import os
import sys

def enrich_excel_metadata(input_path, output_path):
    print(f"Loading Excel file: {input_path}")
    
    try:
        xls = pd.read_excel(input_path, sheet_name=None)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Helper to find sheet by keywords (matching vanna's loader logic but stricter)
    def find_sheet(sheets, keywords, min_cols=1, required_col=None):
        # First pass: exact match for keywords
        for name, df in sheets.items():
            key = name.lower()
            # If required_col is specified, check if it exists
            if required_col:
                cols = [c.lower() for c in df.columns]
                if not any(required_col.lower() in c for c in cols):
                    continue
            
            if any(k in key for k in keywords) and df.shape[1] >= min_cols:
                return name
        return None

    # 1. Identify Sheets
    # Prioritize "Table Glossary" and ensure it has a "Table" column
    table_sheet_name = find_sheet(xls, ["table glossary", "table_glossary", "tables"], min_cols=2, required_col="Table")
    if not table_sheet_name:
        # Fallback to general search
        table_sheet_name = find_sheet(xls, ["table", "tab", "schema"], min_cols=2)
        
    column_sheet_name = find_sheet(xls, ["column"], min_cols=2)

    if not table_sheet_name:
        print("Error: Could not find Table sheet.")
        return
    if not column_sheet_name:
        print("Error: Could not find Column sheet.")
        return

    print(f"Found Table sheet: {table_sheet_name}")
    print(f"Found Column sheet: {column_sheet_name}")

    # 2. Enrich Tables
    df_tables = xls[table_sheet_name]
    print(f"Table Sheet Columns: {list(df_tables.columns)}")
    print(f"Table Sheet Rows: {len(df_tables)}")
    
    # Check if 'Table Role' exists, if not create it
    role_col = next((c for c in df_tables.columns if 'role' in c.lower()), 'Table Role')
    if role_col not in df_tables.columns:
        print("Adding 'Table Role' column...")
        df_tables[role_col] = None
    
    # Find the table name column
    table_name_col = next((c for c in df_tables.columns if any(k in c.lower() for k in ['table', 'tab_name'])), None)
    print(f"Identified Table Name Column: {table_name_col}")
    
    if table_name_col:
        def guess_role(name):
            if pd.isna(name): return "DIM"
            name = str(name).upper()
            # Heuristics for Fact tables
            fact_keywords = ['FACT', 'HIST', 'TRANS', 'SALES', 'EVENT', 'METRIC', 'MEASURE', 'FCST', 'FORECAST']
            if any(k in name for k in fact_keywords):
                return 'FACT'
            return 'DIM'

        # Apply guessing only where missing
        mask = df_tables[role_col].isna() | (df_tables[role_col] == '')
        # Make sure we only guess for rows that have a table name
        mask = mask & df_tables[table_name_col].notna()
        
        df_tables.loc[mask, role_col] = df_tables.loc[mask, table_name_col].apply(guess_role)
        print(f"Enriched {mask.sum()} table roles.")
        
        # Print a few examples
        print("Example enrichments:")
        print(df_tables[[table_name_col, role_col]].head(10))
    
    xls[table_sheet_name] = df_tables

    # 3. Enrich Columns
    df_columns = xls[column_sheet_name]
    
    # Check if 'Semantic Type' exists, if not create it
    sem_col = next((c for c in df_columns.columns if 'semantic' in c.lower() and 'type' in c.lower()), 'Semantic Type')
    if sem_col not in df_columns.columns:
        print("Adding 'Semantic Type' column...")
        df_columns[sem_col] = None

    # Find column name column
    col_name_col = next((c for c in df_columns.columns if any(k in c.lower() for k in ['column', 'col_name'])), None)
    print(f"Identified Column Name Column: {col_name_col}")
    
    if col_name_col:
        def guess_semantic_type(name):
            if pd.isna(name): return "Attribute"
            name = str(name).upper()
            
            # Heuristics
            if any(k in name for k in ['ID', 'KEY', 'CODE', 'NBR', 'SKU', 'NUM']):
                return 'Identifier'
            if any(k in name for k in ['AMT', 'PRICE', 'COST', 'QTY', 'SALES', 'REV', 'TOTAL', 'COUNT', 'PCT', 'RATE']):
                return 'Metric'
            if any(k in name for k in ['DATE', 'TIME', 'YEAR', 'MONTH', 'WEEK', 'DAY', 'DT', 'TS']):
                return 'Time'
            
            return 'Attribute'

        # Apply guessing only where missing
        mask = df_columns[sem_col].isna() | (df_columns[sem_col] == '')
        # Make sure we only guess for rows that have a column name
        mask = mask & df_columns[col_name_col].notna()
        
        df_columns.loc[mask, sem_col] = df_columns.loc[mask, col_name_col].apply(guess_semantic_type)
        print(f"Enriched {mask.sum()} column semantic types.")

    xls[column_sheet_name] = df_columns

    # 4. Save
    print(f"Saving enriched Excel to: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in xls.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("Done! Please verify the enriched file.")

if __name__ == "__main__":
    input_file = "Promo Semantic Data_20250519.xlsx"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = os.path.splitext(input_file)[0] + "_Enriched.xlsx"
    
    if os.path.exists(input_file):
        enrich_excel_metadata(input_file, output_file)
    else:
        print(f"File {input_file} not found. Please place it in the directory or provide path.")
