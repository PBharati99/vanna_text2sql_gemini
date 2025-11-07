"""
SQL validation utilities to prevent data manipulation queries.

This module provides validation to ensure only SELECT queries are allowed,
blocking UPDATE, INSERT, DELETE, DROP, ALTER, TRUNCATE, and other DDL/DML operations.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# SQL keywords that indicate data manipulation (case-insensitive)
FORBIDDEN_KEYWORDS = [
    "UPDATE",
    "INSERT",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK",
    "EXEC",
    "EXECUTE",
    "CALL",
]


def validate_sql_readonly(sql: str) -> Tuple[bool, Optional[str]]:
    """Validate that SQL query is read-only (SELECT only).
    
    Args:
        sql: SQL query string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if query is safe (SELECT only), False otherwise
        - error_message: None if valid, error description if invalid
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    
    # Normalize: remove comments and extra whitespace
    sql_normalized = _normalize_sql(sql)
    
    # Extract first significant keyword (skip WITH, comments, etc.)
    first_keyword = _extract_first_keyword(sql_normalized)
    
    if not first_keyword:
        return False, "Could not determine query type"
    
    first_keyword_upper = first_keyword.upper()
    
    # Allow SELECT and WITH (CTEs)
    if first_keyword_upper in ("SELECT", "WITH"):
        # Double-check: even WITH must lead to SELECT
        if first_keyword_upper == "WITH":
            # Check if there's a SELECT after the WITH clause
            if "SELECT" not in sql_normalized.upper():
                return False, "WITH clause must be followed by SELECT"
        return True, None
    
    # Block all manipulation keywords
    if first_keyword_upper in FORBIDDEN_KEYWORDS:
        return False, f"Query type '{first_keyword_upper}' is not allowed. Only SELECT queries are permitted."
    
    # Check for forbidden keywords anywhere in the query (to catch subqueries with manipulation)
    sql_upper = sql_normalized.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        # Use word boundaries to avoid false positives (e.g., "SELECTED" shouldn't match "SELECT")
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper):
            return False, f"Forbidden keyword '{keyword}' detected. Only SELECT queries are permitted."
    
    # If we get here and it's not SELECT/WITH, it's probably something we don't recognize
    # Be conservative and block it
    if first_keyword_upper not in ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"):
        return False, f"Unknown or unsupported query type '{first_keyword}'. Only SELECT queries are permitted."
    
    return True, None


def _normalize_sql(sql: str) -> str:
    """Normalize SQL by removing comments and extra whitespace."""
    # Remove single-line comments (-- ...)
    sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Normalize whitespace
    sql = ' '.join(sql.split())
    
    return sql.strip()


def _extract_first_keyword(sql: str) -> Optional[str]:
    """Extract the first SQL keyword from a query.
    
    Handles:
    - WITH ... SELECT (CTEs)
    - SELECT
    - Comments before keywords
    """
    if not sql:
        return None
    
    sql_upper = sql.upper().strip()
    
    # Skip leading whitespace
    sql_upper = sql_upper.lstrip()
    
    # Extract first word (keyword)
    match = re.match(r'^(\w+)', sql_upper)
    if match:
        return match.group(1)
    
    return None

