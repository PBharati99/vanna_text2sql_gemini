"""
Prompt builder that injects schema, relationships, and business terms from the
Excel-backed SchemaProvider into a concise system prompt for Gemini.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .provider import ExcelSchemaProvider


class ExcelSchemaPromptBuilder:
    """Builds a schema-aware system prompt summarizing key tables/columns/joins.

    Options allow trimming to keep the prompt compact while remaining helpful.
    """

    def __init__(
        self,
        provider: ExcelSchemaProvider,
        max_tables: int = 25,
        max_columns_per_table: int = 20,
        include_relationships: bool = True,
        include_business_terms: bool = True,
    ) -> None:
        self.provider = provider
        self.max_tables = max_tables
        self.max_columns_per_table = max_columns_per_table
        self.include_relationships = include_relationships
        self.include_business_terms = include_business_terms

    def build_system_prompt(self, extra_instructions: Optional[str] = None, allow_sql_execution: bool = False) -> str:
        lines: List[str] = []
        if allow_sql_execution:
            lines.append(
                "You are a SQL assistant. Your role is to discover the relevant schema and then generate and execute SQL queries to answer the user's question."
            )
        else:
            lines.append(
                "You are a schema discovery assistant. Your role is to find and document the relevant tables, columns, and relationships needed to answer the user's question."
            )
        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.append("- This is a READ-ONLY system. Database modifications are strictly prohibited.")
        if allow_sql_execution:
            lines.append("- ONLY generate SELECT queries. NEVER generate UPDATE, INSERT, DELETE, DROP, ALTER, TRUNCATE, or any data manipulation queries.")
            lines.append("- After discovering the schema, you MUST generate and execute a SQL query using the run_sql tool to answer the user's question.")
        else:
            lines.append("- DO NOT generate SQL queries in this step. Focus on discovering and documenting the schema.")
            lines.append("- After finding relevant schema elements, output a structured summary that will be used in the next step for SQL generation.")
        lines.append("")
        lines.append("SCHEMA DISCOVERY PROCESS (MANDATORY):")
        lines.append("1. For EVERY user question, ALWAYS start by calling search_schema with relevant keywords from the question.")
        lines.append("   - Example: If user asks about 'SKUs during Christmas', search for: 'SKU', 'Christmas', 'promotion', 'holiday', etc.")
        lines.append("   - DO NOT rely on the schema overview below - it's just a sample of commonly-used tables!")
        lines.append("2. Review ALL search results to see which tables/columns match the question.")
        lines.append("3. If column names might exist in multiple tables, search_schema will show ALL matches - review all options before choosing.")
        lines.append("4. Use table_info to see all columns in a specific table when needed.")
        lines.append("5. Use column_info to get detailed information about specific columns (data types, descriptions, relationships).")
        lines.append("6. Use relationships tool to find join paths between relevant tables.")
        lines.append("7. Use resolve_term if the user mentions business terms or synonyms.")
        
        if allow_sql_execution:
            lines.append("")
            lines.append("SQL GENERATION AND EXECUTION (MANDATORY AFTER SCHEMA DISCOVERY):")
            lines.append("8. After discovering the schema, IMMEDIATELY generate a SQL SELECT query to answer the user's question.")
            lines.append("9. Use the run_sql tool to execute the generated SQL query.")
            lines.append("10. Present the query results to the user with a clear explanation.")
            lines.append("")
            lines.append("DATE/WEEK CONVERSION RULES (CRITICAL - ALWAYS APPLY):")
            lines.append("- week_nbr column format: YYYYWW (e.g., 202521 = Year 2025, Week 21)")
            lines.append("- WHEN SELECTING week_nbr, ALWAYS include human-readable week dates in your query:")
            lines.append("  SELECT week_nbr,")
            lines.append("         TO_DATE(week_nbr::VARCHAR || '1', 'IYYYIW DAY') as week_start_date,")
            lines.append("         DATEADD('day', 6, TO_DATE(week_nbr::VARCHAR || '1', 'IYYYIW DAY')) as week_end_date")
            lines.append("- WHEN PRESENTING RESULTS with week_nbr, ALWAYS explain in natural language:")
            lines.append("  'Week 202521 corresponds to Week 21 of 2025 (May 19-25, 2025)'")
            lines.append("- When user asks about specific date ranges, convert to week_nbr format for filtering")
            lines.append("- Example week_nbr conversions:")
            lines.append("  * 202521 → Week 21 of 2025 (May 19-25, 2025)")
            lines.append("  * 202452 → Week 52 of 2024 (December 23-29, 2024)")
            lines.append("")
            lines.append("SNOWFLAKE SQL SYNTAX REQUIREMENTS:")
            lines.append("- Generate ONLY valid Snowflake SQL - no comments, no markdown, no explanations in the SQL string")
            lines.append("- Use fully-qualified table names: DATABASE.SCHEMA.TABLE (e.g., APP_PROMOFCST.PROMO_FORECAST.NML_AD_STR_SKU_HIST)")
            lines.append("- Column names should be used as-is from the schema (case-sensitive)")
            lines.append("- Use proper JOIN syntax: INNER JOIN, LEFT JOIN, etc. with ON clauses")
            lines.append("- Do NOT include SQL comments (-- or /* */) in the query string")
            lines.append("- Do NOT wrap SQL in markdown code blocks (```sql ... ```)")
            lines.append("- The SQL string passed to run_sql must be pure, executable SQL only")
            lines.append("")
            lines.append("IMPORTANT: Do NOT stop after schema discovery. You MUST generate and execute SQL to answer the question!")
        else:
            lines.append("")
            lines.append("OUTPUT FORMAT (REQUIRED):")
            lines.append("After discovering the schema, provide a structured summary in the following format:")
            lines.append("")
            lines.append("## Schema Discovery Summary")
            lines.append("")
            lines.append("### Selected Tables:")
            lines.append("- [Table FQN]: [Description]")
            lines.append("  - Reasoning: [Why this table is relevant to the user's question]")
            lines.append("")
            lines.append("### Selected Columns:")
            lines.append("- [Table FQN].[Column Name] ([Data Type]): [Description]")
            lines.append("  - Reasoning: [Why this column is needed for the query]")
            lines.append("")
            lines.append("### Relationships/Joins:")
            lines.append("- [Table1].[Column1] = [Table2].[Column2] ([Cardinality]): [Description]")
            lines.append("  - Reasoning: [Why this join is needed]")
            lines.append("")
            lines.append("### Business Terms Mapped:")
            lines.append("- [User's term] -> [Table].[Column]")
            lines.append("")
            lines.append("### Additional Notes:")
            lines.append("[Any important considerations, filters, aggregations, or constraints that should be considered]")

        if extra_instructions:
            lines.append("")
            lines.append("Additional instructions:")
            lines.append(extra_instructions)

        # Tables and columns - use relevance-based ordering
        table_names = self.provider.list_tables_by_relevance(limit=self.max_tables)
        if table_names:
            lines.append("")
            lines.append("=" * 60)
            lines.append("SCHEMA OVERVIEW (SAMPLE - MOST RELEVANT/CONNECTED TABLES):")
            lines.append("This shows the most connected/important tables (by relationships and size).")
            lines.append("For the user's specific question, ALWAYS use search_schema to find the exact tables needed!")
            lines.append("=" * 60)
            for tname in table_names:
                t = self.provider.get_table(tname)
                if not t:
                    continue
                desc = t.description or ""
                lines.append(f"- {t.table_fqn}: {desc}")
                # columns
                if t.columns:
                    col_lines: List[str] = []
                    for c in t.columns[: self.max_columns_per_table]:
                        col_desc = c.description or ""
                        col_dtype = f" ({c.data_type})" if c.data_type else ""
                        col_lines.append(f"    · {c.column_name}{col_dtype}: {col_desc}")
                    if len(t.columns) > self.max_columns_per_table:
                        col_lines.append(
                            f"    · ... ({len(t.columns) - self.max_columns_per_table} more columns)"
                        )
                    lines.extend(col_lines)

        # Relationships
        if self.include_relationships:
            lines.append("")
            lines.append("Known relationships (suggested joins):")
            rels_added = 0
            for tname in table_names:
                for r in self.provider.find_relationships(tname):
                    lines.append(
                        f"- {r.left_table_fqn}.{r.left_column} = {r.right_table_fqn}.{r.right_column}"
                        + (f" [{r.cardinality}]" if r.cardinality else "")
                        + (f" :: {r.predicate}" if r.predicate else "")
                    )
                    rels_added += 1
                    if rels_added >= 100:
                        lines.append("- ... (more relationships omitted)")
                        break
                if rels_added >= 100:
                    break

        # Business terms (selected)
        if self.include_business_terms and self.provider.schema.business_terms:
            lines.append("")
            lines.append("Business terms (selected):")
            for bt in self.provider.schema.business_terms[:100]:
                loc = (
                    f" ({bt.applies_to} -> {bt.table_fqn or ''}.{bt.column_name or ''})".rstrip(".")
                    if bt.applies_to
                    else ""
                )
                syn = f" [synonyms: {', '.join(bt.synonyms)}]" if bt.synonyms else ""
                lines.append(f"- {bt.term}{loc}{syn}: {bt.description or ''}")
            if len(self.provider.schema.business_terms) > 100:
                lines.append("- ... (more business terms omitted)")

        return "\n".join(lines)


