"""
SchemaProvider backed by the Excel-ingested DatabaseSchemaModel.

Provides lookup/search utilities for:
- Tables and columns
- Relationships
- Business term resolution (map terms/synonyms to physical columns)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import numpy as np
import logging

from .excel_loader import ExcelSchemaLoader
from .models import (
    BusinessTerm,
    ColumnDefinition,
    DatabaseSchemaModel,
    RelationshipDefinition,
    TableDefinition,
)

logger = logging.getLogger(__name__)

class ExcelSchemaProvider:
    """Loads an Excel workbook and provides schema lookup APIs."""

    def __init__(self, excel_path: str, api_key: Optional[str] = None) -> None:
        self.excel_path = excel_path
        self.schema: DatabaseSchemaModel = ExcelSchemaLoader(excel_path).load()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # Build quick indices
        self._term_to_columns: Dict[str, List[Tuple[str, str]]] = {}
        self._index_terms()
        
        # Vector embeddings storage
        self.embeddings: List[Dict[str, Any]] = []
        self.embedding_cache_path = f"{excel_path}.embeddings.json"
        
        if self.api_key:
            self._init_embeddings()

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

    def _init_embeddings(self) -> None:
        """Initialize vector embeddings for hybrid search."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            logger.warning("google-generativeai not installed. Vector search disabled.")
            return

        if os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'r') as f:
                    self.embeddings = json.load(f)
                logger.info(f"Loaded {len(self.embeddings)} embeddings from cache.")
                return
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")

        logger.info("Generating embeddings for schema...")
        self._generate_embeddings()

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all tables and columns."""
        if not hasattr(self, 'genai'):
            return

        # 1. Embed Tables
        for table in self.schema.tables.values():
            text = f"Table: {table.table_fqn}\n"
            if table.role:
                text += f"Role: {table.role}\n"
            if table.description:
                text += f"Description: {table.description}\n"
            
            # Add top columns keywords to table embedding context
            col_names = [c.column_name for c in table.columns[:10]]
            text += f"Columns: {', '.join(col_names)}"

            vec = self._get_embedding(text)
            if vec:
                self.embeddings.append({
                    "type": "table",
                    "key": table.table_fqn,
                    "text": text,
                    "embedding": vec,
                    "metadata": {"table_fqn": table.table_fqn}
                })

        # 2. Embed Columns (only those with descriptions or specific semantic types)
        for table in self.schema.tables.values():
            for col in table.columns:
                # Skip boring columns unless they have descriptions
                if not col.description and not col.semantic_type:
                    continue
                
                text = f"Column: {col.column_name} (Table: {table.table_fqn})\n"
                if col.semantic_type:
                    text += f"Type: {col.semantic_type}\n"
                if col.description:
                    text += f"Description: {col.description}"
                
                vec = self._get_embedding(text)
                if vec:
                    self.embeddings.append({
                        "type": "column",
                        "key": f"{table.table_fqn}.{col.column_name}",
                        "text": text,
                        "embedding": vec,
                        "metadata": {"table_fqn": table.table_fqn, "column_name": col.column_name}
                    })

        # Save cache
        try:
            with open(self.embedding_cache_path, 'w') as f:
                json.dump(self.embeddings, f)
            logger.info(f"Saved {len(self.embeddings)} embeddings to cache.")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using Gemini."""
        try:
            # Use embedding-001 model
            result = self.genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title="Schema Embedding"
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _search_vectors(self, query: str, item_type: str = None, limit: int = 10) -> List[Tuple[float, Dict[str, Any]]]:
        """Search embeddings by cosine similarity. Returns list of (score, item)."""
        if not self.embeddings or not hasattr(self, 'genai'):
            return []

        try:
            # Embed query
            q_vec = self.genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )['embedding']
            
            scores = []
            for item in self.embeddings:
                if item_type and item['type'] != item_type:
                    continue
                
                # Cosine similarity
                vec = item['embedding']
                dot = np.dot(q_vec, vec)
                norm_q = np.linalg.norm(q_vec)
                norm_v = np.linalg.norm(vec)
                sim = dot / (norm_q * norm_v) if norm_q * norm_v > 0 else 0
                
                scores.append((float(sim), item))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            return scores[:limit]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

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
        """Search tables with hybrid search (keyword + vector)."""
        q = text.strip().lower()
        if not q:
            return []
        
        scores: Dict[str, float] = {}

        # --- 1. Keyword Search ---
        query_words = [w for w in q.split() if len(w) > 2]
        if not query_words:
            query_words = [q]
        
        for t in self.schema.tables.values():
            hay = " ".join([
                t.table_fqn,
                t.table,
                " ".join(t.business_terms or []),
                t.description or "",
                " ".join([f"{k}:{v}" for k, v in t.metadata.items()]),
            ]).lower()
            
            kw_score = 0.0
            for word in query_words:
                if word in hay:
                    kw_score += 1.0
                    if q in hay: # Exact phrase match
                        kw_score += 2.0
                    if word in t.table.lower() or word in t.table_fqn.lower():
                        kw_score += 1.0
            
            if kw_score > 0:
                scores[t.table_fqn] = kw_score

        # --- 2. Vector Search ---
        # Search for more candidates (limit*2 to get good coverage)
        vector_hits = self._search_vectors(text, item_type="table", limit=limit*2)
        
        for score, item in vector_hits:
            table_fqn = item['metadata']['table_fqn']
            # Weight vector score (0-1) to be comparable to keyword counts
            # Boost vector score significantly to surface semantic matches
            weighted_score = score * 5.0
            scores[table_fqn] = scores.get(table_fqn, 0.0) + weighted_score

        # --- 3. Sort & Return ---
        sorted_fqns = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        results = []
        for fqn in sorted_fqns[:limit]:
            t = self.get_table(fqn)
            if t:
                results.append(t)
                
        return results


    # --- Column APIs ---
    def list_columns(self, table_fqn: str) -> List[str]:
        return self.schema.list_columns(table_fqn)

    def get_column(self, table_fqn: str, column_name: str) -> Optional[ColumnDefinition]:
        return self.schema.get_column(table_fqn, column_name)

    def search_columns(self, text: str, limit: int = 50) -> List[Tuple[str, ColumnDefinition]]:
        """Search columns with hybrid search (keyword + vector)."""
        q = text.strip().lower()
        if not q:
            return []
        
        scores: Dict[Tuple[str, str], float] = {}

        # --- 1. Keyword Search ---
        query_words = [w for w in q.split() if len(w) > 2]
        if not query_words:
            query_words = [q]
        
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
                
                kw_score = 0.0
                for word in query_words:
                    if word in hay:
                        kw_score += 1.0
                        if q in hay:
                            kw_score += 2.0
                        if word in c.column_name.lower():
                            kw_score += 2.0
                        if any(word in bt.lower() for bt in c.business_terms):
                            kw_score += 1.0
                
                if kw_score > 0:
                    scores[(t.table_fqn, c.column_name)] = kw_score

        # --- 2. Vector Search ---
        vector_hits = self._search_vectors(text, item_type="column", limit=limit*2)
        
        for score, item in vector_hits:
            key = (item['metadata']['table_fqn'], item['metadata']['column_name'])
            weighted_score = score * 5.0
            scores[key] = scores.get(key, 0.0) + weighted_score

        # --- 3. Sort & Return ---
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        results = []
        for tfqn, cname in sorted_keys[:limit]:
            c = self.get_column(tfqn, cname)
            if c:
                results.append((tfqn, c))
                
        return results

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


