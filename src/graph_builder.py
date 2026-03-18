"""
graph_builder.py
----------------
Loads extracted SPO triples into a Neo4j knowledge graph.

Each node is enriched with:
  - name        : entity surface form
  - description : fetched from Google Knowledge Graph API
  - node_labels : entity type labels from the API
  - url         : Wikipedia / Knowledge Graph URL
  - word_vec    : 300-dimensional Word2Vec embedding (stored as a list)

Each edge represents a relation (predicate) between two entity nodes.

Usage:
    from src.graph_builder import GraphBuilder

    builder = GraphBuilder(uri="bolt://localhost:7687", user="neo4j", password="secret")
    builder.add_nodes(node_tuples)
    builder.add_edges(edge_tuples)
"""

import json
import numpy as np
import requests
import spacy
from pathlib import Path
from tqdm import tqdm

try:
    from py2neo import Graph, Node, Relationship, NodeMatcher
    from py2neo.bulk import merge_nodes
    _PY2NEO_AVAILABLE = True
except ImportError:
    _PY2NEO_AVAILABLE = False


def query_google_kg(entity_name: str, api_key: str, limit: int = 1):
    """
    Query the Google Knowledge Graph Search API for an entity.

    Returns:
        (descriptions, node_label_lists, urls) — each a list of length `limit`.
    """
    url = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {"query": entity_name, "key": api_key, "limit": limit, "indent": True}
    response = requests.get(url, params=params)
    data = response.json()

    descriptions, node_labels, urls = [], [], []
    for item in data.get("itemListElement", []):
        result = item.get("result", {})
        desc = result.get("detailedDescription", {}).get("articleBody", "")
        labels = result.get("@type", [])
        url = result.get("detailedDescription", {}).get("url", "")
        descriptions.append(desc)
        node_labels.append(labels)
        urls.append(url)

    return descriptions, node_labels, urls


def create_word_vector(text: str, nlp) -> list[float]:
    """Generate a 300-dimensional Word2Vec embedding for a piece of text."""
    doc = nlp(text)
    if doc.has_vector:
        return [float(v) for v in doc.vector]
    # Random fallback for out-of-vocabulary text
    return [float(v) for v in np.random.uniform(low=-1.0, high=1.0, size=(300,))]


class GraphBuilder:
    """Wraps py2neo operations for bulk-loading a knowledge graph into Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        if not _PY2NEO_AVAILABLE:
            raise ImportError("py2neo is required: pip install py2neo")
        self.graph = Graph(uri, auth=(user, password))
        self.nodes_matcher = NodeMatcher(self.graph)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_nodes(self, node_tuples: list[tuple]) -> None:
        """
        Bulk-upsert nodes into Neo4j.

        Args:
            node_tuples: List of tuples with fields:
                (name, description, node_labels, url, word_vec)
                where word_vec is a list of 300 floats.
        """
        keys = ["name", "description", "node_labels", "url", "word_vec"]
        merge_nodes(self.graph.auto(), node_tuples, ("Node", "name"), keys=keys)
        count = self.graph.nodes.match("Node").count()
        print(f"Nodes in graph: {count}")

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edges(self, edge_tuples: list[tuple]) -> None:
        """
        Create relationships between existing nodes.

        Args:
            edge_tuples: List of tuples with fields:
                (subject_name, predicate_label, object_name, ...)
        """
        # Group edges by predicate label
        edge_groups: dict[str, list] = {}
        for tup in edge_tuples:
            label = tup[1]
            edge_groups.setdefault(label, []).append(tup)

        for edge_label, tuples in tqdm(edge_groups.items(), desc="Adding edges"):
            tx = self.graph.begin()
            for tup in tuples:
                src_node = self.nodes_matcher.match(name=tup[0]).first()
                tgt_node = self.nodes_matcher.match(name=tup[2]).first()

                if not src_node:
                    src_node = Node("Node", name=tup[0])
                    tx.create(src_node)
                if not tgt_node:
                    tgt_node = Node("Node", name=tup[2])
                    tx.create(tgt_node)

                try:
                    rel = Relationship(src_node, edge_label, tgt_node)
                    tx.create(rel)
                except Exception as exc:
                    print(f"  Warning: could not create edge ({tup[0]}) -[{edge_label}]-> ({tup[2]}): {exc}")

            tx.commit()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def deduplicate_nodes(node_tuples: list[tuple]) -> list[tuple]:
        """Remove duplicate nodes by name (keeps first occurrence)."""
        seen: set[str] = set()
        unique = []
        for tup in node_tuples:
            if tup[0] not in seen:
                seen.add(tup[0])
                unique.append(tup)
        return unique

    @staticmethod
    def filter_self_referential(triples: list[tuple]) -> list[tuple]:
        """Remove triples where subject == object."""
        return [t for t in triples if t[0] != t[2]]


if __name__ == "__main__":
    # Quick connectivity check
    import os

    cred_path = Path(__file__).resolve().parent.parent / "neo4j_credentials.json"
    if not cred_path.exists():
        print("No neo4j_credentials.json found — skipping connectivity check.")
    else:
        with open(cred_path) as f:
            creds = json.load(f)
        builder = GraphBuilder(
            uri=creds["uri"], user=creds["user"], password=creds["password"]
        )
        print("Connected to Neo4j:", builder.graph)
