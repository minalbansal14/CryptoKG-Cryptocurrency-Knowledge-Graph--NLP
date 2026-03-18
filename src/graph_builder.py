"""
graph_builder.py
----------------
Handles all Neo4j interactions:
  - Connecting to a Neo4j instance via py2neo
  - Creating/merging entity nodes with metadata
  - Creating/merging typed relationship edges
  - Storing Word2Vec embeddings per node
  - Bulk-loading extracted triples

Node schema:
    (:Node {name, description, node_labels, url, word_vec})

Edge types (examples):
    invested_in, partnered_with, developed_by, hiring, based_on, ...
"""

import os
import json
import pathlib
from typing import Optional

import numpy as np

try:
    from py2neo import Graph, Node, Relationship
except ImportError:
    raise ImportError("py2neo is required: pip install py2neo")

CREDS_FILE = pathlib.Path(__file__).parent.parent / "neo4j_credentials.json"


def connect(uri: str = None, user: str = None, password: str = None) -> Graph:
    """
    Connect to Neo4j. Falls back to neo4j_credentials.json if args not supplied.

    Returns:
        py2neo Graph object.
    """
    if not (uri and user and password):
        if CREDS_FILE.exists():
            with open(CREDS_FILE) as f:
                creds = json.load(f)
            uri = creds["uri"]
            user = creds["user"]
            password = creds["password"]
        else:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")

    graph = Graph(uri, auth=(user, password))
    print(f"[graph_builder] Connected to Neo4j at {uri}")
    return graph


def create_or_merge_node(
    graph: "Graph",
    name: str,
    description: str = "",
    node_labels: list = None,
    url: str = "",
    word_vec: Optional[np.ndarray] = None,
) -> "Node":
    """
    MERGE a node by name; update its properties if it already exists.

    Args:
        graph:       py2neo Graph.
        name:        Canonical entity name (used as unique key).
        description: Human-readable description (from Google KG API).
        node_labels: List of entity type strings (e.g. ["Organisation"]).
        url:         Wikipedia / Wikidata URL.
        word_vec:    300-dimensional Word2Vec embedding as numpy array.

    Returns:
        py2neo Node.
    """
    node = graph.nodes.match("Node", name=name).first()
    props = {
        "name": name,
        "description": description,
        "node_labels": json.dumps(node_labels or []),
        "url": url,
        "word_vec": word_vec.tolist() if word_vec is not None else [],
    }
    if node is None:
        node = Node("Node", **props)
        graph.create(node)
    else:
        node.update(props)
        graph.push(node)
    return node


def create_or_merge_relationship(
    graph: "Graph",
    subject_node: "Node",
    rel_type: str,
    object_node: "Node",
):
    """
    MERGE a typed relationship between two nodes.

    Args:
        graph:        py2neo Graph.
        subject_node: Source Node.
        rel_type:     Relationship type string (e.g. "invested_in").
        object_node:  Target Node.
    """
    # Normalise to valid Neo4j relationship type (uppercase, spaces→underscores)
    rel_type_clean = rel_type.upper().replace(" ", "_").replace("-", "_")
    rel = Relationship(subject_node, rel_type_clean, object_node)
    graph.merge(rel, rel_type_clean)


def load_triples(graph: "Graph", triples: list, node_metadata: dict = None):
    """
    Bulk-load a list of SPO triples into Neo4j.

    Args:
        graph:         Connected py2neo Graph.
        triples:       List of dicts from triple_extractor.extract_triples_from_sentence.
        node_metadata: Optional dict mapping entity name → {description, node_labels, url, word_vec}.
    """
    node_metadata = node_metadata or {}
    node_cache = {}

    for triple in triples:
        subj_name = triple["subject"]
        obj_name = triple["object"]
        pred = triple["predicate"]

        for name in (subj_name, obj_name):
            if name not in node_cache:
                meta = node_metadata.get(name, {})
                node_cache[name] = create_or_merge_node(
                    graph,
                    name=name,
                    description=meta.get("description", ""),
                    node_labels=meta.get("node_labels", []),
                    url=meta.get("url", ""),
                    word_vec=meta.get("word_vec"),
                )

        create_or_merge_relationship(
            graph,
            subject_node=node_cache[subj_name],
            rel_type=pred,
            object_node=node_cache[obj_name],
        )

    print(f"[graph_builder] Loaded {len(triples)} triples → {len(node_cache)} nodes.")


def fetch_graph_stats(graph: "Graph") -> dict:
    """Return basic statistics about the current graph."""
    node_count = graph.run("MATCH (n) RETURN count(n) AS c").evaluate()
    edge_count = graph.run("MATCH ()-[r]->() RETURN count(r) AS c").evaluate()
    rel_types = graph.run("CALL db.relationshipTypes()").data()
    return {
        "nodes": node_count,
        "edges": edge_count,
        "relationship_types": [r["relationshipType"] for r in rel_types],
    }
