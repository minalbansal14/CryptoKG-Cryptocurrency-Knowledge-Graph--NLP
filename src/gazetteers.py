"""
gazetteers.py
-------------
Loads entity and relation gazetteers from the data/ directory and returns
flat lists used to initialise spaCy PhraseMatcher instances.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_entities() -> list[str]:
    """Return a flat list of all entity surface forms (labels + aliases)."""
    with open(DATA_DIR / "entities.json", encoding="utf-8") as f:
        records = json.load(f)

    surface_forms: list[str] = []
    for rec in records:
        surface_forms.append(rec["label"])
        surface_forms.extend(rec.get("aliases", []))

    return surface_forms


def load_relations() -> list[str]:
    """Return a flat list of all relation surface forms (labels + aliases)."""
    with open(DATA_DIR / "relations.json", encoding="utf-8") as f:
        records = json.load(f)

    surface_forms: list[str] = []
    for rec in records:
        surface_forms.append(rec["label"])
        surface_forms.extend(rec.get("aliases", []))

    return surface_forms


def load_entity_wikidata_map() -> dict[str, str]:
    """Return a mapping from entity surface form → Wikidata Q-ID."""
    with open(DATA_DIR / "entities.json", encoding="utf-8") as f:
        records = json.load(f)

    mapping: dict[str, str] = {}
    for rec in records:
        qid = rec["qid"]
        mapping[rec["label"]] = qid
        for alias in rec.get("aliases", []):
            mapping[alias] = qid

    return mapping


def load_relation_wikidata_map() -> dict[str, str]:
    """Return a mapping from relation surface form → Wikidata P-ID."""
    with open(DATA_DIR / "relations.json", encoding="utf-8") as f:
        records = json.load(f)

    mapping: dict[str, str] = {}
    for rec in records:
        pid = rec["pid"]
        mapping[rec["label"]] = pid
        for alias in rec.get("aliases", []):
            mapping[alias] = pid

    return mapping


if __name__ == "__main__":
    print("Entities:", load_entities())
    print()
    print("Relations:", load_relations())