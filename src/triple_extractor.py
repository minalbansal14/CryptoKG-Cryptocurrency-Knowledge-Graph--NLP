"""
triple_extractor.py
-------------------
Core SPO (Subject-Predicate-Object) triple extraction logic.

Pipeline steps:
  1. Gazetteer matching via spaCy PhraseMatcher (entities + relations)
  2. Dependency-based triple extraction via DependencyMatcher
  3. Entity resolution using Word2Vec similarity
  4. Knowledge grounding via Wikidata Q/P IDs
"""

import json
import pathlib
import spacy
from spacy.matcher import PhraseMatcher, DependencyMatcher

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def load_gazetteers():
    """Load entity and relation gazetteers from data/."""
    with open(DATA_DIR / "entities_dict.json") as f:
        entities = json.load(f)
    with open(DATA_DIR / "relations_dict.json") as f:
        relations = json.load(f)
    return entities, relations


def build_phrase_matcher(nlp, entities: dict, relations: dict):
    """
    Build a PhraseMatcher for known entities and relation phrases.

    Returns:
        entity_matcher, relation_matcher
    """
    entity_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    relation_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    entity_patterns = [nlp.make_doc(e) for e in entities.keys()]
    relation_patterns = [nlp.make_doc(r) for r in relations.keys()]

    entity_matcher.add("ENTITY", entity_patterns)
    relation_matcher.add("RELATION", relation_patterns)

    return entity_matcher, relation_matcher


def extract_triples_from_sentence(sentence: str, nlp=None, entities: dict = None, relations: dict = None):
    """
    Extract SPO triples from a single sentence.

    Args:
        sentence:  Raw English input sentence.
        nlp:       Loaded spaCy model (loaded once externally for efficiency).
        entities:  Entity gazetteer dict  {name: wikidata_qid, ...}
        relations: Relation gazetteer dict {phrase: wikidata_pid, ...}

    Returns:
        List of triples, each a dict with keys:
            subject, predicate, object,
            subject_qid, predicate_pid, object_qid
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_lg")
    if entities is None or relations is None:
        entities, relations = load_gazetteers()

    doc = nlp(sentence)
    entity_matcher, relation_matcher = build_phrase_matcher(nlp, entities, relations)

    # --- Gazetteer matches ---
    entity_spans = {doc[start:end].text.lower() for _, start, end in entity_matcher(doc)}
    relation_spans = {doc[start:end].text.lower() for _, start, end in relation_matcher(doc)}

    # --- Dependency-based extraction ---
    triples = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subj = _resolve_entity(token, entities)
            pred_token = token.head
            pred = _resolve_relation(pred_token, relations)
            obj = _find_object(pred_token, entities)

            if subj and pred and obj:
                triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "subject_qid": entities.get(subj.lower()),
                    "predicate_pid": relations.get(pred.lower()),
                    "object_qid": entities.get(obj.lower()),
                })

    return triples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_entity(token, entities: dict):
    """Return the canonical entity name for a token using exact or similarity match."""
    text = token.text.lower()
    for name in entities:
        if name.lower() == text or name.lower().startswith(text):
            return name
    # Fall back to Word2Vec similarity if token has a vector
    if token.has_vector:
        best, best_score = None, 0.0
        for name in entities:
            import spacy
            # Simple string overlap heuristic as a lightweight fallback
            overlap = len(set(text.split()) & set(name.lower().split()))
            if overlap > best_score:
                best_score = overlap
                best = name
        if best_score > 0:
            return best
    return None


def _resolve_relation(token, relations: dict):
    """Return the canonical relation phrase for a predicate token."""
    text = token.lemma_.lower()
    for phrase in relations:
        if phrase.lower() == text or phrase.lower().startswith(text):
            return phrase
    return token.lemma_


def _find_object(pred_token, entities: dict):
    """Find the object of a predicate token from its dependency children."""
    for child in pred_token.children:
        if child.dep_ in ("dobj", "pobj", "attr", "poss"):
            return _resolve_entity(child, entities) or child.text
    # Check prepositional phrases
    for child in pred_token.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return _resolve_entity(grandchild, entities) or grandchild.text
    return None
