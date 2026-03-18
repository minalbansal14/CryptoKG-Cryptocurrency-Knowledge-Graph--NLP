"""
triple_extractor.py
-------------------
Core SPO (Subject-Predicate-Object) triple extraction pipeline.

Pipeline steps:
  1. Gazetteer-based PhraseMatcher for entity and relation detection
  2. Dependency-based triple extraction via DependencyMatcher
  3. Entity resolution using Word2Vec cosine similarity
  4. Wikidata ID grounding (Q-IDs for entities, P-IDs for relations)

Usage:
    from src.triple_extractor import TripleExtractor

    extractor = TripleExtractor()
    triples = extractor.extract("Ripple partnered with Visa to improve their payment system.")
    # [('Q1307473', 'Ripple', 'P1327', 'partnered with', 'Q328840', 'Visa')]
"""

import spacy
from spacy.matcher import PhraseMatcher, DependencyMatcher

from .gazetteers import (
    load_entities,
    load_relations,
    load_entity_wikidata_map,
    load_relation_wikidata_map,
)


# ---------------------------------------------------------------------------
# Dependency patterns for SPO extraction
# ---------------------------------------------------------------------------
# Pattern: subject <── predicate head ──> object
# Covers active voice (nsubj + dobj/pobj), passive voice (nsubjpass + pobj),
# and possessive modifiers (poss).

_SPO_PATTERNS = [
    # Active: "Ripple partnered with Visa"
    [
        {"RIGHT_ID": "predicate", "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "relcl", "advcl"]}}},
        {"LEFT_ID": "predicate", "REL_OP": ">", "RIGHT_ID": "subject",
         "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
        {"LEFT_ID": "predicate", "REL_OP": ">", "RIGHT_ID": "object",
         "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "attr"]}}},
    ],
    # Prepositional chain: "IBM invested in Stellar"
    [
        {"RIGHT_ID": "predicate", "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "relcl"]}}},
        {"LEFT_ID": "predicate", "REL_OP": ">", "RIGHT_ID": "subject",
         "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
        {"LEFT_ID": "predicate", "REL_OP": ">>", "RIGHT_ID": "object",
         "RIGHT_ATTRS": {"DEP": "pobj"}},
    ],
]


class TripleExtractor:
    """Extracts SPO triples from a sentence using spaCy and gazetteer matching."""

    def __init__(self, custom_ner_path: str | None = None):
        """
        Args:
            custom_ner_path: Optional path to a custom spaCy NER model trained
                             on crypto-domain data (see notebooks/03_custom_ner_training.ipynb).
                             If None, only the default en_core_web_lg NER is used.
        """
        self.nlp = spacy.load("en_core_web_lg")

        if custom_ner_path:
            custom_nlp = spacy.load(custom_ner_path)
            custom_nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])
            self.nlp.add_pipe("ner", source=custom_nlp, name="custom_ner", before="ner")

        self._entity_forms = load_entities()
        self._relation_forms = load_relations()
        self._entity_wikidata = load_entity_wikidata_map()
        self._relation_wikidata = load_relation_wikidata_map()

        # Phrase matchers
        self._rel_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._ent_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        rel_patterns = [self.nlp.make_doc(r) for r in self._relation_forms]
        self._rel_matcher.add("relations", rel_patterns)

        ent_patterns = [self.nlp.make_doc(e) for e in self._entity_forms]
        self._ent_matcher.add("entities", ent_patterns)

        # Dependency matcher
        self._dep_matcher = DependencyMatcher(self.nlp.vocab)
        for i, pattern in enumerate(_SPO_PATTERNS):
            self._dep_matcher.add(f"SPO_{i}", [pattern])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, sentence: str) -> list[tuple]:
        """
        Extract SPO triples from a single sentence.

        Returns:
            List of tuples: (subj_qid, subj_text, pred_pid, pred_text, obj_qid, obj_text)
            Returns an empty list if no triple is found.
        """
        doc = self.nlp(sentence)

        # Step 1 — Phrase matching for relations and entities
        rel_spans = spacy.util.filter_spans(
            [doc[s:e] for _, s, e in self._rel_matcher(doc)]
        )
        ent_spans = spacy.util.filter_spans(
            [doc[s:e] for _, s, e in self._ent_matcher(doc)]
        )

        relations = {span.text.lower(): span for span in rel_spans}
        entities = {span.text: span for span in ent_spans}

        if not relations or not entities:
            return []

        # Step 2 — Dependency-based triple extraction
        raw_triples = self._extract_dep_triples(doc, rel_spans)

        # Step 3 — Entity resolution via Word2Vec similarity
        resolved = self._resolve_entities(raw_triples, entities, doc)

        # Step 4 — Wikidata ID grounding
        grounded = self._ground_wikidata(resolved)

        return grounded

    def extract_batch(self, sentences: list[str]) -> list[list[tuple]]:
        """Extract triples from a list of sentences."""
        return [self.extract(s) for s in sentences]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_dep_triples(self, doc, rel_spans) -> list[tuple]:
        """Use DependencyMatcher to find (subject, predicate_head, object) token triples."""
        triples = []
        dep_matches = self._dep_matcher(doc)

        for match_id, token_ids in dep_matches:
            # token_ids order matches pattern RIGHT_IDs: predicate, subject, object
            if len(token_ids) < 3:
                continue
            predicate_token = doc[token_ids[0]]
            subject_token = doc[token_ids[1]]
            object_token = doc[token_ids[2]]

            # Find the full relation span that contains the predicate head
            pred_span = None
            for span in rel_spans:
                if predicate_token.i >= span.start and predicate_token.i < span.end:
                    pred_span = span
                    break
            if pred_span is None:
                pred_span = predicate_token

            triples.append((subject_token.text, pred_span.text, object_token.text))

        return triples

    def _resolve_entities(self, raw_triples, entity_spans, doc) -> list[tuple]:
        """
        Resolve raw token strings to the best-matching known entity surface form
        using Word2Vec cosine similarity.
        """
        resolved = []
        entity_texts = list(entity_spans.keys())

        for subj_text, pred_text, obj_text in raw_triples:
            subj_resolved = self._best_entity_match(subj_text, entity_texts, doc)
            obj_resolved = self._best_entity_match(obj_text, entity_texts, doc)

            if subj_resolved and obj_resolved and subj_resolved != obj_resolved:
                resolved.append((subj_resolved, pred_text, obj_resolved))

        return resolved

    def _best_entity_match(self, token_text: str, entity_texts: list[str], doc) -> str | None:
        """Return the entity surface form most similar to token_text by Word2Vec cosine similarity."""
        token_doc = self.nlp(token_text)
        if not token_doc.has_vector:
            # Exact or partial string match fallback
            for e in entity_texts:
                if token_text.lower() in e.lower() or e.lower() in token_text.lower():
                    return e
            return None

        best, best_sim = None, -1.0
        for e_text in entity_texts:
            e_doc = self.nlp(e_text)
            if e_doc.has_vector:
                sim = token_doc.similarity(e_doc)
                if sim > best_sim:
                    best_sim = sim
                    best = e_text

        return best if best_sim > 0.5 else None

    def _ground_wikidata(self, resolved: list[tuple]) -> list[tuple]:
        """Map entity/relation surface forms to Wikidata IDs."""
        grounded = []
        for subj_text, pred_text, obj_text in resolved:
            subj_qid = self._entity_wikidata.get(subj_text, "NIL")
            obj_qid = self._entity_wikidata.get(obj_text, "NIL")
            pred_pid = self._relation_wikidata.get(pred_text.lower(), "NIL")
            grounded.append((subj_qid, subj_text, pred_pid, pred_text, obj_qid, obj_text))
        return grounded


if __name__ == "__main__":
    extractor = TripleExtractor()
    test_sentences = [
        "Ripple partnered with Visa to improve their payment system.",
        "IBM invested in Stellar.",
        "Ethereum was developed by Vitalik Buterin in 2014.",
        "Polygon is hiring VC analyst.",
    ]
    for sentence in test_sentences:
        print(f"Input:  {sentence}")
        triples = extractor.extract(sentence)
        for t in triples:
            print(f"Triple: {t}")
        print()
