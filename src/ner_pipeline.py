"""
ner_pipeline.py
---------------
Loads and composes the NLP pipeline:
  - Base model: spaCy en_core_web_lg
  - Custom NER model trained on crypto/fintech domain data
    injected BEFORE the default NER so domain labels take priority.

Custom NER labels:
  CRYPTOCURRENCY  — e.g. Bitcoin, Ethereum, Ripple, Cardano, Polygon
  COMPANY         — e.g. Visa, IBM, Tesla, Santander Bank
  PERSON          — e.g. Vitalik Buterin, Satoshi Nakamoto
"""

import pathlib
import spacy

MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
CUSTOM_NER_PATH = MODEL_DIR / "custom_ner"


def load_pipeline(custom_ner_path=None):
    """
    Load the composed spaCy pipeline.

    Args:
        custom_ner_path: Path to the trained custom NER model directory.
                         Defaults to models/custom_ner/.

    Returns:
        Composed spaCy Language object with custom NER injected before default NER.
    """
    nlp = spacy.load("en_core_web_lg")

    path = custom_ner_path or CUSTOM_NER_PATH
    if path.exists():
        custom_nlp = spacy.load(path)
        nlp.add_pipe("ner", source=custom_nlp, name="custom_ner", before="ner")
        print(f"[ner_pipeline] Custom NER loaded from {path}")
    else:
        print(
            f"[ner_pipeline] WARNING: custom NER model not found at {path}. "
            "Using default spaCy NER only."
        )

    return nlp


def get_entities(doc, labels=("CRYPTOCURRENCY", "COMPANY", "PERSON", "ORG")):
    """
    Extract named entities from a processed spaCy Doc, filtered by label.

    Args:
        doc:    Processed spaCy Doc object.
        labels: Tuple of entity labels to keep.

    Returns:
        List of (text, label) tuples.
    """
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in labels]


# ---------------------------------------------------------------------------
# Training helpers (called from the training notebook)
# ---------------------------------------------------------------------------

def prepare_training_data(annotated_sentences: list):
    """
    Convert annotated sentences into spaCy v3 DocBin training format.

    Args:
        annotated_sentences: List of (text, {"entities": [(start, end, label), ...]}) tuples.

    Returns:
        spacy.tokens.DocBin ready for spacy train.
    """
    from spacy.tokens import DocBin

    nlp = spacy.blank("en")
    db = DocBin()

    for text, annotations in annotated_sentences:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    return db