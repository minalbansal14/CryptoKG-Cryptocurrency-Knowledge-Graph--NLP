# 🔗 Cryptocurrency Knowledge Graph — NLP Information Extraction Pipeline

> Extracting structured knowledge from unstructured text and building a queryable graph database using NLP, custom NER, and Neo4j.

---

## 📌 Overview

This project builds an end-to-end **Information Extraction (IE) pipeline** that reads plain English sentences about the cryptocurrency and fintech domain and automatically constructs a **knowledge graph** — a database of real-world facts represented as interconnected nodes and relationships.

Given a sentence like:

> *"Ripple partnered with Visa to improve their cross-border payment system."*

The system extracts the structured fact:

> `(Ripple) —[partnered with]→ (Visa)`

...and stores it in Neo4j alongside dozens of other facts, creating a queryable semantic network grounded in Wikidata identifiers.

---

## 🎯 What Problem Does This Solve?

Vast amounts of valuable knowledge exist in unstructured text — news articles, reports, research papers. Reading and manually cataloguing this information is slow and doesn't scale. This project automates that process:

- **Reads** natural language sentences
- **Identifies** named entities (companies, people, cryptocurrencies)
- **Extracts** the relationships between them
- **Stores** everything in a graph database that can be queried, traversed, and reasoned over

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│             domain-specific sentences (crypto/fintech)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     NLP PROCESSING LAYER                            │
│                                                                     │
│   spaCy en_core_web_lg pipeline                                     │
│   ├── Tokenization                                                  │
│   ├── Part-of-Speech Tagging                                        │
│   ├── Dependency Parsing          ← finds grammatical structure     │
│   └── Named Entity Recognition    ← default + custom model          │
│                                                                     │
│   Custom NER Model (trained on domain data)                         │
│   └── Labels: CRYPTOCURRENCY, COMPANY, PERSON                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  INFORMATION EXTRACTION LAYER                       │
│                                                                     │
│  Step 1 — Gazetteer Matching (PhraseMatcher)                        │
│           ├── Entity gazetteer: 27 known entities                   │
│           └── Relation gazetteer: 37 known relation phrases         │
│                                                                     │
│  Step 2 — Dependency-Based Triple Extraction (DependencyMatcher)    │
│           └── Pattern: subject ←── predicate ──→ object             │
│                         nsubj/nsubjpass      dobj/pobj/poss         │
│                                                                     │
│  Step 3 — Entity Resolution (Word2Vec Similarity)                   │
│           └── Maps token candidates → full entity names             │
│                                                                     │
│  Step 4 — Knowledge Grounding (Wikidata)                            │
│           ├── Entities → Q-IDs  (e.g., Visa → Q328840)              │
│           └── Relations → P-IDs (e.g., "developed by" → P61)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE ENRICHMENT LAYER                        │
│                                                                     │
│  Google Knowledge Graph API                                         │
│  └── Fetches descriptions, entity types, URLs for each node         │
│                                                                     │
│  Second-hop Extraction                                              │
│  └── Runs IE pipeline on fetched descriptions → discovers           │
│      additional facts not in original sentences                     │
│                                                                     │
│  Word2Vec Node Embeddings (300-dimensional)                         │
│  └── Every node gets a semantic vector for similarity queries       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER — Neo4j Graph                     │
│                                                                     │
│   26 nodes  │  ~50 edges  │  15 relationship types                  │
│                                                                     │
│   Node properties:  name, description, node_labels, url, word_vec   │
│   Edge types:       invested_in, partnered_with, developed_by,      │
│                     hiring, based_on, ...                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Activities Breakdown

The project is structured across 6 progressive activities:

### Activity 1 — Data Collection
Curated a domain-specific dataset of more than 100 English sentences covering real and fictional relationships between cryptocurrency entities (Ripple, Cardano, Ethereum, Visa, IBM, Tesla, etc.). Sentences were designed to cover varied syntactic structures: active voice, passive voice, compound subjects, and embedded clauses.

### Activity 2 — SPO Triple Extraction
Built the core Information Extraction function that converts any input sentence into one or more **Subject-Predicate-Object triples**. Key components:

- **Gazetteer-based PhraseMatcher** for entity and relation detection
- **Dependency parsing** with spaCy's `DependencyMatcher` to identify grammatical roles
- **Word2Vec similarity** for resolving partial token matches to full entity names
- **Wikidata lookup** to assign Q-IDs (entities) and P-IDs (properties)

### Activity 3 — Custom NER Model Training
The default spaCy NER model (`en_core_web_lg`) fails on domain-specific terms — it classifies "Polygon" as a geometry, not a cryptocurrency. A **custom NER model** was trained using annotated domain data and added to the spaCy pipeline with priority over the default model:

```python
nlp.add_pipe("ner", source=custom_nlp, name="custom_ner", before="ner")
```

### Activity 4 — Extended Extraction
Applied the pipeline to an expanded and more complex sentence set. Addressed additional linguistic patterns including passive constructions with agentive `by`-phrases and multi-clause sentences.

### Activity 5 — Pipeline Refinement
Refined extraction quality with improved dependency patterns and better handling of prepositional object chains (`pobj`), possessive modifiers (`poss`), and clause-embedded relations.

### Activity 6 — Knowledge Graph Construction
Connected all extracted triples to Neo4j:

1. **Google Knowledge Graph API** enriches each entity node with real-world metadata
2. **Second-hop extraction** discovers additional facts from entity descriptions
3. **Word2Vec embeddings** (300d) stored per node enable semantic similarity queries
4. **py2neo** bulk-loads nodes and edges into a live Neo4j instance
5. Final graph: **26 nodes, ~50 relationships, 15 distinct edge types**

---

## 🛠️ Technology Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core implementation |
| NLP Framework | spaCy `en_core_web_lg` | Tokenization, POS, dependency parsing, NER |
| Custom NER | spaCy training pipeline | Domain-specific entity recognition |
| Entity Matching | `PhraseMatcher` | Gazetteer-based entity/relation detection |
| Syntactic Parsing | `DependencyMatcher` | SPO triple extraction |
| Semantic Similarity | Word2Vec (300d vectors) | Entity resolution + node embeddings |
| Knowledge Base | Wikidata (Q/P IDs) | Grounding entities and relations |
| Node Enrichment | Google Knowledge Graph API | Entity descriptions, types, URLs |
| Graph Database | Neo4j | Storage and querying of knowledge graph |
| Graph Driver | py2neo | Python ↔ Neo4j interface |
| Environment | Google Colab | Development and execution |

---

## 📂 Repository Structure

```
knowledge-graph-nlp/
│
├── README.md
│
├── notebooks/
│   ├── 01_data_collection.ipynb          # Corpus and sentence curation
│   ├── 02_triple_extraction.ipynb         # Core SPO extraction algorithm
│   ├── 03_custom_ner_training.ipynb       # NER model training
│   ├── 04_extended_extraction.ipynb       # Expanded sentence set
│   ├── 05_pipeline_refinement.ipynb       # Improved dependency patterns
│   └── 06_knowledge_graph_construction.ipynb  # Neo4j graph building
│
├── src/
│   ├── triple_extractor.py                # Extraction logic (portable)
│   ├── ner_pipeline.py                    # NER model loader
│   └── graph_builder.py                   # Neo4j node/edge loading
│
├── data/
│   ├── sentences.txt                      # Input corpus
│   ├── entities_dict.json                 # Entity gazetteer + Wikidata IDs
│   └── relations_dict.json                # Relation gazetteer + Wikidata IDs
│
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
pip install spacy py2neo neo4j requests numpy tqdm
python -m spacy download en_core_web_lg
```

### Running the Extraction

```python
from src.triple_extractor import extract_triples_from_sentence

sentence = "Ripple partnered with Visa to improve their payment system."
triples = extract_triples_from_sentence(sentence)

# Output: [('Q1307473', 'Ripple'), ('P1327', 'partnered with'), ('Q328840', 'Visa')]
```

### Neo4j Connection

Create a `.env` file or credentials JSON (not committed to git):

```json
{
  "uri": "bolt://localhost:7687",
  "user": "neo4j",
  "password": "your_password"
}
```

Then run `notebooks/06_knowledge_graph_construction.ipynb` end to end.

---

## 📊 Sample Extracted Triples

| Subject | Predicate | Object |
|---|---|---|
| Ethereum | developed by | Vitalik Buterin |
| Ripple | partnered with | Visa |
| IBM | invested in | Stellar |
| Tesla | invests | Bitcoin |
| Cardano | partnered | Ethereum |
| Santander Bank | joined forces | Visa |
| Visa | collaborates with | cryptocurrency platforms |
| Polygon | hiring | VC analyst |

---

## 🔍 Example Neo4j Queries

Once the graph is built, you can query it directly:

```cypher
-- Find everything Ripple is connected to
MATCH (n {name: "Ripple"})-[r]->(m) RETURN n, r, m

-- Find all investment relationships
MATCH (a)-[:invested_in]->(b) RETURN a.name, b.name

-- Find nodes similar to Ethereum (requires GDS plugin for vector similarity)
MATCH (n:Node) WHERE n.name <> "Ethereum"
RETURN n.name, gds.similarity.cosine(n.word_vec, $ethereum_vec) AS similarity
ORDER BY similarity DESC LIMIT 5
```

---

## 🚧 Known Limitations

- **Closed-world entity recognition**: the pipeline only recognises entities and relations explicitly listed in the gazetteers. Novel entities in unseen sentences will be missed.
- **Self-referential triples**: for symmetric relations like "partners with", the subject resolution occasionally produces `(X, partners with, X)` due to how Word2Vec similarity selects candidates. Filtering is applied post-extraction but is not exhaustive.
- **Dependency parser sensitivity**: deeply nested or ambiguous sentences can produce incorrect subject/object assignments — a known limitation of rule-based dependency extraction without neural coreference resolution.
- **Google API dependency**: the enrichment layer requires a valid API key and internet access. Without it, nodes are stored with empty metadata fields.

---

## 🔭 Potential Extensions

- Replace gazetteer-based extraction with a **fine-tuned relation extraction model** (e.g., REBEL, LUKE) for open-domain coverage
- Add **coreference resolution** (e.g., NeuralCoref) to handle pronouns across sentences
- Integrate a **SPARQL endpoint** to auto-populate Wikidata properties instead of manual ID mapping
- Build a **graph neural network** on top of the Neo4j graph using the stored word vectors for entity classification or link prediction

---

## 👥 Team

Developed as part of an Information Extraction course project.

---

## 📄 License

MIT License — see `LICENSE` for details.