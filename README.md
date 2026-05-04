# Mental Health RAG + KG System

A hallucination-mitigating mental health dialogue system combining **Retrieval-Augmented Generation (RAG)** with a **dynamic Knowledge Graph (KG)** and confidence-based fusion.

## Architecture

```
User Input
    |
    v
Safety Check (Keyword + LLM)
    |
    v
Input Processor (Tokenize -> Embed -> Extract Symptoms -> Refine Query)
    |
    +-------------------+-------------------+
    |                                       |
    v                                       v
RAG Engine                              KG Engine
(Retrieval + Generation)                (Graph Query + Matching)
    |                                       |
    v                                       v
LLM Confidence (R,G,C)                  KG Confidence (K,M,G)
    |                                       |
    +-------------------+-------------------+
                        |
                        v
              Fusion Layer
         (Hallucination Detection)
                        |
                        v
              Final Response
    (High-Confidence, Low-Hallucination)
```

## Folder Structure

```
mental_health_rag_kg/
├── notebooks/              # Colab notebooks (run in order)
│   ├── 01_setup_and_install.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_build_rag.ipynb
│   ├── 04_build_kg.ipynb
│   ├── 05_pipeline_test.ipynb
│   └── 06_evaluation.ipynb
├── src/                    # Core modules
│   ├── llm_client.py       # SambaNova API wrapper
│   ├── safety_checker.py   # Safety filtering
│   ├── input_processor.py  # Preprocessing
│   ├── rag_engine.py       # RAG + confidence scoring
│   ├── kg_engine.py        # KG + confidence scoring
│   ├── fusion_layer.py     # Score fusion
│   └── pipeline.py         # Main orchestrator
├── rag/                    # RAG components
│   ├── pubmed_fetcher.py
│   ├── document_processor.py
│   ├── embedder.py
│   └── vector_store.py
├── kg/                     # KG components
│   ├── builder.py
│   ├── scraper.py
│   └── query.py
├── configs/
│   ├── config.yaml         # All hyperparameters
│   └── prompts.yaml        # LLM prompts
├── data/                   # Data storage
├── tests/                  # Unit tests
└── requirements.txt
```

## Setup Instructions (Google Colab)

### Step 1: Upload to Drive
1. Copy this entire folder to `MyDrive/mental_health_rag_kg/`

### Step 2: Run Notebooks in Order
1. Open `notebooks/01_setup_and_install.ipynb` in Colab
2. Mount Drive and install dependencies
3. Create `.env` file with your API keys
4. Run `02_data_collection.ipynb` to fetch PubMed articles
5. Run `03_build_rag.ipynb` to build FAISS index
6. Run `04_build_kg.ipynb` to build knowledge graph
7. Run `05_pipeline_test.ipynb` to test the pipeline
8. Run `06_evaluation.ipynb` for full evaluation

### API Keys Required
- **SambaNova API Key**: For LLM (Meta-Llama-3.3-70B-Instruct) and Safety Check
- **PubMed API Key**: For fetching medical articles

Add these to a `.env` file in the project root.

## Confidence Scoring

### LLM Confidence: C = f(R, G, C)
- **R** (Retrieval Quality): Average similarity of retrieved docs
- **G** (Grounding Score): Token overlap between response and context
- **C** (Consistency Score): Relevance to query + contradiction detection

### KG Confidence: K = f(K, M, G)
- **K** (Known Match): Ratio of user symptoms found in KG
- **M** (Alignment): Coverage of matched conditions
- **G** (Graph Consistency): Structural validity of matches

### Fusion Rules
- Both high (>0.75): Merge responses
- KG high only: KG-dominant answer
- RAG high + low hallucination: RAG-dominant answer
- Both medium: Cautious merged response
- Both low: `[REJ]` Abstention

## Safety Features
1. **Keyword filtering**: Fast check for self-harm terms
2. **LLM semantic check**: Meta-Llama-3.3-70B classifies intent
3. **Automatic rejection**: `[REJ]` response with crisis resources
4. **No medical advice disclaimer**: All outputs include disclaimer

## Evaluation
The system is evaluated on:
- **Safety Accuracy**: Correct rejection of harmful queries
- **Confidence Calibration**: High confidence = high quality
- **Hallucination Rate**: Detected via KG fact verification
- **Abstention Rate**: Appropriate uncertainty handling

## License
Research / Educational Use
