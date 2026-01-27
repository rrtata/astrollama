# AstroLlama: Astronomy Research Assistant

## Fine-tuned Llama 3.3 70B + RAG + Agents on AWS Bedrock

A comprehensive AI system for astronomical research including data mining, catalog queries, literature search, image analysis, and automated plotting.

**Deployment**: AWS Bedrock (us-west-2)  
**Base Model**: Llama 3.3 70B Instruct (`meta.llama3-3-70b-instruct-v1:0:128k`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    (Jupyter / CLI / API / Gradio)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│                         (LangChain / LlamaIndex)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Router    │→ │    Agent     │→ │   Memory     │→ │  Output Parser  │  │
│  │  (Intent)   │  │  Executor    │  │  (Context)   │  │  (Structured)   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────────┐
│     TOOLS LAYER       │ │    RAG LAYER      │ │       LLM LAYER           │
│                       │ │                   │ │                           │
│ • Catalog Query       │ │ • Vector Store    │ │ • Fine-tuned Llama 3.3  │
│ • Cross-Match         │ │   (OpenSearch/    │ │   70B via AWS Bedrock   │
│ • Plotting            │ │    Pinecone)      │ │                           │
│ • Image Reduction     │ │                   │ │ • Context: 128K tokens    │
│ • ADS/Arxiv Search    │ │ • Embeddings      │ │                           │
│ • Citation Generator  │ │   (Titan/BGE)     │ │ • Region: us-west-2       │
│ • Object Detection    │ │ • Document Store  │ │                           │
└───────────────────────┘ └───────────────────┘ └───────────────────────────┘
          │                         │
          ▼                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                              │
│                                                                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ VizieR  │ │  MAST   │ │  Gaia   │ │ NED/    │ │ NASA    │ │  ESA    │ │
│  │ Catalog │ │ Archive │ │ Archive │ │ SIMBAD  │ │ ADS     │ │ Archive │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
astro_assistant/
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml           # API keys, model settings
│   └── prompts.yaml          # System prompts for agents
├── data/
│   ├── training/             # Fine-tuning Q&A pairs
│   │   ├── catalog_queries.jsonl
│   │   ├── data_reduction.jsonl
│   │   ├── plotting.jsonl
│   │   └── literature.jsonl
│   └── rag/                  # Documents for RAG
│       ├── textbooks/
│       └── catalogs/
├── src/
│   ├── __init__.py
│   ├── tools/                # Agent tools
│   │   ├── __init__.py
│   │   ├── catalog_tools.py
│   │   ├── plotting_tools.py
│   │   ├── literature_tools.py
│   │   ├── image_tools.py
│   │   └── crossmatch_tools.py
│   ├── rag/                  # RAG components
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   └── retriever.py
│   ├── agents/               # Agent definitions
│   │   ├── __init__.py
│   │   ├── research_agent.py
│   │   └── analysis_agent.py
│   └── llm/                  # LLM integration
│       ├── __init__.py
│       └── together_client.py
├── notebooks/
│   ├── 01_prepare_training_data.ipynb
│   ├── 02_fine_tune_model.ipynb
│   ├── 03_setup_rag.ipynb
│   └── 04_test_agents.ipynb
└── scripts/
    ├── fine_tune.py
    └── deploy.py
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp config/config.example.yaml config/config.yaml
# Edit with your API keys
```

### 3. Prepare Training Data
```bash
python scripts/prepare_data.py
```

### 4. Fine-tune Model
```bash
python scripts/fine_tune.py --model meta-llama/Llama-3.1-70B-Instruct
```

### 5. Run Assistant
```bash
python -m src.main
```

---

## Cost Estimates

| Phase | Platform | Cost |
|-------|----------|------|
| Fine-tuning | Together.ai / RunPod | $100-300 one-time |
| Inference (1K queries/mo) | Together.ai API | $2-5/month |
| RAG Vector Store | Pinecone Free Tier | $0/month |
| Total Dev Phase | | ~$150-400 |

---

## Capabilities

### ✅ Catalog Queries
- VizieR, MAST, Gaia, SIMBAD, NED
- Cone search, polygon search
- Cross-matching between catalogs

### ✅ Literature & Citations
- NASA ADS search
- Arxiv queries
- BibTeX generation
- Citation formatting (AAS, MNRAS, etc.)

### ✅ Data Visualization
- Color-magnitude diagrams
- Spectral energy distributions
- Light curves
- Sky maps (Aitoff projection)

### ✅ Image Processing
- FITS image reduction
- Source extraction (SEP/SExtractor)
- Astrometric calibration
- Photometric calibration

### ✅ Object Detection & Classification
- Star/galaxy separation
- Photometric redshifts
- Color-based selection cuts
