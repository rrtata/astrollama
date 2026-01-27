# AstroLlama Phase 2: RAG, Agents & Web UI

## Overview

This phase adds:
1. **RAG (Retrieval Augmented Generation)** - Query your own astronomy documents
2. **Agents with Tools** - Query ADS, arXiv, Gaia, SDSS, 2MASS, WISE catalogs
3. **Public Web UI** - Shareable interface for colleagues

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (Streamlit)                       │
│                    (Public URL via AWS/Hugging Face)            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Router    │  │   Auth      │  │   Session Management    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│  AstroLlama     │   │      RAG        │   │      Agents         │
│  (Bedrock)      │   │   (Pinecone)    │   │    (Tool Use)       │
│                 │   │                 │   │                     │
│ Fine-tuned      │   │ - Papers        │   │ - ADS Search        │
│ Llama 3.3 70B   │   │ - Textbooks     │   │ - arXiv Search      │
│                 │   │ - Docs          │   │ - Gaia Query        │
│                 │   │                 │   │ - SDSS Query        │
└─────────────────┘   └─────────────────┘   │ - 2MASS Query       │
                                            │ - WISE Query        │
                                            │ - VizieR Query      │
                                            └─────────────────────┘
```

---

## Components

### 1. RAG with Pinecone (FREE tier)
- Vector store for astronomy documents
- Embed papers, textbooks, documentation
- Retrieve relevant context for queries

### 2. Agent Tools
| Tool | Description | API |
|------|-------------|-----|
| `search_ads` | Search NASA ADS for papers | NASA ADS API |
| `search_arxiv` | Search arXiv preprints | arXiv API |
| `query_gaia` | Query Gaia DR3 catalog | Gaia TAP |
| `query_sdss` | Query SDSS photometry/spectra | SDSS CasJobs |
| `query_2mass` | Query 2MASS catalog | VizieR |
| `query_wise` | Query AllWISE catalog | IRSA |
| `query_vizier` | Query any VizieR catalog | VizieR TAP |

### 3. Web UI Options

| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **Streamlit Cloud** | Easy, free hosting | Limited customization | FREE |
| **Hugging Face Spaces** | Free, good for ML | Some limitations | FREE |
| **AWS App Runner** | Scalable, AWS native | More setup | ~$5-25/mo |
| **EC2 + nginx** | Full control | More maintenance | ~$10-50/mo |

**Recommendation**: Start with **Streamlit Cloud** (free) or **Hugging Face Spaces** (free)

---

## Implementation Plan

### Phase 2.1: RAG Setup (Day 1)
- [ ] Create Pinecone index
- [ ] Ingest astronomy documents
- [ ] Test retrieval

### Phase 2.2: Agent Tools (Day 2)
- [ ] Implement ADS search tool
- [ ] Implement arXiv search tool
- [ ] Implement catalog query tools
- [ ] Test tool calling with Bedrock

### Phase 2.3: Web UI (Day 3)
- [ ] Create Streamlit app
- [ ] Add authentication (optional)
- [ ] Deploy to cloud
- [ ] Share with colleagues

---

## Quick Start

```bash
# 1. Setup RAG
python scripts/setup_rag.py setup
python scripts/setup_rag.py ingest --source ./data/papers/

# 2. Test agents
python scripts/test_agents.py

# 3. Run web UI locally
streamlit run app/streamlit_app.py

# 4. Deploy
streamlit deploy app/streamlit_app.py
```

---

## Files to Create

```
astro_assistant/
├── app/
│   ├── streamlit_app.py      # Main web UI
│   ├── api_server.py         # FastAPI backend
│   └── config.py             # App configuration
├── src/
│   ├── rag/
│   │   ├── embeddings.py     # Embedding generation
│   │   ├── retriever.py      # RAG retrieval
│   │   └── ingest.py         # Document ingestion
│   ├── agents/
│   │   ├── tools.py          # All agent tools
│   │   ├── ads_tool.py       # ADS search
│   │   ├── arxiv_tool.py     # arXiv search
│   │   ├── catalog_tools.py  # Gaia, SDSS, etc.
│   │   └── agent.py          # Main agent orchestrator
│   └── llm/
│       └── bedrock_client.py # Bedrock API client
├── scripts/
│   ├── setup_rag.py          # RAG setup script
│   └── deploy.py             # Deployment script
└── requirements.txt          # Dependencies
```

---

## Estimated Costs (Monthly)

| Component | Cost |
|-----------|------|
| Pinecone (Free tier) | $0 |
| AstroLlama inference | ~$5-20 (usage-based) |
| Streamlit Cloud | $0 (free tier) |
| Model storage | ~$1-2 |
| **Total** | **~$6-25/month** |

---

## Next Steps

1. Run `python scripts/setup_phase2.py` to create all files
2. Configure Pinecone API key
3. Ingest your documents
4. Test locally
5. Deploy to Streamlit Cloud

Ready to proceed?
