# AstroLlama Phase 2: Full Research Platform

## Vision
A comprehensive AI-powered astronomy research assistant that can:
- Answer questions using your ingested documents (RAG)
- Query any major astronomical catalog/archive
- Execute Python code for analysis and visualization
- Process and analyze astronomical images
- Perform data reduction pipelines

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT WEB UI                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐   │
│  │  Chat   │ │  Tools  │ │   RAG   │ │  Code   │ │  Image Processing   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BEDROCK AGENT ORCHESTRATOR                          │
│                                                                             │
│   AstroLlama (Fine-tuned Llama 3.3 70B) + Tool Use                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│     RAG     │ │  Catalog    │ │   Code      │ │   Image     │ │   Data      │
│  Pinecone   │ │  Queries    │ │  Executor   │ │ Processing  │ │ Reduction   │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
      │              │              │              │              │
      ▼              ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Your Papers │ │ Gaia/2MASS  │ │ matplotlib  │ │ photutils   │ │ astropy     │
│ Textbooks   │ │ WISE/VizieR │ │ astropy     │ │ sep/sewpy   │ │ ccdproc     │
│ Survey Docs │ │ MAST/IRSA   │ │ numpy       │ │ scikit-img  │ │ specutils   │
└─────────────┘ │ ESO/Euclid  │ │ pandas      │ │ reproject   │ │ photutils   │
               └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Components

### 1. RAG System (Pinecone)
**Documents to Ingest:**
- Your own papers/research
- Key brown dwarf papers from ADS (~500-1000 papers)
- Survey documentation:
  - Gaia DR3 documentation
  - Euclid mission papers
  - JWST instrument handbooks
  - 2MASS/WISE documentation
  - SDSS documentation
- Textbooks (if available as PDFs)
- Observation notes/data

**Implementation:**
- Pinecone free tier (100K vectors)
- Sentence-transformers for embeddings
- Chunk size: 512 tokens with overlap

### 2. Catalog Query Tools

| Archive | API | Data Types |
|---------|-----|------------|
| **Gaia** | TAP/ADQL | Astrometry, photometry, spectra |
| **2MASS** | VizieR/IRSA | JHK photometry |
| **WISE/AllWISE** | IRSA | W1-W4 mid-IR photometry |
| **VizieR** | TAP | Any of 20,000+ catalogs |
| **SDSS** | CasJobs/SkyServer | ugriz photometry, spectra |
| **Pan-STARRS** | MAST | grizy photometry |
| **MAST (HST/JWST)** | astroquery | Images, spectra |
| **IRSA** | astroquery | Spitzer, WISE, 2MASS |
| **ESO Archive** | astroquery | VLT/VISTA data |
| **Aladin** | MOC/HiPS | Image cutouts, overlays |
| **Simbad** | TAP | Object cross-matching |
| **NED** | astroquery | Extragalactic data |

### 3. Code Execution Sandbox
**Capabilities:**
- Execute Python in isolated environment
- Pre-installed astronomy packages
- Generate plots (return as images)
- Save results to session

**Packages Available:**
```python
# Core
numpy, scipy, pandas, matplotlib

# Astronomy
astropy, astroquery, photutils, specutils
reproject, regions, ccdproc

# Image processing
scikit-image, opencv-python, sep, photutils

# Machine Learning
scikit-learn, tensorflow (optional)
```

### 4. Image Processing Tools

| Task | Library | Use Case |
|------|---------|----------|
| Source extraction | sep, photutils | Find stars/objects |
| Aperture photometry | photutils | Measure fluxes |
| PSF photometry | photutils | Crowded fields |
| Image alignment | reproject, astroalign | Stack images |
| Background subtraction | sep, photutils | Sky removal |
| WCS operations | astropy.wcs | Coordinate transforms |
| FITS manipulation | astropy.io.fits | Read/write FITS |
| Image cutouts | astropy.nddata | Extract regions |
| Object classification | scikit-learn, CNN | Star/galaxy separation |

### 5. Data Reduction Pipelines

| Pipeline | Tools | Purpose |
|----------|-------|---------|
| CCD reduction | ccdproc | Bias, dark, flat |
| Spectral extraction | specutils | 1D spectra |
| Spectral analysis | specutils | Line fitting |
| Photometric calibration | photutils | Flux calibration |
| Astrometric calibration | astropy, astrometry.net | WCS solving |

---

## Implementation Phases

### Phase 2.1: RAG Setup (Week 1)
- [ ] Set up Pinecone index
- [ ] Create document ingestion pipeline
- [ ] Ingest brown dwarf papers from ADS
- [ ] Ingest survey documentation
- [ ] Test retrieval quality
- [ ] Integrate with chat UI

### Phase 2.2: Extended Catalog Tools (Week 1)
- [ ] Add MAST (HST/JWST) queries
- [ ] Add IRSA (Spitzer/WISE) queries
- [ ] Add ESO archive queries
- [ ] Add Aladin image cutouts
- [ ] Add Simbad/NED cross-matching
- [ ] Test all catalog tools

### Phase 2.3: Code Execution (Week 2)
- [ ] Set up sandboxed Python environment
- [ ] Install astronomy packages
- [ ] Create code execution tool
- [ ] Return plots as images
- [ ] Handle errors gracefully
- [ ] Add to UI

### Phase 2.4: Image Processing (Week 2)
- [ ] Source extraction tool
- [ ] Photometry tools
- [ ] Image alignment tool
- [ ] Background subtraction
- [ ] WCS tools
- [ ] Add to UI

### Phase 2.5: Data Reduction (Week 3)
- [ ] CCD reduction pipeline
- [ ] Spectral extraction
- [ ] Photometric calibration
- [ ] Integration with UI

### Phase 2.6: Bedrock Agent (Week 3)
- [ ] Set up Bedrock Agent
- [ ] Configure action groups
- [ ] Connect all tools
- [ ] Test multi-step reasoning
- [ ] Deploy

---

## File Structure

```
astro_assistant/
├── app/
│   ├── streamlit_app.py          # Main UI
│   ├── pages/
│   │   ├── 1_📊_Data_Query.py    # Catalog queries
│   │   ├── 2_📚_RAG_Search.py    # Document search
│   │   ├── 3_💻_Code_Lab.py      # Code execution
│   │   ├── 4_🖼️_Image_Tools.py   # Image processing
│   │   └── 5_🔬_Reduction.py     # Data reduction
│   └── components/
│       ├── chat.py
│       ├── code_editor.py
│       └── image_viewer.py
├── src/
│   ├── rag/
│   │   ├── embeddings.py         # Generate embeddings
│   │   ├── ingest.py             # Document ingestion
│   │   ├── retriever.py          # RAG retrieval
│   │   └── chunker.py            # Text chunking
│   ├── agents/
│   │   ├── tools.py              # All agent tools
│   │   ├── orchestrator.py       # Agent orchestration
│   │   └── bedrock_agent.py      # Bedrock agent setup
│   ├── catalogs/
│   │   ├── gaia.py               # Gaia queries
│   │   ├── mast.py               # MAST (HST/JWST)
│   │   ├── irsa.py               # IRSA (Spitzer/WISE)
│   │   ├── vizier.py             # VizieR
│   │   ├── eso.py                # ESO archive
│   │   └── aladin.py             # Aladin cutouts
│   ├── code/
│   │   ├── sandbox.py            # Code execution sandbox
│   │   ├── plotter.py            # Plot generation
│   │   └── validator.py          # Code validation
│   ├── imaging/
│   │   ├── source_extraction.py  # Find sources
│   │   ├── photometry.py         # Measure fluxes
│   │   ├── alignment.py          # Align images
│   │   └── background.py         # Background subtraction
│   └── reduction/
│       ├── ccd_pipeline.py       # CCD reduction
│       ├── spectral.py           # Spectral extraction
│       └── calibration.py        # Photometric cal
├── scripts/
│   ├── setup_pinecone.py         # Initialize Pinecone
│   ├── ingest_papers.py          # Ingest ADS papers
│   ├── ingest_docs.py            # Ingest documentation
│   └── setup_agent.py            # Set up Bedrock agent
├── data/
│   ├── rag/
│   │   ├── papers/               # Downloaded papers
│   │   ├── docs/                 # Survey documentation
│   │   └── embeddings/           # Cached embeddings
│   └── temp/                     # Temporary files
└── requirements.txt
```

---

## Cost Estimates

| Component | Monthly Cost |
|-----------|--------------|
| Pinecone (free tier) | $0 |
| AstroLlama inference | $10-50 (usage) |
| Streamlit Cloud | $0 (free tier) |
| Model storage | $2 |
| **Total** | **$12-52/month** |

---

## Next Steps

1. **Today**: Set up Pinecone RAG + ingest papers
2. **Tomorrow**: Extended catalog tools + code execution
3. **This Week**: Image processing + data reduction
4. **Next Week**: Bedrock Agent integration

Ready to start with RAG setup?
