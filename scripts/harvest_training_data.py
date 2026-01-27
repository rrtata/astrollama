#!/usr/bin/env python3
"""
AstroLlama - Comprehensive Astronomy Data Harvester
Downloads training material from multiple sources for fine-tuning.

Focus areas:
- Substellar astronomy (brown dwarfs, exoplanets, ultracool dwarfs)
- Euclid space telescope
- Major surveys (Gaia, SDSS, 2MASS, WISE, JWST)
- Textbooks and review articles

Sources:
- arXiv (astro-ph papers)
- NASA ADS (abstracts, metadata)
- Survey documentation
- Open access textbooks
- Catalog schemas

Usage:
    python harvest_training_data.py --output ./data/raw/ --max-size 500
    python harvest_training_data.py --topics substellar,euclid --papers 5000
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.parse
import re

# Check for required packages
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("Note: arxiv package not installed. Run: pip install arxiv")

try:
    import ads
    ADS_AVAILABLE = True
except ImportError:
    ADS_AVAILABLE = False
    print("Note: ads package not installed. Run: pip install ads")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Note: requests package not installed. Run: pip install requests")


# =============================================================================
# Configuration
# =============================================================================

# arXiv categories for astronomy
ARXIV_CATEGORIES = {
    "astro-ph.EP": "Earth and Planetary Astrophysics (Exoplanets)",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.IM": "Instrumentation and Methods",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
}

# Search terms for different topics
SEARCH_TOPICS = {
    "substellar": [
        "brown dwarf",
        "ultracool dwarf",
        "L dwarf",
        "T dwarf",
        "Y dwarf",
        "substellar object",
        "planetary mass object",
        "free-floating planet",
        "ultracool atmosphere",
        "methane dwarf",
        "brown dwarf binary",
        "substellar companion",
        "brown dwarf formation",
        "deuterium burning",
        "lithium test",
        "ultracool dwarf variability",
        "brown dwarf weather",
        "substellar benchmark",
    ],
    "exoplanets": [
        "exoplanet",
        "transiting planet",
        "radial velocity planet",
        "hot Jupiter",
        "super Earth",
        "mini Neptune",
        "habitable zone",
        "exoplanet atmosphere",
        "transmission spectroscopy",
        "emission spectroscopy",
        "phase curve",
        "secondary eclipse",
        "planetary formation",
        "protoplanetary disk",
        "debris disk",
        "planet migration",
        "Kepler planet",
        "TESS planet",
        "direct imaging exoplanet",
    ],
    "euclid": [
        "Euclid mission",
        "Euclid telescope",
        "Euclid survey",
        "Euclid cosmology",
        "Euclid weak lensing",
        "Euclid photometric redshift",
        "Euclid galaxy clustering",
        "Euclid dark energy",
        "Euclid VIS",
        "Euclid NISP",
        "Euclid data release",
        "Euclid pipeline",
        "Euclid calibration",
    ],
    "jwst": [
        "JWST",
        "James Webb Space Telescope",
        "JWST NIRCam",
        "JWST NIRSpec",
        "JWST MIRI",
        "JWST observation",
        "JWST brown dwarf",
        "JWST exoplanet",
        "JWST galaxy",
        "JWST early release",
    ],
    "gaia": [
        "Gaia DR3",
        "Gaia astrometry",
        "Gaia photometry",
        "Gaia radial velocity",
        "Gaia parallax",
        "Gaia proper motion",
        "Gaia binary",
        "Gaia variability",
        "Gaia XP spectra",
        "Gaia astrophysical parameters",
    ],
    "spectroscopy": [
        "stellar spectroscopy",
        "spectral classification",
        "spectral template",
        "equivalent width",
        "line profile",
        "abundance analysis",
        "spectroscopic binary",
        "spectral energy distribution",
        "SED fitting",
        "synthetic spectrum",
    ],
    "surveys": [
        "SDSS spectroscopy",
        "2MASS catalog",
        "WISE survey",
        "Pan-STARRS",
        "VISTA survey",
        "UKIDSS",
        "VHS survey",
        "ZTF",
        "LSST",
        "Roman Space Telescope",
    ],
}

# Documentation URLs
DOCUMENTATION_URLS = {
    # Euclid
    "euclid_overview": "https://www.cosmos.esa.int/web/euclid",
    "euclid_consortium": "https://www.euclid-ec.org/",
    
    # Gaia
    "gaia_dr3_doc": "https://gea.esac.esa.int/archive/documentation/GDR3/index.html",
    "gaia_datamodel": "https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/",
    
    # JWST
    "jwst_docs": "https://jwst-docs.stsci.edu/",
    
    # Brown dwarf archives
    "ultracool_sheet": "https://zenodo.org/records/4169085",  # Ultracool dwarf database
    
    # Exoplanet archives
    "nasa_exoplanet_archive": "https://exoplanetarchive.ipac.caltech.edu/",
    
    # SDSS
    "sdss_dr18_doc": "https://www.sdss.org/dr18/",
}


# =============================================================================
# arXiv Paper Harvesting
# =============================================================================

def search_arxiv_papers(query: str, max_results: int = 100, 
                        category: str = "astro-ph") -> List[Dict]:
    """Search arXiv for papers matching query."""
    if not ARXIV_AVAILABLE:
        print("arxiv package not available")
        return []
    
    search = arxiv.Search(
        query=f"cat:{category} AND ({query})",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in search.results():
        papers.append({
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "categories": result.categories,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
        })
    
    return papers


def download_arxiv_pdfs(papers: List[Dict], output_dir: Path, 
                        max_downloads: int = 100) -> int:
    """Download PDFs from arXiv."""
    if not ARXIV_AVAILABLE:
        return 0
    
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    for paper in papers[:max_downloads]:
        arxiv_id = paper["arxiv_id"]
        pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        
        if pdf_path.exists():
            downloaded += 1
            continue
        
        try:
            # Download PDF
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            result.download_pdf(dirpath=str(pdf_dir), 
                              filename=f"{arxiv_id.replace('/', '_')}.pdf")
            downloaded += 1
            print(f"  Downloaded: {arxiv_id}")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  Failed to download {arxiv_id}: {e}")
    
    return downloaded


def harvest_arxiv_by_topic(topic: str, output_dir: Path, 
                           max_papers: int = 500,
                           download_pdfs: bool = True) -> Dict:
    """Harvest arXiv papers for a specific topic."""
    
    if topic not in SEARCH_TOPICS:
        print(f"Unknown topic: {topic}")
        return {}
    
    all_papers = []
    search_terms = SEARCH_TOPICS[topic]
    papers_per_term = max_papers // len(search_terms)
    
    print(f"\nHarvesting {topic} papers from arXiv...")
    
    for term in search_terms:
        print(f"  Searching: {term}")
        papers = search_arxiv_papers(term, max_results=papers_per_term)
        all_papers.extend(papers)
        print(f"    Found {len(papers)} papers")
        time.sleep(2)  # Rate limiting
    
    # Deduplicate by arxiv_id
    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique_papers.append(p)
    
    print(f"  Total unique papers: {len(unique_papers)}")
    
    # Save metadata
    topic_dir = output_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = topic_dir / "papers_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(unique_papers, f, indent=2)
    
    # Save abstracts as training text
    abstracts_file = topic_dir / "abstracts.jsonl"
    with open(abstracts_file, "w") as f:
        for p in unique_papers:
            entry = {
                "text": f"Title: {p['title']}\n\nAbstract: {p['abstract']}",
                "source": f"arXiv:{p['arxiv_id']}",
                "topic": topic
            }
            f.write(json.dumps(entry) + "\n")
    
    # Download PDFs
    pdf_count = 0
    if download_pdfs:
        pdf_count = download_arxiv_pdfs(unique_papers, topic_dir, 
                                        max_downloads=min(100, len(unique_papers)))
    
    return {
        "topic": topic,
        "papers_found": len(unique_papers),
        "pdfs_downloaded": pdf_count,
        "metadata_file": str(metadata_file),
        "abstracts_file": str(abstracts_file)
    }


# =============================================================================
# NASA ADS Harvesting
# =============================================================================

def search_ads_papers(query: str, rows: int = 100, 
                      years: str = "2019-2025") -> List[Dict]:
    """Search NASA ADS for papers."""
    if not ADS_AVAILABLE:
        print("ads package not available")
        return []
    
    # Check for API token
    if not ads.config.token:
        token = os.environ.get("ADS_TOKEN") or os.environ.get("ADS_DEV_KEY")
        if token:
            ads.config.token = token
        else:
            print("ADS_TOKEN not set - skipping ADS search")
            return []
    
    try:
        search = ads.SearchQuery(
            q=query,
            fq=f"year:{years}",
            sort="citation_count desc",
            rows=rows,
            fl=["bibcode", "title", "abstract", "author", "year", 
                "citation_count", "keyword", "doi"]
        )
        
        papers = []
        for paper in search:
            papers.append({
                "bibcode": paper.bibcode,
                "title": paper.title[0] if paper.title else "",
                "abstract": paper.abstract or "",
                "authors": paper.author[:10] if paper.author else [],
                "year": paper.year,
                "citations": paper.citation_count,
                "keywords": paper.keyword or [],
                "doi": paper.doi[0] if paper.doi else None
            })
        
        return papers
    except Exception as e:
        print(f"ADS search error: {e}")
        return []


def harvest_ads_reviews(topic: str, output_dir: Path, max_papers: int = 200) -> Dict:
    """Harvest review articles from ADS."""
    
    review_queries = {
        "substellar": [
            "brown dwarf review",
            "ultracool dwarf comprehensive",
            "substellar atmosphere model",
            "brown dwarf population",
        ],
        "exoplanets": [
            "exoplanet review",
            "exoplanet atmosphere comprehensive",
            "planet formation review",
            "habitable zone review",
        ],
        "euclid": [
            "Euclid mission overview",
            "Euclid science",
            "Euclid preparation",
        ],
    }
    
    queries = review_queries.get(topic, [f"{topic} review"])
    
    all_papers = []
    for query in queries:
        print(f"  ADS search: {query}")
        papers = search_ads_papers(query, rows=max_papers // len(queries))
        all_papers.extend(papers)
        time.sleep(1)
    
    # Deduplicate
    seen = set()
    unique = []
    for p in all_papers:
        if p["bibcode"] not in seen:
            seen.add(p["bibcode"])
            unique.append(p)
    
    # Save
    topic_dir = output_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    ads_file = topic_dir / "ads_papers.json"
    with open(ads_file, "w") as f:
        json.dump(unique, f, indent=2)
    
    # Create training text
    ads_text_file = topic_dir / "ads_abstracts.jsonl"
    with open(ads_text_file, "w") as f:
        for p in unique:
            if p["abstract"]:
                entry = {
                    "text": f"Title: {p['title']}\n\nAbstract: {p['abstract']}",
                    "source": f"ADS:{p['bibcode']}",
                    "citations": p["citations"],
                    "topic": topic
                }
                f.write(json.dumps(entry) + "\n")
    
    return {
        "topic": topic,
        "papers_found": len(unique),
        "file": str(ads_file)
    }


# =============================================================================
# Documentation Harvesting
# =============================================================================

def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a file from URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AstroLlama/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def harvest_survey_schemas(output_dir: Path) -> Dict:
    """Download catalog schemas and column descriptions."""
    
    schemas_dir = output_dir / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    
    # Gaia DR3 table descriptions
    gaia_tables = [
        ("gaia_source", "Main source catalog with astrometry and photometry"),
        ("astrophysical_parameters", "Stellar parameters from GSP-Phot and GSP-Spec"),
        ("vari_summary", "Summary of variable source classifications"),
        ("nss_two_body_orbit", "Non-single star orbital solutions"),
        ("xp_summary", "BP/RP spectra summary"),
    ]
    
    gaia_schema = {
        "survey": "Gaia DR3",
        "description": "Gaia Data Release 3 - 1.8 billion sources",
        "tables": {}
    }
    
    for table_name, desc in gaia_tables:
        gaia_schema["tables"][table_name] = {
            "description": desc,
            "key_columns": get_gaia_columns(table_name)
        }
    
    with open(schemas_dir / "gaia_dr3_schema.json", "w") as f:
        json.dump(gaia_schema, f, indent=2)
    
    # Other survey schemas
    surveys = {
        "2MASS": {
            "description": "Two Micron All Sky Survey - JHK photometry",
            "columns": ["ra", "dec", "j_m", "h_m", "k_m", "j_msigcom", "h_msigcom", "k_msigcom", "ph_qual", "rd_flg"]
        },
        "WISE": {
            "description": "Wide-field Infrared Survey Explorer - W1-W4 bands",
            "columns": ["ra", "dec", "w1mpro", "w2mpro", "w3mpro", "w4mpro", "w1sigmpro", "w2sigmpro", "cc_flags", "ext_flg"]
        },
        "SDSS": {
            "description": "Sloan Digital Sky Survey - ugriz photometry and spectroscopy",
            "columns": ["ra", "dec", "u", "g", "r", "i", "z", "type", "class", "subclass", "z_spec"]
        },
        "Pan-STARRS": {
            "description": "Panoramic Survey Telescope and Rapid Response System - grizy",
            "columns": ["raMean", "decMean", "gMeanPSFMag", "rMeanPSFMag", "iMeanPSFMag", "zMeanPSFMag", "yMeanPSFMag"]
        },
    }
    
    with open(schemas_dir / "survey_schemas.json", "w") as f:
        json.dump(surveys, f, indent=2)
    
    return {"schemas_saved": len(surveys) + 1}


def get_gaia_columns(table_name: str) -> List[Dict]:
    """Get key column descriptions for Gaia tables."""
    columns = {
        "gaia_source": [
            {"name": "source_id", "description": "Unique source identifier"},
            {"name": "ra", "description": "Right ascension (deg)"},
            {"name": "dec", "description": "Declination (deg)"},
            {"name": "parallax", "description": "Parallax (mas)"},
            {"name": "parallax_error", "description": "Parallax uncertainty (mas)"},
            {"name": "pmra", "description": "Proper motion in RA (mas/yr)"},
            {"name": "pmdec", "description": "Proper motion in Dec (mas/yr)"},
            {"name": "phot_g_mean_mag", "description": "G-band mean magnitude"},
            {"name": "bp_rp", "description": "BP-RP color"},
            {"name": "radial_velocity", "description": "Radial velocity (km/s)"},
            {"name": "ruwe", "description": "Renormalized unit weight error"},
        ],
        "astrophysical_parameters": [
            {"name": "teff_gspphot", "description": "Effective temperature from GSP-Phot (K)"},
            {"name": "logg_gspphot", "description": "Surface gravity from GSP-Phot (log cgs)"},
            {"name": "mh_gspphot", "description": "Metallicity from GSP-Phot ([M/H])"},
            {"name": "ag_gspphot", "description": "Extinction in G band (mag)"},
            {"name": "distance_gspphot", "description": "Distance from GSP-Phot (pc)"},
        ],
        "vari_summary": [
            {"name": "source_id", "description": "Source identifier"},
            {"name": "num_selected_g_fov", "description": "Number of G-band observations"},
            {"name": "mean_mag_g_fov", "description": "Mean G magnitude"},
            {"name": "std_dev_mag_g_fov", "description": "Standard deviation of G mag"},
            {"name": "best_class_name", "description": "Best variability classification"},
        ],
        "nss_two_body_orbit": [
            {"name": "source_id", "description": "Source identifier"},
            {"name": "period", "description": "Orbital period (days)"},
            {"name": "eccentricity", "description": "Orbital eccentricity"},
            {"name": "semi_amplitude_primary", "description": "RV semi-amplitude (km/s)"},
        ],
        "xp_summary": [
            {"name": "source_id", "description": "Source identifier"},
            {"name": "bp_n_relevant_basisfunctions", "description": "Number of BP basis functions"},
            {"name": "rp_n_relevant_basisfunctions", "description": "Number of RP basis functions"},
        ],
    }
    return columns.get(table_name, [])


# =============================================================================
# Substellar Astronomy Specific Data
# =============================================================================

def create_substellar_qa_data(output_dir: Path) -> int:
    """Create Q&A training data specific to substellar astronomy."""
    
    qa_pairs = [
        # Brown dwarf classification
        {
            "user": "What is the spectral sequence for brown dwarfs?",
            "assistant": """Brown dwarfs follow a spectral sequence that extends from late M dwarfs through L, T, and Y types:

**M dwarfs (late)**: M7-M9.5
- Temperature: ~2700-2400 K
- Features: TiO, VO bands dominant
- Some may be brown dwarfs (age-dependent)

**L dwarfs**: L0-L9
- Temperature: ~2400-1400 K
- Features: TiO/VO disappear, metal hydrides (FeH, CrH) appear
- Dust clouds form in atmosphere
- Alkali lines (Na I, K I) prominent

**T dwarfs**: T0-T9
- Temperature: ~1400-500 K
- Features: Strong CH4 absorption (methane)
- H2O bands strengthen
- Dust clouds sink below photosphere
- Blue J-K colors

**Y dwarfs**: Y0-Y2+
- Temperature: <500 K (some <300 K)
- Features: NH3 absorption appears
- Very red colors
- Extremely faint (M_J > 20)

```python
# Spectral type to temperature (approximate)
def spt_to_teff(spt):
    '''Convert spectral type to Teff (rough estimate).'''
    spt_dict = {
        'M7': 2700, 'M8': 2500, 'M9': 2400,
        'L0': 2200, 'L2': 2000, 'L4': 1800, 
        'L6': 1600, 'L8': 1400,
        'T0': 1300, 'T2': 1200, 'T4': 1100,
        'T6': 900, 'T8': 700,
        'Y0': 450, 'Y1': 350
    }
    return spt_dict.get(spt, None)
```"""
        },
        {
            "user": "How do I distinguish brown dwarfs from low-mass stars?",
            "assistant": """Distinguishing brown dwarfs from low-mass stars requires multiple diagnostics:

**1. Lithium Test**
Brown dwarfs below ~65 M_Jup preserve lithium; stars deplete it.

```python
# Check for Li I 6708 Å absorption
def check_lithium(spectrum, wavelength):
    '''Look for Li I resonance line.'''
    li_region = spectrum[(wavelength > 6705) & (wavelength < 6712)]
    # Li EW > 0.5 Å suggests substellar for late-M/L types
    return measure_ew(li_region, 6707.8)
```

**2. Luminosity/Age Degeneracy**
Young brown dwarfs can be as luminous as old stars - need age estimate.

```python
# Mass-luminosity depends on age
from astropy import units as u

def estimate_mass(L_bol, age_Myr, model='ATMO2020'):
    '''Use evolutionary models to estimate mass.'''
    # Substellar boundary: ~75 M_Jup
    # Deuterium burning: ~13 M_Jup
    # Example: L ~ 10^-4 L_sun at 1 Gyr → ~60 M_Jup
    pass
```

**3. Spectral Features**
- Late-M/early-L: gravity-sensitive features (VO, alkali lines)
- FeH at 0.99 μm weakens in young objects
- H-band shape (triangular in young, peaked in old)

**4. Kinematics**
Young (substellar-likely) objects have low velocity dispersion:
```python
# Young disk: σ_U,V,W ~ 10-20 km/s
# Old disk: σ_U,V,W ~ 30-50 km/s
total_velocity = np.sqrt(U**2 + V**2 + W**2)
likely_young = total_velocity < 40  # km/s
```

**5. Color-Magnitude Position**
Young objects are overluminous for their color:
```python
# M_J vs J-K diagram
# Substellar sequence is distinct from stellar MS
def is_substellar_cmd(M_J, J_K, age_est='field'):
    if age_est == 'field':
        # Field brown dwarf sequence
        return M_J > 10 + 2 * J_K  # Rough cut
```"""
        },
        {
            "user": "Query Gaia for ultracool dwarf candidates",
            "assistant": """Here's how to find ultracool dwarf candidates in Gaia DR3:

```python
from astroquery.gaia import Gaia
import numpy as np

# Ultracool dwarfs (>M7) are red and faint
# Gaia detection limit means mostly nearby objects

# Method 1: Color and absolute magnitude cuts
query = \"\"\"
SELECT source_id, ra, dec, parallax, parallax_error,
       phot_g_mean_mag, bp_rp, pmra, pmdec
FROM gaiadr3.gaia_source
WHERE bp_rp > 2.5                          -- Very red
  AND parallax > 20                         -- Within 50 pc
  AND parallax_over_error > 10              -- Good parallax
  AND phot_g_mean_mag + 5*LOG10(parallax/100) > 12  -- Faint absolute mag
  AND ruwe < 1.4                            -- Good astrometry
\"\"\"

job = Gaia.launch_job_async(query)
ucd_candidates = job.get_results()

# Calculate absolute magnitude
ucd_candidates['M_G'] = (ucd_candidates['phot_g_mean_mag'] + 
                         5 * np.log10(ucd_candidates['parallax'] / 100))

# Further refine with tangential velocity
ucd_candidates['v_tan'] = (4.74 * 
    np.sqrt(ucd_candidates['pmra']**2 + ucd_candidates['pmdec']**2) * 
    1000 / ucd_candidates['parallax'])

# Young UCDs have low velocities
young_candidates = ucd_candidates[ucd_candidates['v_tan'] < 30]

print(f"Found {len(ucd_candidates)} UCD candidates")
print(f"Of which {len(young_candidates)} may be young")

# Method 2: Cross-match with known UCD catalogs
# Use SIMBAD or dedicated catalogs like the L and T dwarf archive

from astroquery.simbad import Simbad

Simbad.add_votable_fields('sptype', 'parallax')
result = Simbad.query_criteria('sptype >= L0 & sptype <= Y2')
```

For a complete sample, also cross-match with:
- 2MASS (J-K colors)
- WISE (W1-W2 colors for T/Y dwarfs)
- PS1 (i-z, z-y colors)"""
        },
        {
            "user": "What are the key WISE colors for brown dwarf classification?",
            "assistant": """WISE colors are excellent for identifying and classifying cool brown dwarfs:

**Color Diagnostics:**

| Type | W1-W2 | W2-W3 | Notes |
|------|-------|-------|-------|
| M/L transition | ~0.2-0.3 | ~0.3-0.5 | Similar to M dwarfs |
| Mid-L | 0.3-0.5 | 0.3-0.6 | Dust reddening |
| Late-L/Early-T | 0.5-1.0 | 0.3-0.5 | Transition zone |
| Mid-T | 1.0-2.0 | 0.2-0.5 | CH4 absorption in W1 |
| Late-T | 2.0-3.0 | 0.0-0.3 | Strong CH4 |
| Y dwarfs | >2.5 | variable | Very red W1-W2 |

```python
from astroquery.ipac.irsa import Irsa
import numpy as np

def classify_by_wise_color(w1, w2, w3=None):
    '''Rough spectral type estimate from WISE colors.'''
    w1_w2 = w1 - w2
    
    if w1_w2 < 0.4:
        return 'M/early-L'
    elif w1_w2 < 0.8:
        return 'L dwarf'
    elif w1_w2 < 1.5:
        return 'T0-T5'
    elif w1_w2 < 2.5:
        return 'T5-T9'
    else:
        return 'Y dwarf candidate'

# Query WISE for brown dwarf candidates
def find_brown_dwarfs_wise(ra, dec, radius_deg=1.0):
    '''Search for T/Y dwarf candidates in WISE.'''
    
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    coords = SkyCoord(ra=ra, dec=dec, unit='deg')
    
    # Query AllWISE
    result = Irsa.query_region(
        coords, 
        catalog='allwise_p3as_psd',
        radius=radius_deg * u.deg
    )
    
    # T/Y dwarf selection
    w1_w2 = result['w1mpro'] - result['w2mpro']
    
    # T dwarf candidates: W1-W2 > 0.8, good photometry
    t_dwarf_mask = (
        (w1_w2 > 0.8) & 
        (result['w1sigmpro'] < 0.1) &
        (result['w2sigmpro'] < 0.1) &
        (result['cc_flags'] == '0000')  # No contamination
    )
    
    # Y dwarf candidates: W1-W2 > 2.0 or W1 dropout
    y_dwarf_mask = (
        (w1_w2 > 2.0) | 
        ((result['w1mpro'] > 17) & (result['w2mpro'] < 15))
    )
    
    return {
        't_candidates': result[t_dwarf_mask],
        'y_candidates': result[y_dwarf_mask & ~t_dwarf_mask]
    }
```

**Key notes:**
- W1 (3.4 μm) is affected by CH4 at ~3.3 μm
- W2 (4.6 μm) is in a "window" - less affected
- Very red W1-W2 is the hallmark of T and Y dwarfs
- W3/W4 often undetected for faint brown dwarfs"""
        },
        {
            "user": "How do I estimate the mass of a brown dwarf?",
            "assistant": """Brown dwarf mass estimation requires evolutionary models and known parameters:

**Required parameters:**
- Luminosity (from parallax + photometry)
- Age (hardest to determine)
- Optionally: Teff from spectroscopy

**Methods:**

```python
import numpy as np
from scipy.interpolate import interp1d

def estimate_bd_mass(L_bol, age_Gyr, model='ATMO2020'):
    '''
    Estimate brown dwarf mass using evolutionary models.
    
    Parameters:
    - L_bol: Bolometric luminosity (L_sun)
    - age_Gyr: Age in Gyr
    - model: 'ATMO2020', 'Sonora', 'COND', 'DUSTY'
    
    Returns:
    - mass in Jupiter masses
    '''
    
    # Example ATMO2020 grid (simplified)
    # Real implementation would load full model grid
    
    if age_Gyr < 0.1:
        # Young: L and mass nearly independent
        log_L = np.log10(L_bol)
        mass = 10 ** (1.5 * log_L + 4.0)  # Rough scaling
    else:
        # Field age: steeper relation
        log_L = np.log10(L_bol)
        mass = 10 ** (0.8 * log_L + 3.5)
    
    return mass  # Jupiter masses

def luminosity_from_photometry(M_J, BC_J=-2.0):
    '''
    Calculate bolometric luminosity from absolute J magnitude.
    
    BC_J: Bolometric correction (depends on SpT)
    '''
    M_bol_sun = 4.74
    M_bol = M_J + BC_J
    L_bol = 10 ** ((M_bol_sun - M_bol) / 2.5)
    return L_bol  # in L_sun

# Bolometric corrections by spectral type (approximate)
BC_J_by_spt = {
    'L0': 1.5, 'L5': 1.8, 
    'T0': 2.0, 'T5': 2.5, 'T8': 2.8,
    'Y0': 3.0
}

# Example: T5 dwarf at 10 pc with J=15
M_J = 15 - 5 * np.log10(10/10)  # = 15
BC_J = BC_J_by_spt['T5']
L_bol = luminosity_from_photometry(M_J, BC_J)
print(f"L_bol = {L_bol:.2e} L_sun")

# Estimate mass for field age (~5 Gyr)
mass = estimate_bd_mass(L_bol, age_Gyr=5.0)
print(f"Estimated mass: {mass:.0f} M_Jup")
```

**Key uncertainties:**
1. **Age**: Largest uncertainty. Field brown dwarfs have wide age range.
2. **Metallicity**: Affects cooling rate
3. **Clouds**: Dusty vs. clear atmospheres
4. **Model choice**: ATMO, Sonora, COND give different results

**Best cases for mass determination:**
- Binaries (dynamical masses)
- Members of clusters/associations (known age)
- Benchmark systems (independent age constraints)"""
        },
        # Euclid-specific examples
        {
            "user": "What data products will Euclid provide?",
            "assistant": """Euclid will deliver multiple data products for cosmology and astronomy:

**Primary Instruments:**

1. **VIS (Visible Instrument)**
   - Single broad band (550-900 nm, R+I+Z)
   - 0.1 arcsec/pixel
   - PSF FWHM ~0.16 arcsec
   - Depth: ~24.5 mag (10σ extended source)

2. **NISP (Near Infrared Spectrometer and Photometer)**
   - Photometry: Y, J, H bands (0.95-2.0 μm)
   - 0.3 arcsec/pixel
   - Depth: ~24 mag (5σ point source)
   - Spectroscopy: 1.25-1.85 μm slitless (R~450)

**Survey Parameters:**
- Wide Survey: ~14,000 deg² (extragalactic sky)
- Deep Survey: ~40 deg² (3 fields, 2 mag deeper)

**Data Products:**

```python
# Euclid data access (when available)
# Expected through ESA archives and Euclid Science Archive

# Key catalogs:
euclid_products = {
    'photometry': {
        'bands': ['VIS', 'Y', 'J', 'H'],
        'columns': ['ra', 'dec', 'flux_vis', 'flux_y', 'flux_j', 'flux_h',
                   'flux_err_*', 'mag_*', 'kron_radius', 'class_star']
    },
    'photometric_redshifts': {
        'columns': ['z_phot', 'z_phot_err', 'pdf_z', 'odds']
    },
    'spectroscopic_redshifts': {
        'columns': ['z_spec', 'z_spec_err', 'quality_flag']
    },
    'morphology': {
        'columns': ['sersic_n', 'r_eff', 'ellipticity', 'bulge_fraction']
    },
    'weak_lensing': {
        'columns': ['e1', 'e2', 'weight', 'psf_e1', 'psf_e2']
    }
}

# Cross-matching with Gaia
# Euclid will include Gaia cross-matches for:
# - Proper motion cleaning
# - Star/galaxy separation
# - Astrometric calibration
```

**Science Applications Beyond Cosmology:**
- Galaxy evolution to z~2
- Stellar populations in nearby galaxies
- Brown dwarf and exoplanet searches
- Strong gravitational lenses
- High-z quasars
- Solar system objects

**Timeline:**
- Launch: July 2023
- Early Release Observations: 2024
- DR1: ~2026 (10% of survey)
- Final release: ~2030"""
        },
        {
            "user": "How can I use Euclid data for brown dwarf science?",
            "assistant": """Euclid's NIR imaging will be valuable for brown dwarf science:

**Euclid for Brown Dwarfs:**

1. **Y, J, H Photometry**
   - Sensitive to ~24 mag in J
   - Can detect T dwarfs to ~100 pc
   - Y dwarfs to ~20-30 pc

2. **Proper Motion from Multi-epoch Data**
   - Wide survey visits each field multiple times
   - Deep fields have many epochs
   - Can measure proper motions for faint brown dwarfs

3. **Color Selection**

```python
import numpy as np

def euclid_bd_selection(Y, J, H, Y_err, J_err, H_err):
    '''
    Select brown dwarf candidates from Euclid photometry.
    '''
    # Colors
    Y_J = Y - J
    J_H = J - H
    
    # T dwarf selection (CH4 absorption affects J)
    t_dwarf_mask = (
        (Y_J > 0.5) &           # Red Y-J
        (J_H < 0.3) &           # Blue J-H (CH4 in H)
        (J < 23) &              # Reasonable S/N
        (J_err < 0.1) &
        (H_err < 0.1)
    )
    
    # Late-T/Y selection (even redder)
    late_t_y_mask = (
        (Y_J > 1.0) &           # Very red Y-J
        (J > 20) &              # Faint
        (Y_err < 0.2)
    )
    
    # L dwarf selection
    l_dwarf_mask = (
        (Y_J > 0.3) & (Y_J < 0.8) &
        (J_H > 0.3) & (J_H < 0.7)
    )
    
    return {
        't_dwarfs': t_dwarf_mask,
        'late_t_y': late_t_y_mask,
        'l_dwarfs': l_dwarf_mask
    }

# Expected yields (Wide Survey)
# - L dwarfs: ~100,000 (many already known)
# - T dwarfs: ~10,000 (many new discoveries)
# - Y dwarfs: ~100-1000 (significant new sample)
```

**Cross-matching Strategy:**

```python
def cross_match_euclid_gaia_wise(euclid_cat):
    '''
    Cross-match Euclid brown dwarf candidates with other surveys.
    '''
    # 1. Gaia - proper motion and parallax
    # Brown dwarfs with Gaia parallax: distance + absolute mag
    
    # 2. WISE - W1, W2 colors
    # Confirm T/Y classification with W1-W2
    
    # 3. Ground-based spectroscopy
    # Confirm spectral types
    
    pass

# Key science cases:
# 1. Complete local census to 50 pc
# 2. Substellar luminosity function
# 3. Binary fraction at low masses
# 4. Young brown dwarfs in associations
# 5. Ultracool subdwarfs (low metallicity)
```

**Advantages of Euclid:**
- Uniform photometry over large area
- Space-based PSF (no seeing variations)
- Simultaneous Y, J, H
- Cross-calibration with VIS for star/galaxy separation"""
        },
    ]
    
    # Save Q&A pairs
    qa_dir = output_dir / "qa_substellar"
    qa_dir.mkdir(parents=True, exist_ok=True)
    
    qa_file = qa_dir / "substellar_qa.jsonl"
    with open(qa_file, "w") as f:
        for qa in qa_pairs:
            entry = {
                "messages": [
                    {"role": "system", "content": "You are AstroLlama, an expert in substellar astronomy including brown dwarfs, exoplanets, and ultracool objects."},
                    {"role": "user", "content": qa["user"]},
                    {"role": "assistant", "content": qa["assistant"]}
                ]
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"Created {len(qa_pairs)} substellar Q&A pairs")
    return len(qa_pairs)


# =============================================================================
# PDF Text Extraction (if available)
# =============================================================================

def extract_text_from_pdfs(pdf_dir: Path, output_dir: Path) -> int:
    """Extract text from downloaded PDFs."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not available. Install with: pip install PyMuPDF")
        return 0
    
    text_dir = output_dir / "extracted_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_file)
            text = ""
            for page in doc:
                text += page.get_text()
            
            text_file = text_dir / f"{pdf_file.stem}.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            count += 1
        except Exception as e:
            print(f"Error extracting {pdf_file.name}: {e}")
    
    return count


# =============================================================================
# Training Data Compilation
# =============================================================================

def compile_training_data(raw_dir: Path, output_dir: Path) -> Dict:
    """Compile all harvested data into training format."""
    
    training_examples = []
    
    # Load all JSONL files
    for jsonl_file in raw_dir.rglob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # Convert to training format if needed
                    if "messages" in entry:
                        training_examples.append(entry)
                    elif "text" in entry:
                        # Create Q&A from abstract
                        text = entry["text"]
                        if "Title:" in text and "Abstract:" in text:
                            parts = text.split("Abstract:")
                            title = parts[0].replace("Title:", "").strip()
                            abstract = parts[1].strip() if len(parts) > 1 else ""
                            
                            # Generate Q&A
                            example = {
                                "messages": [
                                    {"role": "system", "content": "You are AstroLlama, an expert astronomy research assistant."},
                                    {"role": "user", "content": f"Summarize the paper: {title}"},
                                    {"role": "assistant", "content": abstract}
                                ]
                            }
                            training_examples.append(example)
                except:
                    continue
    
    # Save compiled data
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    import random
    random.shuffle(training_examples)
    
    split_idx = int(len(training_examples) * 0.9)
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    train_file = output_dir / "train.jsonl"
    with open(train_file, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    
    val_file = output_dir / "validation.jsonl"
    with open(val_file, "w") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")
    
    return {
        "total_examples": len(training_examples),
        "train": len(train_data),
        "validation": len(val_data),
        "train_file": str(train_file),
        "val_file": str(val_file)
    }


# =============================================================================
# Main Harvesting Pipeline
# =============================================================================

def run_full_harvest(output_dir: Path, topics: List[str], 
                     max_papers_per_topic: int = 500,
                     download_pdfs: bool = False,
                     max_size_gb: float = 10.0) -> Dict:
    """Run the full data harvesting pipeline."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "topics": {},
        "total_papers": 0,
        "total_pdfs": 0,
        "total_size_mb": 0
    }
    
    print("=" * 60)
    print("AstroLlama Data Harvester")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Topics: {', '.join(topics)}")
    print(f"Max papers per topic: {max_papers_per_topic}")
    print(f"Download PDFs: {download_pdfs}")
    print("=" * 60)
    
    # Harvest arXiv papers for each topic
    for topic in topics:
        print(f"\n{'='*60}")
        print(f"Topic: {topic.upper()}")
        print(f"{'='*60}")
        
        arxiv_result = harvest_arxiv_by_topic(
            topic, 
            output_dir / "arxiv", 
            max_papers=max_papers_per_topic,
            download_pdfs=download_pdfs
        )
        
        # Also harvest from ADS
        ads_result = harvest_ads_reviews(
            topic,
            output_dir / "ads",
            max_papers=100
        )
        
        results["topics"][topic] = {
            "arxiv": arxiv_result,
            "ads": ads_result
        }
        results["total_papers"] += arxiv_result.get("papers_found", 0)
        results["total_papers"] += ads_result.get("papers_found", 0)
        results["total_pdfs"] += arxiv_result.get("pdfs_downloaded", 0)
    
    # Create specialized Q&A data
    print(f"\n{'='*60}")
    print("Creating specialized Q&A data...")
    print(f"{'='*60}")
    
    qa_count = create_substellar_qa_data(output_dir)
    
    # Harvest survey schemas
    print("\nHarvesting survey documentation...")
    schema_result = harvest_survey_schemas(output_dir)
    
    # Compile all training data
    print(f"\n{'='*60}")
    print("Compiling training data...")
    print(f"{'='*60}")
    
    compile_result = compile_training_data(output_dir, output_dir / "compiled")
    
    # Calculate total size
    total_size = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    )
    results["total_size_mb"] = total_size / (1024 * 1024)
    
    # Summary
    print(f"\n{'='*60}")
    print("HARVEST COMPLETE")
    print(f"{'='*60}")
    print(f"Total papers harvested: {results['total_papers']}")
    print(f"PDFs downloaded: {results['total_pdfs']}")
    print(f"Total size: {results['total_size_mb']:.1f} MB")
    print(f"Training examples: {compile_result['total_examples']}")
    print(f"  - Train: {compile_result['train']}")
    print(f"  - Validation: {compile_result['validation']}")
    print(f"\nOutput files:")
    print(f"  {compile_result['train_file']}")
    print(f"  {compile_result['val_file']}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harvest astronomy training data from multiple sources"
    )
    parser.add_argument("--output", "-o", default="./data/raw/",
                        help="Output directory")
    parser.add_argument("--topics", "-t", 
                        default="substellar,exoplanets,euclid,jwst,gaia",
                        help="Comma-separated topics")
    parser.add_argument("--papers", "-p", type=int, default=500,
                        help="Max papers per topic")
    parser.add_argument("--pdfs", action="store_true",
                        help="Download PDFs (requires more storage)")
    parser.add_argument("--max-size", type=float, default=10.0,
                        help="Maximum total size in GB")
    
    args = parser.parse_args()
    
    topics = [t.strip() for t in args.topics.split(",")]
    output_dir = Path(args.output)
    
    # Check for required packages
    if not ARXIV_AVAILABLE:
        print("Installing arxiv package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "arxiv"], 
                      check=True)
        import arxiv
    
    results = run_full_harvest(
        output_dir=output_dir,
        topics=topics,
        max_papers_per_topic=args.papers,
        download_pdfs=args.pdfs,
        max_size_gb=args.max_size
    )
    
    # Save results
    with open(output_dir / "harvest_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nNext steps:")
    print(f"1. Review harvested data in {output_dir}")
    print(f"2. Combine with generated Q&A data:")
    print(f"   cat {output_dir}/compiled/train.jsonl ./data/training/train.jsonl > ./data/training/combined_train.jsonl")
    print(f"3. Upload to S3 and start fine-tuning")


if __name__ == "__main__":
    main()
