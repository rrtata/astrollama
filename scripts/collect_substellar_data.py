#!/usr/bin/env python3
"""
AstroLlama - Comprehensive Substellar Astronomy Data Collector
Downloads papers, documentation, and training material for fine-tuning.

Focus areas:
- Brown dwarfs (L, T, Y dwarfs)
- Exoplanets and atmospheres
- Euclid space telescope
- JWST observations
- Cool star/substellar photometry
- Spectral classification

Usage:
    python collect_substellar_data.py --output ./data/collected/ --max-papers 5000
    python collect_substellar_data.py --categories brown_dwarfs,exoplanets,euclid
    python collect_substellar_data.py --download-all
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("./data/collected/")
MAX_PAPERS_PER_CATEGORY = 1000
ARXIV_DELAY = 3  # Seconds between arXiv requests (be nice to servers)

# Search queries for different topics
SEARCH_QUERIES = {
    # Brown Dwarfs
    "brown_dwarfs": [
        "brown dwarf spectroscopy",
        "L dwarf classification",
        "T dwarf atmosphere",
        "Y dwarf discovery",
        "ultracool dwarf",
        "brown dwarf binary",
        "brown dwarf metallicity",
        "substellar mass function",
        "brown dwarf cooling",
        "brown dwarf variability",
    ],
    
    # Exoplanets
    "exoplanets": [
        "exoplanet atmosphere spectroscopy",
        "hot Jupiter atmosphere",
        "super Earth characterization",
        "exoplanet transit spectroscopy",
        "direct imaging exoplanet",
        "exoplanet mass radius",
        "habitable zone planet",
        "exoplanet formation",
        "protoplanetary disk",
        "planet metallicity correlation",
    ],
    
    # Euclid Space Telescope
    "euclid": [
        "Euclid space telescope",
        "Euclid survey",
        "Euclid photometry",
        "Euclid weak lensing",
        "Euclid galaxy clustering",
        "Euclid cosmology",
        "Euclid data processing",
        "Euclid calibration",
        "Euclid VIS instrument",
        "Euclid NISP instrument",
    ],
    
    # JWST
    "jwst": [
        "JWST spectroscopy",
        "JWST NIRSpec",
        "JWST MIRI",
        "JWST exoplanet",
        "JWST brown dwarf",
        "JWST high redshift",
        "JWST imaging",
        "JWST calibration",
        "JWST early release",
        "Webb telescope observations",
    ],
    
    # Cool Stars
    "cool_stars": [
        "M dwarf spectroscopy",
        "M dwarf activity",
        "M dwarf metallicity",
        "cool star photometry",
        "late-type star",
        "red dwarf characterization",
        "stellar effective temperature",
        "cool star atmosphere",
        "low mass star",
        "M dwarf planet",
    ],
    
    # Spectral Classification
    "spectral_classification": [
        "stellar spectral classification",
        "spectral type determination",
        "stellar parameter estimation",
        "machine learning spectroscopy",
        "spectroscopic survey",
        "LAMOST stellar parameters",
        "APOGEE spectroscopy",
        "GALAH survey",
        "spectral template",
        "spectral indices",
    ],
    
    # Photometric Surveys
    "photometric_surveys": [
        "Gaia photometry brown dwarf",
        "2MASS ultracool",
        "WISE brown dwarf",
        "Pan-STARRS variability",
        "ZTF transient",
        "SDSS spectroscopy",
        "UKIDSS survey",
        "VHS infrared survey",
        "photometric calibration",
        "color magnitude diagram",
    ],
    
    # Atmosphere Models
    "atmosphere_models": [
        "stellar atmosphere model",
        "PHOENIX atmosphere",
        "BT-Settl model",
        "MARCS atmosphere",
        "brown dwarf atmosphere model",
        "exoplanet atmosphere model",
        "cloud formation substellar",
        "opacity calculation",
        "radiative transfer",
        "chemical equilibrium atmosphere",
    ],
}

# Key review papers and important references (bibcodes)
KEY_PAPERS = [
    # Brown Dwarf Reviews
    "2014prpl.conf..619C",  # Chabrier+ - Brown dwarf formation
    "2005ARA&A..43..195K",  # Kirkpatrick - Brown dwarfs review
    "2021ApJ...918...11L",  # Late T/Y dwarf review
    
    # Exoplanet Reviews  
    "2018haex.bookE.100M",  # Madhusudhan - Exoplanet atmospheres
    "2016SSRv..205..285M",  # Marley & Robinson - Brown dwarf/exoplanet atmospheres
    "2021JGRE..12606639F",  # Fortney+ - Giant planet atmospheres
    
    # Euclid
    "2022A&A...662A.112E",  # Euclid mission paper
    "2024A&A...685A...1E",  # Euclid early release
    
    # JWST
    "2023PASP..135d8001R",  # JWST overview
    "2023Natur.614..649J",  # JWST exoplanet results
    
    # Gaia
    "2023A&A...674A...1G",  # Gaia DR3
    "2021A&A...649A...1G",  # Gaia EDR3
]

# =============================================================================
# arXiv Downloader
# =============================================================================

def search_arxiv(query: str, max_results: int = 100, start: int = 0) -> List[Dict]:
    """Search arXiv for papers matching query."""
    
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": start,
        "max_results": min(max_results, 100),  # arXiv limit
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    
    # Filter to astro-ph
    params["search_query"] = f"cat:astro-ph* AND ({params['search_query']})"
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
    except Exception as e:
        print(f"  Error querying arXiv: {e}")
        return []
    
    # Parse XML
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    
    papers = []
    for entry in root.findall("atom:entry", ns):
        try:
            paper = {
                "arxiv_id": entry.find("atom:id", ns).text.split("/")[-1],
                "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                "abstract": entry.find("atom:summary", ns).text.strip().replace("\n", " "),
                "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)],
                "published": entry.find("atom:published", ns).text[:10],
                "categories": [c.get("term") for c in entry.findall("atom:category", ns)],
                "pdf_url": None,
            }
            
            # Get PDF link
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    paper["pdf_url"] = link.get("href")
                    break
            
            papers.append(paper)
        except Exception as e:
            continue
    
    return papers


def download_arxiv_pdf(arxiv_id: str, output_dir: Path) -> Optional[str]:
    """Download PDF from arXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_file = output_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    
    if output_file.exists():
        return str(output_file)
    
    try:
        urllib.request.urlretrieve(pdf_url, output_file)
        return str(output_file)
    except Exception as e:
        print(f"  Failed to download {arxiv_id}: {e}")
        return None


def collect_arxiv_papers(category: str, queries: List[str], output_dir: Path, 
                         max_papers: int = 500) -> List[Dict]:
    """Collect papers from arXiv for a category."""
    
    category_dir = output_dir / "arxiv" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    all_papers = []
    seen_ids = set()
    
    for query in queries:
        print(f"  Searching: {query}")
        
        papers_per_query = max_papers // len(queries)
        collected = 0
        start = 0
        
        while collected < papers_per_query:
            papers = search_arxiv(query, max_results=100, start=start)
            
            if not papers:
                break
            
            for paper in papers:
                if paper["arxiv_id"] not in seen_ids:
                    seen_ids.add(paper["arxiv_id"])
                    all_papers.append(paper)
                    collected += 1
            
            start += 100
            time.sleep(ARXIV_DELAY)
            
            if start > 500:  # Don't go too deep
                break
    
    # Save metadata
    metadata_file = category_dir / "papers_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(all_papers, f, indent=2)
    
    print(f"  Collected {len(all_papers)} unique papers for {category}")
    
    return all_papers


def download_pdfs_parallel(papers: List[Dict], output_dir: Path, max_downloads: int = 100):
    """Download PDFs in parallel."""
    
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit downloads
    to_download = papers[:max_downloads]
    
    print(f"  Downloading {len(to_download)} PDFs...")
    
    downloaded = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(download_arxiv_pdf, p["arxiv_id"], pdf_dir): p
            for p in to_download
        }
        
        for future in as_completed(futures):
            paper = futures[future]
            result = future.result()
            if result:
                downloaded += 1
            
            if downloaded % 10 == 0:
                print(f"    Downloaded {downloaded}/{len(to_download)}")
            
            time.sleep(1)  # Be nice to arXiv
    
    print(f"  Downloaded {downloaded} PDFs")


# =============================================================================
# NASA ADS Integration
# =============================================================================

def get_ads_token() -> Optional[str]:
    """Get ADS token from environment or secrets."""
    token = os.environ.get("ADS_TOKEN")
    if token:
        return token
    
    # Try AWS Secrets Manager
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name="us-west-2")
        response = client.get_secret_value(SecretId="astrollama/api-keys")
        secrets = json.loads(response["SecretString"])
        return secrets.get("ADS_TOKEN")
    except:
        return None


def search_ads(query: str, token: str, rows: int = 100) -> List[Dict]:
    """Search NASA ADS for papers."""
    
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    params = {
        "q": query,
        "rows": rows,
        "fl": "bibcode,title,abstract,author,year,citation_count,pub",
        "sort": "citation_count desc"
    }
    
    try:
        req = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}",
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  ADS error: {e}")
        return []


def get_paper_by_bibcode(bibcode: str, token: str) -> Optional[Dict]:
    """Get paper details by bibcode."""
    
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    params = {
        "q": f"bibcode:{bibcode}",
        "fl": "bibcode,title,abstract,author,year,citation_count,pub,keyword"
    }
    
    try:
        req = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}",
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            docs = data.get("response", {}).get("docs", [])
            return docs[0] if docs else None
    except Exception as e:
        print(f"  ADS error for {bibcode}: {e}")
        return None


def collect_ads_papers(queries: Dict[str, List[str]], output_dir: Path, 
                       token: str, max_per_query: int = 100) -> List[Dict]:
    """Collect papers from ADS."""
    
    ads_dir = output_dir / "ads"
    ads_dir.mkdir(parents=True, exist_ok=True)
    
    all_papers = []
    seen_bibcodes = set()
    
    for category, query_list in queries.items():
        print(f"\nCollecting ADS papers for {category}...")
        
        for query in query_list:
            print(f"  Query: {query}")
            papers = search_ads(query, token, rows=max_per_query)
            
            for paper in papers:
                if paper.get("bibcode") not in seen_bibcodes:
                    seen_bibcodes.add(paper["bibcode"])
                    paper["category"] = category
                    all_papers.append(paper)
            
            time.sleep(1)  # Rate limit
    
    # Get key papers
    print("\nCollecting key reference papers...")
    for bibcode in KEY_PAPERS:
        if bibcode not in seen_bibcodes:
            paper = get_paper_by_bibcode(bibcode, token)
            if paper:
                paper["category"] = "key_references"
                all_papers.append(paper)
                seen_bibcodes.add(bibcode)
        time.sleep(0.5)
    
    # Save
    with open(ads_dir / "ads_papers.json", "w") as f:
        json.dump(all_papers, f, indent=2)
    
    print(f"\nCollected {len(all_papers)} papers from ADS")
    return all_papers


# =============================================================================
# Documentation Collectors
# =============================================================================

def collect_euclid_documentation(output_dir: Path):
    """Download Euclid public documentation."""
    
    euclid_dir = output_dir / "documentation" / "euclid"
    euclid_dir.mkdir(parents=True, exist_ok=True)
    
    # Euclid public documentation URLs
    euclid_docs = [
        ("https://www.euclid-ec.org/public/mission/", "mission_overview.html"),
        ("https://euclid.esac.esa.int/dr/eucliddr1_vega_a_202411/documents/", "data_release_notes.html"),
    ]
    
    # Science papers from arXiv
    euclid_arxiv_queries = [
        "Euclid Collaboration",
        "Euclid survey design",
        "Euclid data processing",
        "Euclid science performance",
    ]
    
    print("Collecting Euclid documentation...")
    papers = []
    
    for query in euclid_arxiv_queries:
        results = search_arxiv(query, max_results=50)
        papers.extend(results)
        time.sleep(ARXIV_DELAY)
    
    # Save Euclid papers
    with open(euclid_dir / "euclid_papers.json", "w") as f:
        json.dump(papers, f, indent=2)
    
    print(f"  Collected {len(papers)} Euclid papers")
    
    # Create Euclid knowledge base content
    euclid_kb = """# Euclid Space Telescope Knowledge Base

## Mission Overview
- **Launch**: July 1, 2023
- **Orbit**: Sun-Earth L2 Lagrange point
- **Mission Duration**: 6 years (nominal)
- **Survey Area**: ~14,000 deg² (Wide Survey) + ~40 deg² (Deep Survey)

## Instruments

### VIS (Visible Instrument)
- Wavelength: 550-900 nm (single broad band)
- Field of View: 0.54 deg²
- Pixel scale: 0.1 arcsec/pixel
- Purpose: Shape measurements for weak lensing

### NISP (Near-Infrared Spectrometer and Photometer)
- Photometry: Y, J, H bands (0.95-2.02 μm)
- Spectroscopy: Red grism (1.25-1.85 μm), Blue grism (0.92-1.25 μm)
- Field of View: 0.53 deg²
- Spectral resolution: R~380 (slitless)

## Data Products
- Images in VIS and NISP bands
- Source catalogs with photometry
- Redshifts (photometric and spectroscopic)
- Weak lensing shear catalogs
- Galaxy clustering catalogs

## Key Science Goals
1. Dark energy equation of state w(z)
2. Modified gravity tests
3. Neutrino masses
4. Galaxy formation and evolution
5. Milky Way science
6. Solar system objects

## Data Access
- ESA Euclid Archive: https://easc-esac.esa.int/
- Public releases at regular intervals
- Data format: FITS files following Euclid data model
"""
    
    with open(euclid_dir / "euclid_knowledge_base.md", "w") as f:
        f.write(euclid_kb)
    
    return papers


def collect_jwst_documentation(output_dir: Path):
    """Collect JWST documentation and papers."""
    
    jwst_dir = output_dir / "documentation" / "jwst"
    jwst_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting JWST documentation...")
    
    # JWST papers from arXiv
    jwst_queries = [
        "JWST NIRSpec brown dwarf",
        "JWST MIRI exoplanet",
        "JWST spectroscopy atmosphere",
        "JWST early release science",
        "JWST calibration",
    ]
    
    papers = []
    for query in jwst_queries:
        results = search_arxiv(query, max_results=100)
        papers.extend(results)
        time.sleep(ARXIV_DELAY)
    
    # Save
    with open(jwst_dir / "jwst_papers.json", "w") as f:
        json.dump(papers, f, indent=2)
    
    # JWST knowledge base
    jwst_kb = """# JWST Knowledge Base for Substellar Astronomy

## JWST Instruments Relevant to Substellar Objects

### NIRSpec (Near-Infrared Spectrograph)
- Wavelength: 0.6-5.3 μm
- Modes: Multi-object (MSA), IFU, Fixed slits
- Spectral resolution: R~100 (prism), R~1000 (gratings), R~2700 (high-res)
- Key for: Brown dwarf spectroscopy, exoplanet atmospheres

### MIRI (Mid-Infrared Instrument)
- Wavelength: 5-28 μm
- Imaging + Low/Medium Resolution Spectroscopy
- Key for: Cool brown dwarfs, debris disks, thermal emission

### NIRCam (Near-Infrared Camera)
- Wavelength: 0.6-5 μm
- Coronagraphic imaging available
- Key for: Direct imaging of planets, brown dwarf companions

## Brown Dwarf Observations with JWST
- Atmospheric characterization (molecules: H2O, CH4, NH3, CO, CO2)
- Cloud properties and variability
- Vertical mixing and disequilibrium chemistry
- Y dwarf discoveries and characterization

## Exoplanet Observations
- Transmission spectroscopy during transit
- Emission spectroscopy during secondary eclipse
- Phase curves for temperature mapping
- Direct imaging and spectroscopy

## Data Products
- Level 3: Calibrated, combined data products
- Spectral data in FITS format
- Pipeline: jwst Python package

## Access
- MAST Archive: https://mast.stsci.edu/
- JWST User Documentation: https://jwst-docs.stsci.edu/
"""
    
    with open(jwst_dir / "jwst_knowledge_base.md", "w") as f:
        f.write(jwst_kb)
    
    print(f"  Collected {len(papers)} JWST papers")
    return papers


def create_substellar_knowledge_base(output_dir: Path):
    """Create comprehensive substellar astronomy knowledge base."""
    
    kb_dir = output_dir / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    
    # Brown Dwarf Knowledge Base
    brown_dwarf_kb = """# Brown Dwarf Knowledge Base

## Spectral Classification

### L Dwarfs (Teff ~ 2200-1400 K)
- Key features: TiO/VO disappear, metal hydrides (FeH, CrH)
- Dust/cloud formation important
- Optical: deep H2O, weak TiO
- NIR: H2O bands at 1.4, 1.9 μm; CO at 2.3 μm

### T Dwarfs (Teff ~ 1400-500 K)
- Key features: CH4 absorption dominates NIR
- Clouds sink below photosphere
- Strong H2O and CH4 bands
- Pressure-broadened K I lines in optical

### Y Dwarfs (Teff < 500 K)
- Coldest brown dwarfs known
- NH3 absorption appears
- H2O clouds possible
- Detected mainly by WISE

## Color-Magnitude Relations

### Gaia Colors
- L dwarfs: BP-RP ~ 4.0-5.0
- T dwarfs: BP-RP > 5 (very red, often undetected in BP)
- Use G-RP or absolute G magnitude

### Infrared Colors
- J-K color: L dwarfs red (J-K~1.5-2), T dwarfs blue (J-K~0)
- W1-W2: Good discriminator for T/Y dwarfs
- CH4 on/off indices for T dwarf classification

## Queries for Brown Dwarfs

### Gaia Query for L/T Dwarf Candidates
```sql
SELECT source_id, ra, dec, phot_g_mean_mag, phot_rp_mean_mag,
       parallax, pmra, pmdec
FROM gaiadr3.gaia_source
WHERE parallax > 50  -- within 20 pc
  AND phot_rp_mean_mag - phot_g_mean_mag > 2.5
  AND phot_g_mean_mag > 18
  AND parallax_over_error > 5
```

### WISE Query for T/Y Dwarfs
- W1-W2 > 0.4 for late-T
- W1-W2 > 0.8 for Y dwarfs
- Cross-match with 2MASS non-detections

## Physical Properties

### Mass-Luminosity Relation
- Hydrogen burning limit: ~75 MJup
- Deuterium burning limit: ~13 MJup
- Brown dwarfs continuously cool with age

### Atmospheric Properties
- Clouds: Fe, Mg-silicates (L dwarfs), sulfides (T dwarfs)
- Chemistry: CO/CH4 transition around L/T boundary
- Mixing length and vertical transport important

## Key Catalogs
- DwarfArchives.org
- UltracoolSheet (J. Gagné)
- Gaia Ultracool Dwarf Sample
- BANYAN Σ (young moving groups)
"""
    
    with open(kb_dir / "brown_dwarf_knowledge.md", "w") as f:
        f.write(brown_dwarf_kb)
    
    # Exoplanet Atmospheres Knowledge Base
    exoplanet_kb = """# Exoplanet Atmosphere Knowledge Base

## Detection Methods

### Transmission Spectroscopy
- During primary transit
- Probes limb/terminator region
- Transit depth varies with wavelength: δ(λ) ∝ Rp(λ)²/R*²
- Sensitive to: upper atmosphere, clouds/hazes, composition

### Emission Spectroscopy
- During secondary eclipse
- Probes dayside atmosphere
- Eclipse depth: (Fp/F*)(λ)
- Sensitive to: thermal structure, composition, clouds

### Phase Curves
- Continuous monitoring over orbit
- Maps temperature vs longitude
- Day-night contrast, hotspot offset

## Key Molecules

### Hydrogen-Dominated Atmospheres
- H2O: 1.4, 1.9, 2.7 μm bands
- CH4: 1.6, 2.3, 3.3 μm (cooler planets)
- CO: 2.3, 4.7 μm (hotter planets)
- CO2: 4.3, 15 μm
- NH3: 1.5, 2.0, 3.0 μm
- HCN: 3.0, 7, 14 μm
- Na/K: 589/769 nm doublets

### High Mean Molecular Weight
- H2O, CO2 dominated (super-Earths)
- O3, O2 (biosignatures)
- N2O (biosignature)

## Temperature-Pressure Profiles

### Types
- Isothermal (simplest assumption)
- Radiative equilibrium
- Inversions (TiO/VO in hot Jupiters)

### Key Pressures
- Transit photosphere: ~1-100 mbar
- Emission photosphere: ~0.1-1 bar

## Cloud/Haze Properties
- Mie scattering for particles
- Rayleigh scattering slope (λ^-4)
- Gray cloud deck
- Photochemical hazes (tholins)

## Retrieval Codes
- NEMESIS
- petitRADTRANS
- PLATON
- Pyrat Bay
- CHIMERA
- POSEIDON

## JWST Observations
- Transmission: NIRSpec G395M/H, NIRISS SOSS
- Emission: MIRI LRS, NIRSpec
- Typical precision: 10-50 ppm
"""
    
    with open(kb_dir / "exoplanet_atmosphere_knowledge.md", "w") as f:
        f.write(exoplanet_kb)
    
    # Spectroscopic Analysis Knowledge Base
    spectroscopy_kb = """# Spectroscopic Analysis Knowledge Base

## Spectral Classification System

### MK System (Main Sequence)
- O B A F G K M (hot to cool)
- Subclasses 0-9 (e.g., G2, K5)
- Luminosity class: I-V (supergiant to dwarf)

### Extended System (Ultracool)
- L0-L9: Dust/metal hydrides
- T0-T9: Methane dominated
- Y: Coldest, ammonia

## Spectral Analysis Techniques

### Template Matching
```python
from PyAstronomy import pyasl
import numpy as np

def match_template(observed_wave, observed_flux, template_wave, template_flux):
    # Resample template to observed wavelength grid
    template_resampled = np.interp(observed_wave, template_wave, template_flux)
    
    # Compute chi-squared
    chi2 = np.sum((observed_flux - template_resampled)**2)
    return chi2
```

### Equivalent Width Measurement
```python
def equivalent_width(wavelength, flux, line_center, line_width, continuum_regions):
    # Define continuum
    cont_mask = (wavelength > continuum_regions[0]) & (wavelength < continuum_regions[1])
    continuum = np.median(flux[cont_mask])
    
    # Line region
    line_mask = np.abs(wavelength - line_center) < line_width
    
    # EW = integral of (1 - F/Fc) dλ
    ew = np.trapz(1 - flux[line_mask]/continuum, wavelength[line_mask])
    return ew
```

### Spectral Indices
- TiO5, CaH2, CaH3 for M dwarfs
- H2O-J, H2O-H for L dwarfs
- CH4-J, CH4-H for T dwarfs

## Radial Velocity Measurement

### Cross-Correlation Method
```python
from scipy.signal import correlate

def measure_rv(observed, template, wavelength):
    # Cross-correlate
    ccf = correlate(observed, template, mode='same')
    
    # Find peak
    peak_idx = np.argmax(ccf)
    
    # Convert to velocity
    c = 299792.458  # km/s
    delta_lambda = wavelength[peak_idx] - wavelength[len(wavelength)//2]
    rv = c * delta_lambda / np.mean(wavelength)
    return rv
```

## Stellar Parameter Determination

### Methods
1. **Photometric**: Color-Teff relations
2. **Spectroscopic**: Line depth ratios, EWs
3. **Model Fitting**: Full spectrum fitting
4. **Machine Learning**: Data-driven approaches

### Key Parameters
- Teff: Effective temperature
- log g: Surface gravity
- [Fe/H]: Metallicity
- [α/Fe]: Alpha enhancement
- v sin i: Projected rotation

## Survey Spectroscopy

### LAMOST
- R~1800
- 3700-9000 Å
- >10 million spectra

### APOGEE
- R~22,500
- H-band (1.5-1.7 μm)
- High-precision abundances

### GALAH
- R~28,000
- 4 optical bands
- Detailed abundances
"""
    
    with open(kb_dir / "spectroscopy_knowledge.md", "w") as f:
        f.write(spectroscopy_kb)
    
    print(f"Created knowledge base files in {kb_dir}")


# =============================================================================
# Training Data Generation from Collected Papers
# =============================================================================

def generate_training_from_papers(papers: List[Dict], output_dir: Path) -> List[Dict]:
    """Generate training examples from collected papers."""
    
    training_examples = []
    
    for paper in papers:
        if not paper.get("abstract"):
            continue
        
        # Generate Q&A from abstract
        title = paper.get("title", ["Unknown"])[0] if isinstance(paper.get("title"), list) else paper.get("title", "Unknown")
        abstract = paper.get("abstract", "")
        authors = paper.get("author", paper.get("authors", []))
        if isinstance(authors, list):
            first_author = authors[0] if authors else "Unknown"
        else:
            first_author = authors
        
        # Question 1: Paper summary
        training_examples.append({
            "messages": [
                {"role": "system", "content": "You are AstroLlama, an expert astronomy research assistant specializing in substellar objects, brown dwarfs, and exoplanets."},
                {"role": "user", "content": f"Summarize the key findings of the paper titled '{title}'"},
                {"role": "assistant", "content": f"The paper by {first_author} et al., titled '{title}', presents the following key findings:\n\n{abstract}"}
            ]
        })
        
        # Question 2: Related question based on topic
        category = paper.get("category", "general")
        
        if "brown dwarf" in title.lower() or "brown dwarf" in abstract.lower():
            training_examples.append({
                "messages": [
                    {"role": "system", "content": "You are AstroLlama, an expert astronomy research assistant."},
                    {"role": "user", "content": "What are the main observational signatures that distinguish brown dwarfs from low-mass stars and giant planets?"},
                    {"role": "assistant", "content": """Brown dwarfs occupy a unique position between stars and planets, with several distinguishing characteristics:

**Spectral Features:**
- L dwarfs: Metal hydrides (FeH, CrH), dust/clouds, weakening TiO/VO
- T dwarfs: Strong methane (CH4) absorption at 1.6 and 2.2 μm
- Y dwarfs: Ammonia (NH3) features, extreme infrared colors

**Photometric Properties:**
- Very red optical-to-NIR colors (L dwarfs: J-K ~ 1.5-2.0)
- T dwarfs show "blue" J-K due to CH4 absorption
- Faint absolute magnitudes (M_J > 12 for late-T)

**Physical Properties:**
- Mass: 13-75 Jupiter masses
- Temperature: <2500 K (cooler than M dwarfs)
- No sustained hydrogen fusion (distinguishes from stars)
- Continuous cooling with age

**Key Diagnostics:**
```python
# Query for brown dwarf candidates in Gaia
query = '''
SELECT source_id, phot_g_mean_mag, bp_rp, parallax
FROM gaiadr3.gaia_source
WHERE bp_rp > 4.0  -- Very red
AND phot_g_mean_mag + 5*log10(parallax/100) > 14  -- Faint absolute mag
AND parallax_over_error > 5
'''
```"""
                }
            ]})
        
        elif "exoplanet" in title.lower() or "atmosphere" in abstract.lower():
            training_examples.append({
                "messages": [
                    {"role": "system", "content": "You are AstroLlama, an expert astronomy research assistant."},
                    {"role": "user", "content": "How do I analyze exoplanet transmission spectra to detect atmospheric molecules?"},
                    {"role": "assistant", "content": """Transmission spectroscopy analysis involves several key steps:

**1. Data Reduction:**
```python
import numpy as np
from scipy.ndimage import median_filter

# Remove systematics
def detrend_lightcurve(flux, time, model='linear'):
    # Fit baseline trend
    coeffs = np.polyfit(time, flux, 1)
    trend = np.polyval(coeffs, time)
    return flux / trend
```

**2. Transit Depth Measurement:**
```python
def measure_transit_depth(wavelength, flux_in_transit, flux_out_transit):
    # Rp/Rs as function of wavelength
    transit_depth = 1 - flux_in_transit / flux_out_transit
    rp_rs = np.sqrt(transit_depth)
    return wavelength, rp_rs
```

**3. Atmospheric Retrieval:**
- Compare to models (PHOENIX, petitRADTRANS)
- Key molecules: H2O (1.4, 1.9 μm), CH4 (2.3 μm), CO (4.6 μm), CO2 (4.3 μm)
- Look for Rayleigh scattering slope in optical

**4. Interpretation:**
- Enhanced absorption → molecule present
- Flat spectrum → clouds/hazes masking features
- Scattering slope → small particles in upper atmosphere

JWST has revolutionized this field with ~10 ppm precision, enabling detection of CO2, SO2, and other trace species."""
                }
            ]})
    
    return training_examples


# =============================================================================
# Main Collection Pipeline
# =============================================================================

def collect_all_data(output_dir: Path, max_papers_per_category: int = 500,
                     download_pdfs: bool = False, categories: List[str] = None):
    """Run the full data collection pipeline."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AstroLlama Substellar Astronomy Data Collection")
    print("=" * 60)
    
    all_papers = []
    
    # Filter categories if specified
    queries = SEARCH_QUERIES
    if categories:
        queries = {k: v for k, v in SEARCH_QUERIES.items() if k in categories}
    
    # Collect from arXiv
    print("\n[1/5] Collecting papers from arXiv...")
    for category, query_list in queries.items():
        print(f"\nCategory: {category}")
        papers = collect_arxiv_papers(category, query_list, output_dir, 
                                     max_papers=max_papers_per_category)
        all_papers.extend(papers)
    
    # Collect from ADS (if token available)
    ads_token = get_ads_token()
    if ads_token:
        print("\n[2/5] Collecting papers from NASA ADS...")
        ads_papers = collect_ads_papers(queries, output_dir, ads_token,
                                       max_per_query=50)
        # Convert to common format
        for p in ads_papers:
            all_papers.append({
                "arxiv_id": p.get("bibcode"),
                "title": p.get("title", [""])[0] if isinstance(p.get("title"), list) else p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "authors": p.get("author", []),
                "published": str(p.get("year", "")),
                "category": p.get("category", ""),
            })
    else:
        print("\n[2/5] Skipping ADS (no token). Set ADS_TOKEN env var for more papers.")
    
    # Collect documentation
    print("\n[3/5] Collecting Euclid documentation...")
    collect_euclid_documentation(output_dir)
    
    print("\n[4/5] Collecting JWST documentation...")
    collect_jwst_documentation(output_dir)
    
    # Create knowledge bases
    print("\n[5/5] Creating knowledge base files...")
    create_substellar_knowledge_base(output_dir)
    
    # Download PDFs if requested
    if download_pdfs:
        print("\n[BONUS] Downloading PDFs...")
        download_pdfs_parallel(all_papers, output_dir, max_downloads=200)
    
    # Generate training data from collected papers
    print("\n[GENERATING] Creating training examples from collected data...")
    training_examples = generate_training_from_papers(all_papers, output_dir)
    
    # Save training data
    training_dir = output_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    with open(training_dir / "substellar_training.jsonl", "w") as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total papers collected: {len(all_papers)}")
    print(f"Training examples generated: {len(training_examples)}")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    for f in output_dir.rglob("*"):
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"  {f.relative_to(output_dir)}: {size:.1f} KB")
    
    return all_papers, training_examples


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect substellar astronomy data for training")
    parser.add_argument("--output", "-o", default="./data/collected/",
                        help="Output directory")
    parser.add_argument("--max-papers", "-n", type=int, default=500,
                        help="Max papers per category")
    parser.add_argument("--categories", "-c", type=str, default=None,
                        help="Comma-separated categories (default: all)")
    parser.add_argument("--download-pdfs", action="store_true",
                        help="Download PDFs (slow, large storage)")
    parser.add_argument("--download-all", action="store_true",
                        help="Download everything including PDFs")
    
    args = parser.parse_args()
    
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    
    download_pdfs = args.download_pdfs or args.download_all
    
    collect_all_data(
        output_dir=Path(args.output),
        max_papers_per_category=args.max_papers,
        download_pdfs=download_pdfs,
        categories=categories
    )


if __name__ == "__main__":
    main()
