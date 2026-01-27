#!/usr/bin/env python3
"""
AstroLlama RAG Setup with Pinecone
Ingest astronomy papers, documentation, and textbooks into vector database.

Usage:
    python scripts/setup_rag.py setup              # Initialize Pinecone
    python scripts/setup_rag.py ingest-ads         # Ingest papers from ADS
    python scripts/setup_rag.py ingest-docs        # Ingest survey documentation
    python scripts/setup_rag.py ingest-dir ./pdfs  # Ingest local PDFs
    python scripts/setup_rag.py test "brown dwarf" # Test retrieval
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import urllib.request
import urllib.parse

# =============================================================================
# Configuration
# =============================================================================

PINECONE_INDEX_NAME = "astrollama-rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50

# Topics to ingest from ADS
ADS_TOPICS = [
    ("brown dwarf spectroscopy", 200),
    ("T dwarf discovery", 100),
    ("Y dwarf", 100),
    ("ultracool dwarf", 150),
    ("L dwarf classification", 100),
    ("brown dwarf atmosphere", 150),
    ("substellar binary", 100),
    ("WISE brown dwarf", 100),
    ("Gaia ultracool", 100),
    ("JWST brown dwarf", 50),
    ("Euclid brown dwarf", 50),
]

# Documentation URLs to ingest
DOCUMENTATION_SOURCES = {
    "gaia_dr3": "https://gea.esac.esa.int/archive/documentation/GDR3/",
    "2mass": "https://irsa.ipac.caltech.edu/data/2MASS/docs/",
    "wise": "https://wise2.ipac.caltech.edu/docs/release/allwise/",
    "sdss": "https://www.sdss.org/dr18/",
    "euclid": "https://www.euclid-ec.org/",
    "jwst": "https://jwst-docs.stsci.edu/",
}


# =============================================================================
# Embedding Model
# =============================================================================

class EmbeddingModel:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                print(f"✓ Loaded embedding model: {self.model_name}")
            except ImportError:
                print("Installing sentence-transformers...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"])
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode([text])[0].tolist()


# =============================================================================
# Text Chunker
# =============================================================================

@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


class TextChunker:
    """Split text into chunks for embedding."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into overlapping chunks."""
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        start = 0
        chunk_num = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            
            # Create unique chunk ID
            chunk_id = hashlib.md5(
                f"{metadata.get('source', 'unknown')}_{chunk_num}_{chunk_text[:50]}".encode()
            ).hexdigest()[:16]
            
            chunk_metadata = {
                **metadata,
                "chunk_num": chunk_num,
                "start_word": start,
                "end_word": end,
            }
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))
            
            start = end - self.overlap
            chunk_num += 1
            
            if end >= len(words):
                break
        
        return chunks


# =============================================================================
# Pinecone Client
# =============================================================================

class PineconeRAG:
    """RAG system using Pinecone vector database."""
    
    def __init__(self, api_key: str = None, index_name: str = PINECONE_INDEX_NAME):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index_name = index_name
        self._client = None
        self._index = None
        self.embedder = EmbeddingModel()
        self.chunker = TextChunker()
    
    def _get_api_key(self) -> str:
        """Get Pinecone API key from environment or Secrets Manager."""
        if self.api_key:
            return self.api_key
        
        # Try Secrets Manager
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name="us-west-2")
            response = client.get_secret_value(SecretId="astrollama/api-keys")
            secrets = json.loads(response["SecretString"])
            return secrets.get("PINECONE_API_KEY", "")
        except:
            return ""
    
    @property
    def client(self):
        if self._client is None:
            try:
                from pinecone import Pinecone
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("PINECONE_API_KEY not set")
                self._client = Pinecone(api_key=api_key)
                print("✓ Connected to Pinecone")
            except ImportError:
                print("Installing pinecone-client...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "pinecone-client"])
                from pinecone import Pinecone
                self._client = Pinecone(api_key=self._get_api_key())
        return self._client
    
    @property
    def index(self):
        if self._index is None:
            self._index = self.client.Index(self.index_name)
        return self._index
    
    def setup_index(self):
        """Create Pinecone index if it doesn't exist."""
        from pinecone import ServerlessSpec
        
        existing_indexes = [idx.name for idx in self.client.list_indexes()]
        
        if self.index_name in existing_indexes:
            print(f"✓ Index '{self.index_name}' already exists")
            stats = self.index.describe_index_stats()
            print(f"  Vectors: {stats.total_vector_count}")
            return
        
        print(f"Creating index '{self.index_name}'...")
        self.client.create_index(
            name=self.index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free tier region
            )
        )
        print(f"✓ Created index '{self.index_name}'")
    
    def ingest_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Ingest documents into Pinecone.
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
            batch_size: Number of vectors to upsert at once
        """
        all_chunks = []
        
        # Chunk all documents
        print("Chunking documents...")
        for doc in documents:
            chunks = self.chunker.chunk_text(doc["text"], doc.get("metadata", {}))
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Generate embeddings and upsert in batches
        print("Generating embeddings and upserting...")
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            # Generate embeddings
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed(texts)
            
            # Prepare vectors
            vectors = []
            for chunk, embedding in zip(batch, embeddings):
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.text[:1000]  # Store truncated text in metadata
                    }
                })
            
            # Upsert
            self.index.upsert(vectors=vectors)
            print(f"  Upserted {i + len(batch)}/{len(all_chunks)} vectors")
        
        print(f"✓ Ingested {len(all_chunks)} chunks")
    
    def query(self, query: str, top_k: int = 5, filter: Dict = None) -> List[Dict]:
        """Query the RAG system.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter: Metadata filter
        
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Format results
        formatted = []
        for match in results.matches:
            formatted.append({
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            })
        
        return formatted
    
    def delete_all(self):
        """Delete all vectors from the index."""
        self.index.delete(delete_all=True)
        print("✓ Deleted all vectors")


# =============================================================================
# ADS Paper Ingestion
# =============================================================================

class ADSIngester:
    """Ingest papers from NASA ADS."""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.environ.get("ADS_TOKEN")
        self.base_url = "https://api.adsabs.harvard.edu/v1"
    
    def _get_token(self) -> str:
        if self.api_token:
            return self.api_token
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name="us-west-2")
            response = client.get_secret_value(SecretId="astrollama/api-keys")
            secrets = json.loads(response["SecretString"])
            return secrets.get("ADS_TOKEN", "")
        except:
            return ""
    
    def search_papers(self, query: str, rows: int = 100) -> List[Dict]:
        """Search ADS for papers."""
        token = self._get_token()
        if not token:
            print("Warning: ADS_TOKEN not set")
            return []
        
        url = f"{self.base_url}/search/query"
        params = {
            "q": query,
            "rows": rows,
            "fl": "bibcode,title,author,year,abstract,citation_count,pub,keyword",
            "sort": "citation_count desc"
        }
        
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get("response", {}).get("docs", [])
        except Exception as e:
            print(f"ADS search error: {e}")
            return []
    
    def papers_to_documents(self, papers: List[Dict]) -> List[Dict]:
        """Convert ADS papers to document format for ingestion."""
        documents = []
        
        for paper in papers:
            # Combine title and abstract
            title = paper.get("title", [""])[0] if isinstance(paper.get("title"), list) else paper.get("title", "")
            abstract = paper.get("abstract", "")
            
            if not abstract:
                continue
            
            text = f"Title: {title}\n\nAbstract: {abstract}"
            
            # Add keywords if available
            keywords = paper.get("keyword", [])
            if keywords:
                text += f"\n\nKeywords: {', '.join(keywords[:10])}"
            
            documents.append({
                "text": text,
                "metadata": {
                    "source": "ads",
                    "bibcode": paper.get("bibcode", ""),
                    "title": title[:200],
                    "year": paper.get("year", ""),
                    "authors": ", ".join(paper.get("author", [])[:3]),
                    "citations": paper.get("citation_count", 0),
                    "publication": paper.get("pub", "")[:100]
                }
            })
        
        return documents
    
    def ingest_topics(self, rag: PineconeRAG, topics: List[tuple] = None):
        """Ingest papers from multiple topics."""
        topics = topics or ADS_TOPICS
        all_documents = []
        seen_bibcodes = set()
        
        for query, num_papers in topics:
            print(f"\nSearching: '{query}' ({num_papers} papers)...")
            papers = self.search_papers(query, rows=num_papers)
            
            # Deduplicate
            new_papers = []
            for p in papers:
                bibcode = p.get("bibcode", "")
                if bibcode and bibcode not in seen_bibcodes:
                    seen_bibcodes.add(bibcode)
                    new_papers.append(p)
            
            documents = self.papers_to_documents(new_papers)
            all_documents.extend(documents)
            print(f"  Found {len(documents)} new papers")
        
        print(f"\nTotal unique papers: {len(all_documents)}")
        
        if all_documents:
            print("\nIngesting into Pinecone...")
            rag.ingest_documents(all_documents)


# =============================================================================
# Documentation Ingestion
# =============================================================================

# Pre-written documentation content (since we can't easily scrape)
SURVEY_DOCUMENTATION = {
    "gaia_dr3": """
Gaia Data Release 3 (DR3) Documentation

Gaia DR3 contains astrometry, photometry, and spectroscopy for nearly 2 billion sources.

Key Tables:
- gaia_source: Main source catalog with positions, proper motions, parallaxes
- astrophysical_parameters: Stellar parameters (Teff, logg, [Fe/H])
- xp_sampled_mean_spectrum: XP spectra

Useful Columns for Brown Dwarf Research:
- ra, dec: Position (ICRS, epoch 2016.0)
- parallax, parallax_error: Parallax in mas
- pmra, pmdec: Proper motion in mas/yr
- phot_g_mean_mag: G-band magnitude
- phot_bp_mean_mag, phot_rp_mean_mag: BP and RP magnitudes
- bp_rp: BP-RP color

Brown Dwarf Selection Criteria:
- High proper motion: sqrt(pmra^2 + pmdec^2) > 100 mas/yr
- Red colors: bp_rp > 2.5
- Faint absolute magnitude: G + 5*log10(parallax/100) > 12

Example ADQL Query for Brown Dwarf Candidates:
SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE parallax > 20 
  AND parallax_over_error > 10
  AND bp_rp > 2.5
  AND phot_g_mean_mag + 5*log10(parallax/100) > 12
""",
    
    "2mass": """
2MASS (Two Micron All Sky Survey) Documentation

2MASS provides JHKs photometry for 470 million point sources.

Key Columns:
- ra, dec: Position (J2000)
- j_m, h_m, k_m: JHKs magnitudes
- j_msigcom, h_msigcom, k_msigcom: Photometric uncertainties
- ph_qual: Photometric quality flags (AAA = best)
- rd_flg: Read flag
- cc_flg: Contamination/confusion flag

Color Selection for Ultracool Dwarfs:
- Late M dwarfs: J-H > 0.5, H-K > 0.2
- L dwarfs: J-H > 0.6, H-K > 0.3
- T dwarfs: J-H < 0.3 (blue J-H), H-K > 0.0

Quality Cuts:
- ph_qual = 'AAA' for best photometry
- rd_flg = '222' for optimal aperture
- cc_flg = '000' for no contamination

VizieR Catalog ID: II/246/out
""",
    
    "wise": """
WISE/AllWISE Documentation

AllWISE provides W1 (3.4μm), W2 (4.6μm), W3 (12μm), W4 (22μm) photometry.

Key Columns:
- ra, dec: Position (J2000)
- w1mpro, w2mpro, w3mpro, w4mpro: Profile-fit magnitudes
- w1sigmpro, etc.: Uncertainties
- cc_flags: Contamination flags
- ext_flg: Extended source flag
- ph_qual: Photometric quality

Color Selection for Brown Dwarfs:
- Late L dwarfs: W1-W2 > 0.4
- T dwarfs: W1-W2 > 0.8 (methane absorption)
- Y dwarfs: W1-W2 > 2.0

Key Diagnostics:
- W1-W2 color is diagnostic of methane absorption
- W2-W3 can indicate disk presence
- J-W2 excellent for finding cold brown dwarfs

T Dwarf Selection:
SELECT * FROM allwise
WHERE w1mpro - w2mpro > 0.8 
  AND ph_qual LIKE 'A%'
  AND cc_flags = '0000'

Y Dwarf Selection:
- Extremely red W1-W2 > 2.0
- Often only detected in W2
- Cross-match with 2MASS (J-band) helpful
""",
    
    "spectral_classification": """
Brown Dwarf Spectral Classification

The MKK system extended to L, T, and Y spectral types:

L Dwarfs (Teff ~2200-1300 K):
- Dust clouds form in atmosphere
- Strong metal hydrides: FeH, CrH
- Weakening TiO, VO bands
- Na I, K I resonance lines strengthen
- H2O bands present
- Subtypes: L0-L9

T Dwarfs (Teff ~1300-500 K):
- Methane (CH4) absorption appears
- Dust clouds sink below photosphere
- Strong H2O absorption
- Collision-induced H2 absorption
- Blue J-H color due to CH4
- Subtypes: T0-T9

Y Dwarfs (Teff < 500 K):
- Ammonia (NH3) absorption
- Extremely red W1-W2 colors
- Water clouds may form
- Coldest known: ~250 K
- Subtypes: Y0-Y2+ (still being defined)

Key Spectral Indices:
- H2O-J index: Water at 1.15 μm
- CH4-J index: Methane at 1.31 μm
- CH4-H index: Methane at 1.6 μm
- NH3-H index: Ammonia in H-band
- W_J index: J-band width

Classification Resources:
- SpeX Prism Library: http://pono.ucsd.edu/~adam/browndwarfs/spexprism/
- Montreal Spectral Library
- DwarfArchives.org
""",
    
    "euclid": """
Euclid Mission Documentation

Euclid is an ESA mission for dark energy and dark matter studies.

Instruments:
- VIS: Visible imaging (550-900 nm, 0.1" resolution)
- NISP: Near-IR imaging (Y, J, H bands) and spectroscopy

Brown Dwarf Science with Euclid:
- Wide survey: 15,000 deg² to Y~24, J~24, H~24
- Proper motions from multi-epoch data
- Spectroscopy for selected targets

Expected Brown Dwarf Yields:
- ~10,000 new L dwarfs
- ~1,000 new T dwarfs
- ~100 new Y dwarf candidates
- Improved parallaxes for nearby brown dwarfs

Data Products:
- Stacked images
- Source catalogs
- Photometric redshifts
- Spectroscopic redshifts

Color Selection for Brown Dwarfs:
- Euclid Y-J, J-H colors complement ground-based surveys
- Cross-match with Gaia for astrometry
- Cross-match with WISE for mid-IR
""",
    
    "jwst": """
JWST Brown Dwarf Observations

JWST Instruments for Brown Dwarf Science:
- NIRCam: 0.6-5 μm imaging
- NIRSpec: 0.6-5.3 μm spectroscopy (R~100-2700)
- MIRI: 5-28 μm imaging and spectroscopy

Key Capabilities:
- Unprecedented sensitivity in thermal IR
- High-resolution spectroscopy of molecular features
- Direct imaging of wide companions

Science Cases:
1. Atmospheric characterization:
   - CH4, H2O, CO, CO2, NH3 abundances
   - Cloud properties
   - Vertical mixing (disequilibrium chemistry)

2. Y dwarf spectroscopy:
   - First detailed spectra of coldest brown dwarfs
   - Ammonia features
   - Water ice signatures

3. Companion detection:
   - Cold planetary-mass companions
   - Brown dwarf binaries at close separations

4. Benchmark systems:
   - Brown dwarfs with measured masses
   - Calibration of evolutionary models

Key Programs:
- ERS 1386: Brown Dwarf Atmospheres
- GO programs targeting specific Y dwarfs
- Exoplanet transit spectroscopy (overlapping science)

Data Access:
- MAST archive: https://mast.stsci.edu/
- astroquery.mast module
"""
}


def ingest_documentation(rag: PineconeRAG):
    """Ingest survey documentation."""
    documents = []
    
    for source, content in SURVEY_DOCUMENTATION.items():
        # Split by sections
        sections = content.strip().split("\n\n")
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:
                continue
            
            documents.append({
                "text": section,
                "metadata": {
                    "source": "documentation",
                    "survey": source,
                    "section": i
                }
            })
    
    print(f"Ingesting {len(documents)} documentation sections...")
    rag.ingest_documents(documents)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AstroLlama RAG Setup")
    parser.add_argument("command", choices=["setup", "ingest-ads", "ingest-docs", "ingest-dir", "test", "stats", "clear"])
    parser.add_argument("arg", nargs="?", help="Query for test, or directory for ingest-dir")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results for test")
    
    args = parser.parse_args()
    
    rag = PineconeRAG()
    
    if args.command == "setup":
        print("=" * 60)
        print("Setting up Pinecone RAG")
        print("=" * 60)
        rag.setup_index()
        
    elif args.command == "ingest-ads":
        print("=" * 60)
        print("Ingesting papers from ADS")
        print("=" * 60)
        ingester = ADSIngester()
        ingester.ingest_topics(rag)
        
    elif args.command == "ingest-docs":
        print("=" * 60)
        print("Ingesting survey documentation")
        print("=" * 60)
        ingest_documentation(rag)
        
    elif args.command == "ingest-dir":
        if not args.arg:
            print("Error: Please provide a directory path")
            sys.exit(1)
        print("=" * 60)
        print(f"Ingesting PDFs from {args.arg}")
        print("=" * 60)
        print("PDF ingestion not yet implemented - coming soon!")
        
    elif args.command == "test":
        query = args.arg or "How do I identify T dwarfs using WISE colors?"
        print("=" * 60)
        print(f"Testing RAG query: {query}")
        print("=" * 60)
        
        results = rag.query(query, top_k=args.top_k)
        
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (score: {result['score']:.3f}) ---")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            if result['metadata'].get('title'):
                print(f"Title: {result['metadata']['title']}")
            print(f"Text: {result['text'][:300]}...")
            
    elif args.command == "stats":
        print("=" * 60)
        print("Pinecone Index Stats")
        print("=" * 60)
        stats = rag.index.describe_index_stats()
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Dimension: {stats.dimension}")
        
    elif args.command == "clear":
        print("=" * 60)
        print("Clearing all vectors")
        print("=" * 60)
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() == "yes":
            rag.delete_all()
        else:
            print("Cancelled")


if __name__ == "__main__":
    main()
