"""
AstroLlama Lambda Functions
Agent action handlers for astronomy tools.

Deploy with:
    cd lambda/
    zip -r function.zip .
    aws lambda create-function --function-name astrollama-tools \
        --runtime python3.11 --handler handler.lambda_handler \
        --role $LAMBDA_ROLE --zip-file fileb://function.zip
"""

import json
import os
import boto3
from typing import Dict, Any
import urllib.request
import urllib.parse


# =============================================================================
# Configuration
# =============================================================================

ADS_TOKEN = None  # Will be loaded from Secrets Manager
PINECONE_API_KEY = None  # Will be loaded from Secrets Manager
BUCKET = os.environ.get("ASTROLLAMA_BUCKET", "")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "astrollama-knowledge")


def get_secrets():
    """Get API keys from environment variables or Secrets Manager."""
    global ADS_TOKEN, PINECONE_API_KEY
    
    if ADS_TOKEN and PINECONE_API_KEY:
        return {"ads": ADS_TOKEN, "pinecone": PINECONE_API_KEY}
    
    # First try environment variables (simpler, no IAM needed)
    ADS_TOKEN = os.environ.get("ADS_TOKEN", "")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
    
    if ADS_TOKEN and PINECONE_API_KEY:
        return {"ads": ADS_TOKEN, "pinecone": PINECONE_API_KEY}
    
    # Fall back to Secrets Manager
    client = boto3.client("secretsmanager")
    try:
        response = client.get_secret_value(SecretId="astrollama/api-keys")
        secrets = json.loads(response["SecretString"])
        ADS_TOKEN = ADS_TOKEN or secrets.get("ADS_TOKEN", "")
        PINECONE_API_KEY = PINECONE_API_KEY or secrets.get("PINECONE_API_KEY", "")
        return {"ads": ADS_TOKEN, "pinecone": PINECONE_API_KEY}
    except Exception as e:
        print(f"Note: Could not get secrets from Secrets Manager: {e}")
        return {"ads": ADS_TOKEN, "pinecone": PINECONE_API_KEY}


def get_ads_token():
    """Get ADS token from Secrets Manager."""
    secrets = get_secrets()
    return secrets.get("ads", "")


# =============================================================================
# Tool Implementations
# =============================================================================

def resolve_object(name: str) -> Dict[str, Any]:
    """Resolve astronomical object name to coordinates using SIMBAD."""
    
    # SIMBAD TAP query
    base_url = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
    
    query = f"""
    SELECT TOP 1 main_id, ra, dec, otype_txt
    FROM basic
    WHERE main_id = '{name}' OR ident.id = '{name}'
    """
    
    params = {
        "request": "doQuery",
        "lang": "ADQL",
        "format": "json",
        "query": query.strip()
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            
            if data.get("data") and len(data["data"]) > 0:
                row = data["data"][0]
                return {
                    "status": "success",
                    "object_name": name,
                    "simbad_id": row[0],
                    "ra_deg": row[1],
                    "dec_deg": row[2],
                    "object_type": row[3],
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"Object '{name}' not found in SIMBAD"
                }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def query_gaia(ra: float, dec: float, radius_arcmin: float = 5.0, limit: int = 100) -> Dict[str, Any]:
    """Query Gaia DR3 catalog."""
    
    base_url = "https://gea.esac.esa.int/tap-server/tap/sync"
    
    radius_deg = radius_arcmin / 60.0
    
    query = f"""
    SELECT TOP {limit}
        source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    ) = 1
    AND phot_g_mean_mag IS NOT NULL
    ORDER BY phot_g_mean_mag
    """
    
    params = {
        "request": "doQuery",
        "lang": "ADQL",
        "format": "json",
        "query": query.strip()
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
            
            columns = [col["name"] for col in data.get("metadata", [])]
            rows = data.get("data", [])
            
            # Convert to list of dicts
            sources = []
            for row in rows[:20]:  # Return first 20 for summary
                source = dict(zip(columns, row))
                sources.append(source)
            
            return {
                "status": "success",
                "catalog": "Gaia DR3",
                "query_position": {"ra": ra, "dec": dec},
                "radius_arcmin": radius_arcmin,
                "total_sources": len(rows),
                "sources_sample": sources,
                "columns": columns,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def search_ads(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Search NASA ADS for papers."""
    
    token = get_ads_token()
    if not token:
        return {
            "status": "error",
            "message": "ADS token not configured"
        }
    
    base_url = "https://api.adsabs.harvard.edu/v1/search/query"
    
    params = {
        "q": query,
        "fl": "bibcode,title,author,year,pub,citation_count,abstract",
        "rows": max_results,
        "sort": "citation_count desc"
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            
            docs = data.get("response", {}).get("docs", [])
            
            papers = []
            for doc in docs:
                authors = doc.get("author", [])
                if len(authors) > 3:
                    author_str = f"{authors[0]} et al."
                else:
                    author_str = "; ".join(authors[:3])
                
                papers.append({
                    "bibcode": doc.get("bibcode"),
                    "title": doc.get("title", [""])[0],
                    "authors": author_str,
                    "year": doc.get("year"),
                    "publication": doc.get("pub"),
                    "citations": doc.get("citation_count", 0),
                })
            
            return {
                "status": "success",
                "query": query,
                "num_results": len(papers),
                "papers": papers
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def get_bibtex(bibcodes: list) -> Dict[str, Any]:
    """Get BibTeX for papers."""
    
    token = get_ads_token()
    if not token:
        return {
            "status": "error",
            "message": "ADS token not configured"
        }
    
    url = "https://api.adsabs.harvard.edu/v1/export/bibtex"
    
    data = json.dumps({"bibcode": bibcodes}).encode()
    
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            
            return {
                "status": "success",
                "bibcodes": bibcodes,
                "bibtex": result.get("export", "")
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def compute_color_cut_stats(
    color_min: float,
    color_max: float,
    mag_min: float = None,
    mag_max: float = None,
    ra: float = None,
    dec: float = None,
    radius_arcmin: float = 10.0
) -> Dict[str, Any]:
    """
    Query Gaia and compute statistics for a color-magnitude selection.
    """
    
    if ra is None or dec is None:
        return {
            "status": "error",
            "message": "RA and Dec required"
        }
    
    # Query Gaia
    gaia_result = query_gaia(ra, dec, radius_arcmin, limit=10000)
    
    if gaia_result["status"] != "success":
        return gaia_result
    
    # Apply color cut (would need numpy in real implementation)
    # This is a simplified version
    total = gaia_result["total_sources"]
    
    # Estimate based on typical CMD distributions
    color_width = color_max - color_min
    estimated_fraction = min(color_width / 3.0, 1.0)  # Rough estimate
    
    return {
        "status": "success",
        "selection_criteria": {
            "color_range": [color_min, color_max],
            "mag_range": [mag_min, mag_max] if mag_min else None,
        },
        "query_position": {"ra": ra, "dec": dec},
        "total_sources_in_field": total,
        "estimated_selected": int(total * estimated_fraction),
        "note": "For precise counts, download full catalog and apply cuts locally"
    }


def retrieve_knowledge(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve relevant documents from Pinecone knowledge base.
    Requires pinecone-client and sentence-transformers in Lambda layer.
    """
    try:
        # Get Pinecone API key
        secrets = get_secrets()
        api_key = secrets.get("pinecone", "")
        
        if not api_key:
            return {
                "status": "error",
                "message": "Pinecone API key not configured"
            }
        
        # Initialize Pinecone (requires pinecone-client in layer)
        # This is a simplified version - full implementation would use embeddings
        
        # For now, return instructions for local RAG
        return {
            "status": "info",
            "message": "RAG retrieval is handled locally via setup_pinecone_rag.py",
            "suggestion": "Use: python scripts/setup_pinecone_rag.py test --query '{}'".format(query),
            "note": "To enable Lambda-based RAG, add pinecone-client and sentence-transformers to a Lambda layer"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def generate_plot_instructions(
    plot_type: str,
    data_source: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate instructions for creating astronomy plots.
    Returns Python code that can be executed locally.
    """
    
    plot_templates = {
        "cmd": '''
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

# Query Gaia for {data_source}
query = """
SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
AND phot_g_mean_mag < 20 AND bp_rp IS NOT NULL
"""
job = Gaia.launch_job(query)
data = job.get_results()

# Create CMD
fig, ax = plt.subplots(figsize=(8, 10))
ax.scatter(data['bp_rp'], data['phot_g_mean_mag'], s=1, alpha=0.5)
ax.set_xlabel('BP - RP [mag]')
ax.set_ylabel('G [mag]')
ax.set_title('CMD: {data_source}')
ax.invert_yaxis()
plt.savefig('cmd_{data_source}.png', dpi=150)
plt.show()
''',
        "sky_map": '''
import matplotlib.pyplot as plt
import numpy as np

# Your data should have ra, dec columns
ra_rad = np.deg2rad(data['ra'])
ra_rad[ra_rad > np.pi] -= 2*np.pi
dec_rad = np.deg2rad(data['dec'])

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='aitoff')
ax.scatter(ra_rad, dec_rad, s=1, alpha=0.5)
ax.grid(True)
plt.savefig('sky_map.png', dpi=150)
''',
        "lightcurve": '''
import matplotlib.pyplot as plt

# Your data should have time, mag, mag_err columns
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(data['time'], data['mag'], yerr=data['mag_err'], fmt='o', ms=3)
ax.set_xlabel('Time [MJD]')
ax.set_ylabel('Magnitude')
ax.invert_yaxis()
plt.savefig('lightcurve.png', dpi=150)
'''
    }
    
    template = plot_templates.get(plot_type, plot_templates["cmd"])
    
    code = template.format(
        data_source=data_source,
        ra=kwargs.get("ra", 0),
        dec=kwargs.get("dec", 0),
        radius=kwargs.get("radius", 0.1)
    )
    
    return {
        "status": "success",
        "plot_type": plot_type,
        "data_source": data_source,
        "python_code": code,
        "instructions": "Execute this Python code in your local environment with astroquery and matplotlib installed."
    }


# =============================================================================
# Lambda Handler
# =============================================================================

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Main Lambda handler for Bedrock Agent actions.
    
    Event structure from Bedrock Agent:
    {
        "actionGroup": "...",
        "function": "...",
        "parameters": [{"name": "...", "value": "..."}]
    }
    """
    
    print(f"Event: {json.dumps(event)}")
    
    # Extract action details
    action_group = event.get("actionGroup", "")
    function_name = event.get("function", "")
    parameters = event.get("parameters", [])
    
    # Convert parameters list to dict
    params = {p["name"]: p["value"] for p in parameters}
    
    # Route to appropriate function
    result = None
    
    if function_name == "resolve_object":
        result = resolve_object(params.get("object_name", ""))
    
    elif function_name == "query_gaia":
        result = query_gaia(
            ra=float(params.get("ra", 0)),
            dec=float(params.get("dec", 0)),
            radius_arcmin=float(params.get("radius_arcmin", 5)),
            limit=int(params.get("limit", 100))
        )
    
    elif function_name == "search_literature":
        result = search_ads(
            query=params.get("query", ""),
            max_results=int(params.get("max_results", 10))
        )
    
    elif function_name == "get_citations":
        bibcodes = params.get("bibcodes", "").split(",")
        result = get_bibtex(bibcodes)
    
    elif function_name == "compute_selection":
        result = compute_color_cut_stats(
            color_min=float(params.get("color_min", 0)),
            color_max=float(params.get("color_max", 3)),
            mag_min=float(params.get("mag_min")) if params.get("mag_min") else None,
            mag_max=float(params.get("mag_max")) if params.get("mag_max") else None,
            ra=float(params.get("ra")) if params.get("ra") else None,
            dec=float(params.get("dec")) if params.get("dec") else None,
        )
    
    elif function_name == "generate_plot":
        result = generate_plot_instructions(
            plot_type=params.get("plot_type", "cmd"),
            data_source=params.get("data_source", ""),
            ra=float(params.get("ra", 0)),
            dec=float(params.get("dec", 0)),
            radius=float(params.get("radius", 0.1))
        )
    
    else:
        result = {
            "status": "error",
            "message": f"Unknown function: {function_name}"
        }
    
    # Format response for Bedrock Agent
    response = {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "function": function_name,
            "functionResponse": {
                "responseBody": {
                    "TEXT": {
                        "body": json.dumps(result, indent=2)
                    }
                }
            }
        }
    }
    
    return response


# =============================================================================
# Local Testing
# =============================================================================

if __name__ == "__main__":
    # Test resolve object
    print("Testing resolve_object...")
    result = resolve_object("M31")
    print(json.dumps(result, indent=2))
    
    # Test Gaia query
    print("\nTesting query_gaia...")
    result = query_gaia(10.68, 41.27, radius_arcmin=1)
    print(json.dumps(result, indent=2))
    
    # Test ADS search (requires token)
    print("\nTesting search_ads...")
    result = search_ads("exoplanet atmosphere JWST", max_results=3)
    print(json.dumps(result, indent=2))
