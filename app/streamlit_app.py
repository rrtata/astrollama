#!/usr/bin/env python3
"""
AstroLlama - Agent-Integrated Chat Interface
=============================================
The fine-tuned AstroLlama model on Bedrock serves as the agent brain.
Every user query goes through the agent loop where the model decides:
- When to search catalogs (coordinates, object names detected)
- When to query RAG (domain knowledge needed)
- When to execute code (calculations, plots requested)
- When to search literature (research questions)
- When to just respond directly (simple questions)
"""

import os
import sys
import json
import base64
import re
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import streamlit as st
import pandas as pd
import numpy as np
import boto3
import requests

# Page configuration
st.set_page_config(
    page_title="AstroLlama",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A5F; }
    .sub-header { font-size: 1.2rem; color: #666; margin-top: 0; }
    .tool-used {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 2px;
        font-weight: 500;
    }
    .tool-catalog { background-color: #e3f2fd; color: #1565c0; }
    .tool-rag { background-color: #f3e5f5; color: #7b1fa2; }
    .tool-code { background-color: #e8f5e9; color: #2e7d32; }
    .tool-literature { background-color: #fff3e0; color: #ef6c00; }
    .tool-lookup { background-color: #fce4ec; color: #c2185b; }
    .error-box {
        background-color: #ffebee;
        border-left: 3px solid #c62828;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============ Configuration ============

@st.cache_resource
def get_secrets():
    """Get secrets from Streamlit secrets or environment"""
    secrets = {}
    keys = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
            "ASTROLLAMA_MODEL_ID", "ADS_TOKEN", "PINECONE_API_KEY"]
    
    for key in keys:
        if hasattr(st, 'secrets') and key in st.secrets:
            secrets[key] = st.secrets[key]
        else:
            secrets[key] = os.environ.get(key, "")
    
    return secrets


@st.cache_resource
def init_bedrock_client():
    """Initialize AWS Bedrock client"""
    secrets = get_secrets()
    try:
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=secrets.get("AWS_REGION", "us-west-2"),
            aws_access_key_id=secrets.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secrets.get("AWS_SECRET_ACCESS_KEY")
        )
    except Exception as e:
        st.error(f"Failed to initialize AWS client: {e}")
        return None


@st.cache_resource
def init_rag_client():
    """Initialize Pinecone RAG client"""
    secrets = get_secrets()
    if not secrets.get("PINECONE_API_KEY"):
        return None
    
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        
        pc = Pinecone(api_key=secrets["PINECONE_API_KEY"])
        index = pc.Index("astrollama-rag")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return {"index": index, "encoder": encoder}
    except Exception as e:
        return None


# ============ Tool Implementations ============

class AstroTools:
    """Collection of tools the agent can use"""
    
    @staticmethod
    def search_literature(query: str, max_results: int = 5) -> Dict:
        """Search NASA ADS for papers"""
        secrets = get_secrets()
        token = secrets.get("ADS_TOKEN")
        
        if not token:
            return {"error": "ADS_TOKEN not configured", "data": []}
        
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://api.adsabs.harvard.edu/v1/search/query"
        
        params = {
            "q": query,
            "fl": "bibcode,title,abstract,author,year,citation_count",
            "rows": max_results,
            "sort": "citation_count desc"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            papers = response.json().get("response", {}).get("docs", [])
            
            results = []
            for p in papers:
                results.append({
                    "title": p.get("title", [""])[0] if isinstance(p.get("title"), list) else p.get("title", ""),
                    "authors": ", ".join(p.get("author", [])[:3]),
                    "year": p.get("year", ""),
                    "citations": p.get("citation_count", 0),
                    "abstract": (p.get("abstract", "")[:200] + "...") if p.get("abstract") else "",
                    "bibcode": p.get("bibcode", "")
                })
            
            return {"data": results, "query": query}
        except Exception as e:
            return {"error": str(e), "data": []}
    
    @staticmethod
    def query_catalog(catalog: str, ra: float, dec: float, radius: float = 60) -> Dict:
        """Query astronomical catalog by coordinates"""
        try:
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
            
            if catalog.lower() == "gaia":
                from astroquery.gaia import Gaia
                Gaia.ROW_LIMIT = 500
                radius_deg = radius / 3600.0
                
                query = f"""
                SELECT source_id, ra, dec, parallax, parallax_error,
                       pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, 
                       phot_rp_mean_mag, bp_rp
                FROM gaiadr3.gaia_source
                WHERE CONTAINS(POINT('ICRS', ra, dec),
                    CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
                ORDER BY phot_g_mean_mag ASC
                """
                job = Gaia.launch_job(query)
                result = job.get_results()
                df = result.to_pandas()
                
            elif catalog.lower() in ["2mass", "allwise", "catwise"]:
                from astroquery.vizier import Vizier
                
                catalog_map = {
                    "2mass": "II/246/out",
                    "allwise": "II/328/allwise",
                    "catwise": "II/365/catwise"
                }
                
                vizier = Vizier(row_limit=500)
                result = vizier.query_region(coord, radius=radius*u.arcsec, 
                                            catalog=catalog_map[catalog.lower()])
                df = result[0].to_pandas() if result else pd.DataFrame()
                
            elif catalog.lower() == "simbad":
                from astroquery.simbad import Simbad
                
                simbad = Simbad()
                simbad.add_votable_fields('otype', 'sptype', 'plx', 'pm')
                result = simbad.query_region(coord, radius=radius*u.arcsec)
                df = result.to_pandas() if result else pd.DataFrame()
            
            else:
                return {"error": f"Unknown catalog: {catalog}", "data": []}
            
            if len(df) > 0:
                return {
                    "catalog": catalog.upper(),
                    "total_found": len(df),
                    "data": df.head(20).to_dict(orient="records"),
                    "columns": list(df.columns),
                    "full_data": df,
                    "ra": ra, "dec": dec, "radius": radius
                }
            else:
                return {"catalog": catalog.upper(), "total_found": 0, "data": [], 
                        "ra": ra, "dec": dec, "radius": radius}
                
        except Exception as e:
            return {"error": str(e), "data": []}
    
    @staticmethod
    def lookup_object(name: str) -> Dict:
        """
        Look up object by name using multiple resolvers.
        Handles: star names, cluster names (M45, NGC), coordinates, 2MASS IDs, etc.
        """
        try:
            from astropy.coordinates import SkyCoord
            from astroquery.simbad import Simbad
            from astropy import units as u
            import re
            
            ra, dec = None, None
            resolved_name = name
            
            # Check if input is already coordinates (e.g., "56.75 24.12" or "56.75, 24.12")
            coord_pattern = r'^[\s]*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[\s]*$'
            coord_match = re.match(coord_pattern, name.strip())
            
            if coord_match:
                ra = float(coord_match.group(1))
                dec = float(coord_match.group(2))
                resolved_name = f"Coordinates ({ra:.4f}, {dec:.4f})"
            else:
                # Try different name formats and resolvers
                names_to_try = [name]
                
                # Handle common cluster aliases
                cluster_aliases = {
                    "pleiades": "M45",
                    "hyades": "Mel 25",
                    "orion nebula": "M42",
                    "crab nebula": "M1",
                    "andromeda": "M31",
                    "praesepe": "M44",
                    "beehive": "M44",
                }
                
                name_lower = name.lower().strip()
                if name_lower in cluster_aliases:
                    names_to_try.insert(0, cluster_aliases[name_lower])
                
                # Also try with "cluster" suffix for known clusters
                if name_lower in ["pleiades", "hyades", "praesepe"]:
                    names_to_try.append(f"{name} cluster")
                
                # Try each name with SkyCoord.from_name (uses Sesame resolver)
                for try_name in names_to_try:
                    try:
                        coord = SkyCoord.from_name(try_name)
                        ra, dec = coord.ra.degree, coord.dec.degree
                        resolved_name = try_name
                        break
                    except Exception:
                        continue
                
                # If still not resolved, try SIMBAD directly with wildcards
                if ra is None:
                    try:
                        simbad = Simbad()
                        # Try exact match first
                        result = simbad.query_object(name)
                        if result and len(result) > 0:
                            ra = float(result['RA'][0].replace(' ', ':').split(':')[0]) * 15  # Rough conversion
                            dec = float(result['DEC'][0].replace(' ', ':').split(':')[0])
                            # Get proper coordinates
                            coord = SkyCoord(result['RA'][0], result['DEC'][0], unit=(u.hourangle, u.deg))
                            ra, dec = coord.ra.degree, coord.dec.degree
                    except Exception:
                        pass
            
            if ra is None or dec is None:
                # Last resort: provide known coordinates for common objects
                known_coords = {
                    "pleiades": (56.75, 24.12),
                    "m45": (56.75, 24.12),
                    "hyades": (66.75, 15.87),
                    "orion": (83.82, -5.39),
                    "m42": (83.82, -5.39),
                    "trapezium": (83.82, -5.39),
                    "taurus": (68.0, 28.0),  # Taurus star-forming region
                    "rho ophiuchi": (246.79, -24.53),
                    "rho oph": (246.79, -24.53),
                    "ic 348": (56.13, 32.17),
                    "chamaeleon": (168.0, -77.0),
                    "upper sco": (244.0, -23.0),
                    "upper scorpius": (244.0, -23.0),
                }
                
                name_lower = name.lower().strip()
                if name_lower in known_coords:
                    ra, dec = known_coords[name_lower]
                    resolved_name = f"{name} (from known coordinates)"
                else:
                    return {"error": f"Could not resolve '{name}'. Try providing coordinates directly (e.g., '56.75 24.12') or a catalog ID.", "resolved": False}
            
            # Now get additional info from SIMBAD
            info = {
                "name": name,
                "resolved_name": resolved_name,
                "ra": round(ra, 6),
                "dec": round(dec, 6),
                "resolved": True
            }
            
            try:
                simbad = Simbad()
                simbad.add_votable_fields('otype', 'sptype', 'plx', 'pm', 
                                          'flux(V)', 'flux(J)', 'flux(K)')
                result = simbad.query_object(resolved_name.split('(')[0].strip())
                
                if result:
                    df = result.to_pandas()
                    if len(df) > 0:
                        row = df.iloc[0]
                        info["object_type"] = str(row.get("OTYPE", ""))
                        info["spectral_type"] = str(row.get("SP_TYPE", ""))
                        if pd.notna(row.get("PLX_VALUE")):
                            info["parallax"] = float(row.get("PLX_VALUE"))
                        if pd.notna(row.get("PMRA")):
                            info["pmra"] = float(row.get("PMRA"))
                        if pd.notna(row.get("PMDEC")):
                            info["pmdec"] = float(row.get("PMDEC"))
            except Exception:
                pass  # SIMBAD info is optional
            
            return info
            
        except Exception as e:
            return {"error": str(e), "resolved": False}
    
    @staticmethod
    def query_rag(query: str, top_k: int = 3) -> Dict:
        """Query the RAG knowledge base"""
        rag_client = init_rag_client()
        
        if not rag_client:
            return {"error": "RAG not configured", "data": []}
        
        try:
            embedding = rag_client["encoder"].encode(query).tolist()
            results = rag_client["index"].query(
                vector=embedding, top_k=top_k, include_metadata=True
            )
            
            passages = []
            for match in results.matches:
                passages.append({
                    "score": round(match.score, 3),
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "unknown"),
                    "title": match.metadata.get("title", "")
                })
            
            return {"data": passages, "query": query}
        except Exception as e:
            return {"error": str(e), "data": []}
    
    @staticmethod
    def execute_code(code: str, data_context: Dict = None) -> Dict:
        """Execute Python code and capture output/plots"""
        import io
        import contextlib
        import traceback
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Remove plt.show() as it doesn't work in non-interactive mode
        code = code.replace('plt.show()', '# plt.show() removed for capture')
        
        namespace = {}
        setup = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
"""
        exec(setup, namespace)
        
        if data_context:
            for name, data in data_context.items():
                if isinstance(data, pd.DataFrame):
                    namespace[name] = data
                elif isinstance(data, dict) and 'full_data' in data:
                    namespace[name] = data['full_data']
                # Also store scalar values (like last_ra, last_dec)
                elif isinstance(data, (int, float, str)):
                    namespace[name] = data
        
        stdout_capture = io.StringIO()
        plots = []
        error = None
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, namespace)
            
            # Capture all figures
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', 
                           dpi=100, facecolor='white')
                buf.seek(0)
                plots.append(base64.b64encode(buf.read()).decode('utf-8'))
            plt.close('all')
            
        except Exception as e:
            error = traceback.format_exc()
        
        return {
            "output": stdout_capture.getvalue(),
            "plots": plots,
            "error": error
        }


# ============ Agent System Prompt ============

AGENT_SYSTEM_PROMPT = """You are AstroLlama, an expert astronomical research assistant specializing in brown dwarfs, substellar objects, and stellar astrophysics. You have access to powerful tools that you MUST use to provide accurate, data-driven responses.

## YOUR TOOLS

### 1. CATALOG_QUERY - Query astronomical databases
Use when user provides coordinates OR after resolving an object name.
Catalogs: gaia, 2mass, allwise, catwise, simbad
Format: TOOL:CATALOG_QUERY|catalog=gaia|ra=180.0|dec=45.0|radius=60
Note: For large regions like clusters, use radius=3600 (1 degree)

### 2. OBJECT_LOOKUP - Resolve object names to coordinates
Use FIRST when user mentions any object by name (stars, clusters, regions).
Works with: Star names, M/NGC numbers, cluster names (Pleiades, Hyades), coordinates
Format: TOOL:OBJECT_LOOKUP|name=Pleiades
Format: TOOL:OBJECT_LOOKUP|name=M45
Format: TOOL:OBJECT_LOOKUP|name=56.75 24.12

### 3. LITERATURE_SEARCH - Search NASA ADS for papers
Use for research questions or to cite sources.
Format: TOOL:LITERATURE_SEARCH|query=brown dwarf atmospheres|max_results=5

### 4. RAG_QUERY - Search knowledge base
Use for background information on concepts/methods.
Format: TOOL:RAG_QUERY|query=T dwarf identification

### 5. CODE_EXECUTION - Run Python code (MUST USE FOR ALL PLOTS)
Use for calculations, plotting, data analysis, filtering.
Available: numpy (np), pandas (pd), matplotlib.pyplot (plt), astropy
Data from CATALOG_QUERY is available as DataFrames: gaia_data, twomass_data, allwise_data, etc.

IMPORTANT 2MASS column names: Jmag, Hmag, Kmag (not J, H, K)
IMPORTANT: Calculate colors like this: twomass_data['Jmag'] - twomass_data['Kmag']

Format: TOOL:CODE_EXECUTION|code=<your python code on single line>

EXAMPLE - To plot WISE colors, you MUST output exactly:
TOOL:CODE_EXECUTION|code=import matplotlib.pyplot as plt; w1w2 = allwise_data['W1mag'] - allwise_data['W2mag']; w2w3 = allwise_data['W2mag'] - allwise_data['W3mag']; plt.figure(); plt.scatter(w1w2, w2w3, s=20); plt.xlabel('W1-W2'); plt.ylabel('W2-W3'); plt.title('WISE Colors')

## CRITICAL RULES

1. **NEVER invent data** - Always use tools for real measurements
2. **Object/cluster mentioned → OBJECT_LOOKUP first** to get coordinates
3. **Coordinates given → CATALOG_QUERY** immediately  
4. **Plot requested → CATALOG_QUERY then CODE_EXECUTION**
5. **Chain tools**: lookup → catalog → code for complete analysis
6. **For clusters**: Use radius=3600 (1 degree) to capture full region
7. **Color calculations**: J-K = Jmag - Kmag, H-K = Hmag - Kmag
8. **ALWAYS EXECUTE CODE** - NEVER show code in markdown blocks. ALWAYS use TOOL:CODE_EXECUTION|code=... to run it. If you write ```python, you are doing it WRONG.
9. **USE LATEST DATA** - Data from CATALOG_QUERY is available as: gaia_data, twomass_data, allwise_data, catwise_data, simbad_data.
10. **DONT USE SAMPLE DATA** - Never create sample/fake data. Use real data from catalog queries.
11. **CODE ON SINGLE LINE** - Put all code on one line with semicolons. Do NOT use newlines in code.

## CODE EXAMPLES FOR COMMON TASKS

Color-Color diagram (2MASS):
```python
jk = twomass_data['Jmag'] - twomass_data['Kmag']
hk = twomass_data['Hmag'] - twomass_data['Kmag']
plt.scatter(hk, jk, s=5, alpha=0.5)
red = jk > 1.0
plt.scatter(hk[red], jk[red], c='red', s=20, label='J-K > 1')
plt.xlabel('H-K'); plt.ylabel('J-K'); plt.legend(); plt.title('Color-Color Diagram')
```

WISE Color-Color diagram:
```python
w1w2 = allwise_data['W1mag'] - allwise_data['W2mag']
w2w3 = allwise_data['W2mag'] - allwise_data['W3mag']
plt.scatter(w1w2, w2w3, s=20, alpha=0.7)
plt.xlabel('W1-W2'); plt.ylabel('W2-W3'); plt.title('WISE Color-Color Diagram')
plt.axvline(x=0.8, color='r', linestyle='--', label='T dwarf cut')
plt.legend()
```

CMD with Gaia:
```python
plt.scatter(gaia_data['bp_rp'], gaia_data['phot_g_mean_mag'], s=5, alpha=0.5)
plt.gca().invert_yaxis()
plt.xlabel('BP-RP'); plt.ylabel('G mag'); plt.title('CMD')
```

Find reddest sources:
```python
twomass_data['JK'] = twomass_data['Jmag'] - twomass_data['Kmag']
red_sources = twomass_data.nlargest(10, 'JK')
print(red_sources[['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Kmag', 'JK']])
```

## RESPONSE FORMAT

When using tools, output EXACTLY:
TOOL:TOOL_NAME|param1=value1|param2=value2

After tool results, either use more tools or provide your final answer WITHOUT tool tags.

Remember: Your value comes from providing REAL DATA from catalogs, not generic information."""


# ============ Agent Core ============

@dataclass
class AgentState:
    """State maintained across the agent loop"""
    tools_used: List[str] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    plots: List[str] = field(default_factory=list)
    data_context: Dict[str, Any] = field(default_factory=dict)


def parse_tool_calls(response: str) -> List[Tuple[str, Dict]]:
    """Parse tool calls from model response"""
    tool_pattern = r'TOOL:(\w+)\|(.+?)(?=TOOL:|$|\n\n)'
    matches = re.findall(tool_pattern, response, re.DOTALL)
    
    calls = []
    for tool_name, params_str in matches:
        params = {}
        for param in params_str.strip().split('|'):
            if '=' in param:
                key, value = param.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['ra', 'dec', 'radius']:
                    try:
                        value = float(value)
                    except:
                        pass
                elif key in ['max_results', 'top_k']:
                    try:
                        value = int(value)
                    except:
                        pass
                
                params[key] = value
        
        calls.append((tool_name.upper(), params))
    
    # Fallback: If no tool calls found but there's a code block with plotting, auto-execute it
    if not calls and ('```python' in response or '```' in response):
        code_pattern = r'```(?:python)?\n(.*?)```'
        code_matches = re.findall(code_pattern, response, re.DOTALL)
        if code_matches:
            # Check if code looks like it should produce a plot
            code = code_matches[0]
            if 'plt.' in code or 'scatter' in code or 'plot' in code:
                calls.append(('CODE_EXECUTION', {'code': code}))
    
    return calls


def execute_tool(tool_name: str, params: Dict, state: AgentState) -> str:
    """Execute a tool and return formatted result"""
    state.tools_used.append(tool_name)
    
    if tool_name == "CATALOG_QUERY":
        result = AstroTools.query_catalog(
            catalog=params.get("catalog", "gaia"),
            ra=params.get("ra", 0),
            dec=params.get("dec", 0),
            radius=params.get("radius", 60)
        )
        state.tool_results.append({"tool": tool_name, "result": result})
        
        if result.get("data") or result.get("full_data") is not None:
            catalog_name = params.get('catalog', 'data').lower()
            # Fix: Python variable names can't start with numbers
            if catalog_name == '2mass':
                catalog_name = 'twomass'
            if 'full_data' in result:
                state.data_context[f"{catalog_name}_data"] = result['full_data']
            else:
                state.data_context[f"{catalog_name}_data"] = pd.DataFrame(result['data'])
            
            df_display = pd.DataFrame(result['data']) if result.get('data') else result.get('full_data', pd.DataFrame()).head(10)
            
            return f"""CATALOG QUERY RESULT ({result.get('catalog', 'Unknown')}):
Found {result.get('total_found', 0)} sources at RA={params.get('ra'):.4f}, Dec={params.get('dec'):.4f}, radius={params.get('radius')}\"
Columns: {', '.join(result.get('columns', [])[:8])}
Data available as '{catalog_name}_data' for code execution.

Sample (first 5 rows):
{df_display.head(5).to_string()}
"""
        else:
            return f"CATALOG QUERY: No sources found. Error: {result.get('error', 'None')}"
    
    elif tool_name == "OBJECT_LOOKUP":
        result = AstroTools.lookup_object(params.get("name", ""))
        state.tool_results.append({"tool": tool_name, "result": result})
        
        if result.get("resolved"):
            state.data_context["last_ra"] = result.get("ra")
            state.data_context["last_dec"] = result.get("dec")
            
            return f"""OBJECT LOOKUP RESULT:
Name: {result.get('name')}
RA: {result.get('ra'):.6f} deg
Dec: {result.get('dec'):.6f} deg
Object Type: {result.get('object_type', 'Unknown')}
Spectral Type: {result.get('spectral_type', 'N/A')}
Parallax: {result.get('parallax', 'N/A')} mas
Proper Motion: ({result.get('pmra', 'N/A')}, {result.get('pmdec', 'N/A')}) mas/yr

You can now query catalogs at these coordinates."""
        else:
            return f"OBJECT LOOKUP FAILED: Could not resolve '{params.get('name')}'. Error: {result.get('error', 'Unknown')}"
    
    elif tool_name == "LITERATURE_SEARCH":
        result = AstroTools.search_literature(
            query=params.get("query", ""),
            max_results=params.get("max_results", 5)
        )
        state.tool_results.append({"tool": tool_name, "result": result})
        
        if result.get("data"):
            papers_str = f"LITERATURE SEARCH RESULT for '{params.get('query')}':\n\n"
            for i, p in enumerate(result["data"], 1):
                papers_str += f"{i}. {p['title']} ({p['year']})\n"
                papers_str += f"   Authors: {p['authors']}\n"
                papers_str += f"   Citations: {p['citations']} | Bibcode: {p['bibcode']}\n\n"
                state.sources.append(p['bibcode'])
            return papers_str
        else:
            return f"LITERATURE SEARCH: No papers found. Error: {result.get('error', 'None')}"
    
    elif tool_name == "RAG_QUERY":
        result = AstroTools.query_rag(
            query=params.get("query", ""),
            top_k=params.get("top_k", 3)
        )
        state.tool_results.append({"tool": tool_name, "result": result})
        
        if result.get("data"):
            rag_str = f"RAG KNOWLEDGE BASE RESULT for '{params.get('query')}':\n\n"
            for i, p in enumerate(result["data"], 1):
                rag_str += f"{i}. Source: {p['source']} (relevance: {p['score']})\n"
                rag_str += f"   {p['text'][:400]}...\n\n"
            return rag_str
        else:
            return f"RAG QUERY: No relevant information found. Error: {result.get('error', 'None')}"
    
    elif tool_name == "CODE_EXECUTION":
        code = params.get("code", "")
        result = AstroTools.execute_code(code, state.data_context)
        state.tool_results.append({"tool": tool_name, "result": result})
        
        if result.get("plots"):
            state.plots.extend(result["plots"])
        
        output_str = "CODE EXECUTION RESULT:\n"
        if result.get("output"):
            output_str += f"Output:\n{result['output']}\n"
        if result.get("plots"):
            output_str += f"✓ Generated {len(result['plots'])} plot(s) - displayed below\n"
        if result.get("error"):
            output_str += f"Error:\n{result['error']}\n"
        if not result.get("output") and not result.get("plots") and not result.get("error"):
            output_str += "Code executed successfully (no output)\n"
        
        return output_str
    
    else:
        return f"Unknown tool: {tool_name}"


def call_astrollama(messages: List[Dict], client, max_retries: int = 5) -> str:
    """Call AstroLlama model on Bedrock using Llama format with retry logic"""
    secrets = get_secrets()
    model_id = secrets.get("ASTROLLAMA_MODEL_ID")
    
    if not model_id:
        return "Error: ASTROLLAMA_MODEL_ID not configured in secrets"
    
    # Build prompt in Llama 3 format
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{AGENT_SYSTEM_PROMPT}<|eot_id|>"
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Llama model request body
    body = {
        "prompt": prompt,
        "max_gen_len": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Retry with exponential backoff for throttling
    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            result = json.loads(response["body"].read())
            return result.get("generation", "")
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a throttling error
            if "ThrottlingException" in error_msg or "Too many requests" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: Model rate limited. Please wait a minute and try again."
            else:
                return f"Error calling AstroLlama model: {error_msg}"
    
    return "Error: Failed after multiple retries"


def run_agent(user_query: str, client, progress_callback=None, max_iterations: int = 4) -> Tuple[str, AgentState]:
    """Run the agent loop with rate limiting protection"""
    state = AgentState()
    
    # Load persistent data from previous queries
    if "persistent_data" in st.session_state:
        state.data_context.update(st.session_state.persistent_data)
    
    messages = [{"role": "user", "content": user_query}]
    
    # If we have existing data, mention it in context
    existing_data = list(state.data_context.keys())
    if existing_data:
        data_names = [k for k in existing_data if k.endswith('_data')]
        if data_names:
            messages[0]["content"] += f"\n\n[Note: Data already available from previous queries: {', '.join(data_names)}]"
    
    for iteration in range(max_iterations):
        if progress_callback:
            progress_callback(f"Step {iteration + 1}: Analyzing...")
        
        # Add small delay between iterations to avoid rate limiting
        if iteration > 0:
            time.sleep(2)
        
        response = call_astrollama(messages, client)
        
        # Check for errors
        if response.startswith("Error"):
            return response, state
        
        tool_calls = parse_tool_calls(response)
        
        if not tool_calls:
            # Save data to persistent state before returning
            st.session_state.persistent_data.update(state.data_context)
            return response, state
        
        tool_results_text = []
        for tool_name, params in tool_calls:
            if progress_callback:
                progress_callback(f"Using {tool_name.replace('_', ' ').title()}...")
            
            result_text = execute_tool(tool_name, params, state)
            tool_results_text.append(result_text)
        
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": f"""Tool Results:

{chr(10).join(tool_results_text)}

---
Based on these results, continue your analysis. 
- If you need more data, use additional tools.
- If you have enough information, provide your comprehensive answer WITHOUT tool tags.
- Remember to reference specific values from the data.
- For plots: ALWAYS use CODE_EXECUTION tool to run the code, don't just show code."""
        })
    
    messages.append({
        "role": "user", 
        "content": "Please provide your final answer now based on all the information gathered."
    })
    final_response = call_astrollama(messages, client)
    
    # Save data to persistent state
    st.session_state.persistent_data.update(state.data_context)
    
    return final_response, state


# ============ Sidebar ============

def render_sidebar():
    """Render the sidebar"""
    with st.sidebar:
        st.markdown("## 🦙 AstroLlama")
        st.markdown("*Agent-Powered Research Assistant*")
        
        st.divider()
        
        st.markdown("### System Status")
        secrets = get_secrets()
        
        statuses = [
            ("AWS Bedrock", bool(secrets.get("AWS_ACCESS_KEY_ID"))),
            ("AstroLlama Model", bool(secrets.get("ASTROLLAMA_MODEL_ID"))),
            ("RAG Knowledge Base", bool(secrets.get("PINECONE_API_KEY"))),
            ("NASA ADS", bool(secrets.get("ADS_TOKEN")))
        ]
        
        for name, status in statuses:
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} {name}")
        
        st.divider()
        
        st.markdown("### 🛠️ Available Tools")
        st.caption("The agent automatically uses these based on your query:")
        tools_info = [
            ("🔭 Catalogs", "Gaia, 2MASS, WISE, SIMBAD"),
            ("🔍 Object Lookup", "Name → Coordinates"),
            ("📚 Literature", "NASA ADS papers"),
            ("🧠 Knowledge Base", "RAG retrieval"),
            ("💻 Code & Plots", "Python analysis")
        ]
        for tool, desc in tools_info:
            st.markdown(f"**{tool}**: {desc}")
        
        st.divider()
        
        # Show cached data info
        if "persistent_data" in st.session_state and st.session_state.persistent_data:
            data_keys = [k for k in st.session_state.persistent_data.keys() if k.endswith('_data')]
            if data_keys:
                st.caption(f"📦 Cached: {', '.join(data_keys)}")
        
        if st.button("🗑️ Clear Chat & Data", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_states = []
            st.session_state.persistent_data = {}
            st.rerun()
        
        st.divider()
        
        st.markdown("### 💡 Try These")
        examples = [
            "What do we know about TRAPPIST-1?",
            "Plot a CMD for RA=83.8, Dec=-5.4 (Orion)",
            "Find brown dwarfs near 2MASS J0559-1404",
            "How do I identify L dwarfs from colors?",
            "Recent papers on Y dwarf atmospheres"
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state.pending_query = ex
                st.rerun()


def render_tool_badges(tools_used: List[str]):
    """Render badges for tools that were used"""
    if not tools_used:
        return
    
    badge_map = {
        "CATALOG_QUERY": ("🔭 Catalog", "tool-catalog"),
        "OBJECT_LOOKUP": ("🔍 Lookup", "tool-lookup"),
        "LITERATURE_SEARCH": ("📚 Literature", "tool-literature"),
        "RAG_QUERY": ("🧠 Knowledge", "tool-rag"),
        "CODE_EXECUTION": ("💻 Code", "tool-code")
    }
    
    badges_html = "<div style='margin: 10px 0;'>"
    for tool in set(tools_used):
        display, css_class = badge_map.get(tool, (tool, "tool-catalog"))
        badges_html += f'<span class="tool-used {css_class}">{display}</span>'
    badges_html += "</div>"
    
    st.markdown(badges_html, unsafe_allow_html=True)


# ============ Main ============

def main():
    """Main application"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_states" not in st.session_state:
        st.session_state.agent_states = []
    if "persistent_data" not in st.session_state:
        st.session_state.persistent_data = {}  # Persists data across messages
    
    render_sidebar()
    
    st.markdown('<p class="main-header">🦙 AstroLlama</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about astronomy — I\'ll search catalogs, find papers, and create plots automatically.</p>', unsafe_allow_html=True)
    
    client = init_bedrock_client()
    if client is None:
        st.error("Could not connect to AWS Bedrock. Check your credentials.")
        return
    
    # Check if model ID is configured
    secrets = get_secrets()
    if not secrets.get("ASTROLLAMA_MODEL_ID"):
        st.warning("⚠️ ASTROLLAMA_MODEL_ID not configured. Please add it to your Streamlit secrets.")
        st.code("""
# Add to .streamlit/secrets.toml or Streamlit Cloud secrets:
ASTROLLAMA_MODEL_ID = "arn:aws:bedrock:us-west-2:917791789035:custom-model-deployment/df4go8aqk6ix"
        """)
    
    st.divider()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and i // 2 < len(st.session_state.agent_states):
                state = st.session_state.agent_states[i // 2]
                
                render_tool_badges(state.tools_used)
                
                for plot_b64 in state.plots:
                    st.image(f"data:image/png;base64,{plot_b64}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if state.sources:
                        with st.expander(f"📚 Sources ({len(set(state.sources))})"):
                            for src in list(set(state.sources))[:5]:
                                st.markdown(f"[{src}](https://ui.adsabs.harvard.edu/abs/{src})")
                
                with col2:
                    data_items = [(k, v) for k, v in state.data_context.items() 
                                 if isinstance(v, pd.DataFrame)]
                    if data_items:
                        with st.expander(f"📊 Data ({len(data_items)} tables)"):
                            for name, df in data_items:
                                st.markdown(f"**{name}**: {len(df)} rows")
                                st.dataframe(df.head(5), use_container_width=True)
                                st.download_button(
                                    f"Download {name}",
                                    df.to_csv(index=False),
                                    f"{name}.csv",
                                    key=f"dl_{name}_{i}"
                                )
    
    # Handle pending query
    if "pending_query" in st.session_state:
        prompt = st.session_state.pending_query
        del st.session_state.pending_query
    else:
        prompt = st.chat_input("Ask about any astronomical object, coordinates, or topic...")
    
    # Process new query
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            status = st.empty()
            
            def update_status(msg):
                status.markdown(f"*{msg}*")
            
            update_status("🔭 Analyzing your query...")
            
            response, state = run_agent(prompt, client, progress_callback=update_status)
            
            status.empty()
            
            # Check for errors
            if response.startswith("Error"):
                st.markdown(f'<div class="error-box">{response}</div>', unsafe_allow_html=True)
            else:
                st.markdown(response)
            
            render_tool_badges(state.tools_used)
            
            for plot_b64 in state.plots:
                st.image(f"data:image/png;base64,{plot_b64}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if state.sources:
                    with st.expander(f"📚 Sources ({len(set(state.sources))})"):
                        for src in list(set(state.sources))[:5]:
                            st.markdown(f"[{src}](https://ui.adsabs.harvard.edu/abs/{src})")
            
            with col2:
                data_items = [(k, v) for k, v in state.data_context.items() 
                             if isinstance(v, pd.DataFrame)]
                if data_items:
                    with st.expander(f"📊 Data ({len(data_items)} tables)"):
                        for name, df in data_items:
                            st.markdown(f"**{name}**: {len(df)} rows")
                            st.dataframe(df.head(5), use_container_width=True)
                            st.download_button(
                                f"Download {name}",
                                df.to_csv(index=False),
                                f"{name}.csv",
                                key=f"dl_{name}_new"
                            )
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.agent_states.append(state)
        
        st.rerun()


if __name__ == "__main__":
    main()
