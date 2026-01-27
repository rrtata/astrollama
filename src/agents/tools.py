#!/usr/bin/env python3
"""
AstroLlama Agent Tools
Tools for searching ADS, arXiv, and astronomical catalogs.

These tools can be used by the LLM agent to retrieve real-time data.
"""

import os
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


# =============================================================================
# Tool Base Class
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    
    def to_dict(self):
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error
        }


# =============================================================================
# ADS Search Tool
# =============================================================================

class ADSSearchTool:
    """Search NASA ADS for astronomy papers."""
    
    name = "search_ads"
    description = """Search NASA ADS for astronomy papers. 
    Use this to find research papers, get abstracts, citations, and bibliographic information.
    Input should be a search query string."""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.environ.get("ADS_TOKEN")
        self.base_url = "https://api.adsabs.harvard.edu/v1"
    
    def _get_token(self) -> str:
        """Get ADS token from environment or Secrets Manager."""
        if self.api_token:
            return self.api_token
        
        # Try Secrets Manager
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name="us-west-2")
            response = client.get_secret_value(SecretId="astrollama/api-keys")
            secrets = json.loads(response["SecretString"])
            return secrets.get("ADS_TOKEN", "")
        except:
            return ""
    
    def search(self, query: str, rows: int = 10, sort: str = "citation_count desc") -> ToolResult:
        """
        Search ADS for papers.
        
        Args:
            query: Search query (e.g., "brown dwarf spectroscopy")
            rows: Number of results to return
            sort: Sort order (citation_count desc, date desc, etc.)
        
        Returns:
            ToolResult with list of papers
        """
        token = self._get_token()
        if not token:
            return ToolResult(success=False, data=None, error="ADS API token not configured")
        
        url = f"{self.base_url}/search/query"
        params = {
            "q": query,
            "rows": rows,
            "fl": "bibcode,title,author,year,abstract,citation_count,pub",
            "sort": sort
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
                papers = data.get("response", {}).get("docs", [])
                
                # Format results
                results = []
                for p in papers:
                    results.append({
                        "bibcode": p.get("bibcode"),
                        "title": p.get("title", [""])[0] if isinstance(p.get("title"), list) else p.get("title", ""),
                        "authors": p.get("author", [])[:5],  # First 5 authors
                        "year": p.get("year"),
                        "abstract": p.get("abstract", "")[:500] + "..." if len(p.get("abstract", "")) > 500 else p.get("abstract", ""),
                        "citations": p.get("citation_count", 0),
                        "publication": p.get("pub", "")
                    })
                
                return ToolResult(success=True, data=results)
                
        except urllib.error.HTTPError as e:
            return ToolResult(success=False, data=None, error=f"ADS API error: {e.code} {e.reason}")
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_paper(self, bibcode: str) -> ToolResult:
        """Get detailed information about a specific paper."""
        token = self._get_token()
        if not token:
            return ToolResult(success=False, data=None, error="ADS API token not configured")
        
        url = f"{self.base_url}/search/query"
        params = {
            "q": f"bibcode:{bibcode}",
            "fl": "bibcode,title,author,year,abstract,citation_count,pub,keyword,doi"
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
                papers = data.get("response", {}).get("docs", [])
                
                if papers:
                    return ToolResult(success=True, data=papers[0])
                else:
                    return ToolResult(success=False, data=None, error="Paper not found")
                
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


# =============================================================================
# arXiv Search Tool
# =============================================================================

class ArxivSearchTool:
    """Search arXiv for preprints."""
    
    name = "search_arxiv"
    description = """Search arXiv for astronomy preprints.
    Use this to find recent papers that may not yet be in ADS.
    Input should be a search query string."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 10, 
               category: str = "astro-ph") -> ToolResult:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            category: arXiv category (astro-ph, astro-ph.SR, astro-ph.EP, etc.)
        
        Returns:
            ToolResult with list of papers
        """
        import xml.etree.ElementTree as ET
        
        params = {
            "search_query": f"cat:{category}* AND all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            results = []
            for entry in root.findall("atom:entry", ns):
                try:
                    paper = {
                        "arxiv_id": entry.find("atom:id", ns).text.split("/")[-1],
                        "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                        "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)][:5],
                        "abstract": entry.find("atom:summary", ns).text.strip()[:500] + "...",
                        "published": entry.find("atom:published", ns).text[:10],
                        "categories": [c.get("term") for c in entry.findall("atom:category", ns)],
                        "pdf_url": None
                    }
                    
                    for link in entry.findall("atom:link", ns):
                        if link.get("title") == "pdf":
                            paper["pdf_url"] = link.get("href")
                            break
                    
                    results.append(paper)
                except:
                    continue
            
            return ToolResult(success=True, data=results)
            
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


# =============================================================================
# Gaia Query Tool
# =============================================================================

class GaiaQueryTool:
    """Query Gaia DR3 catalog via TAP."""
    
    name = "query_gaia"
    description = """Query the Gaia DR3 catalog for stellar data.
    Use this to get astrometry, photometry, and stellar parameters.
    Input should be an ADQL query or search parameters."""
    
    def __init__(self):
        self.tap_url = "https://gea.esac.esa.int/tap-server/tap/sync"
    
    def query(self, adql: str, max_rows: int = 1000) -> ToolResult:
        """
        Execute an ADQL query on Gaia DR3.
        
        Args:
            adql: ADQL query string
            max_rows: Maximum rows to return
        
        Returns:
            ToolResult with query results
        """
        # Add row limit if not present
        if "TOP" not in adql.upper() and max_rows:
            adql = adql.replace("SELECT", f"SELECT TOP {max_rows}", 1)
        
        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": adql
        }
        
        try:
            data = urllib.parse.urlencode(params).encode('utf-8')
            req = urllib.request.Request(self.tap_url, data=data)
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                # Extract column names and data
                columns = [col["name"] for col in result.get("metadata", [])]
                rows = result.get("data", [])
                
                # Convert to list of dicts
                data_list = []
                for row in rows[:100]:  # Limit response size
                    data_list.append(dict(zip(columns, row)))
                
                return ToolResult(success=True, data={
                    "columns": columns,
                    "rows": data_list,
                    "total_rows": len(rows)
                })
                
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def cone_search(self, ra: float, dec: float, radius_arcmin: float = 5,
                    mag_limit: float = 20) -> ToolResult:
        """
        Perform a cone search around coordinates.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            radius_arcmin: Search radius in arcminutes
            mag_limit: Magnitude limit (G band)
        """
        adql = f"""
        SELECT TOP 100 source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, phot_g_mean_mag, bp_rp
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec), 
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60})) = 1
          AND phot_g_mean_mag < {mag_limit}
        ORDER BY phot_g_mean_mag
        """
        return self.query(adql)
    
    def brown_dwarf_candidates(self, max_distance_pc: float = 50) -> ToolResult:
        """
        Search for brown dwarf candidates.
        
        Args:
            max_distance_pc: Maximum distance in parsecs
        """
        min_parallax = 1000 / max_distance_pc  # mas
        
        adql = f"""
        SELECT TOP 500 source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, phot_g_mean_mag, bp_rp,
               phot_g_mean_mag + 5*LOG10(parallax/100) AS abs_g
        FROM gaiadr3.gaia_source
        WHERE parallax > {min_parallax}
          AND parallax_over_error > 10
          AND bp_rp > 2.5
          AND phot_g_mean_mag + 5*LOG10(parallax/100) > 12
        ORDER BY bp_rp DESC
        """
        return self.query(adql)


# =============================================================================
# VizieR Query Tool (2MASS, WISE, etc.)
# =============================================================================

class VizierQueryTool:
    """Query VizieR catalogs (2MASS, WISE, etc.)."""
    
    name = "query_vizier"
    description = """Query VizieR astronomical catalogs including 2MASS, WISE, SDSS, etc.
    Use this for infrared photometry and catalog cross-matching."""
    
    CATALOGS = {
        "2mass": "II/246/out",      # 2MASS Point Source Catalog
        "wise": "II/328/allwise",   # AllWISE
        "sdss": "V/147/sdss12",     # SDSS DR12
        "ps1": "II/349/ps1",        # Pan-STARRS DR1
        "gaia": "I/355/gaiadr3",    # Gaia DR3
    }
    
    def __init__(self):
        self.tap_url = "http://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync"
    
    def query_catalog(self, catalog: str, ra: float, dec: float, 
                      radius_arcmin: float = 5) -> ToolResult:
        """
        Query a VizieR catalog around coordinates.
        
        Args:
            catalog: Catalog name (2mass, wise, sdss, ps1)
            ra: Right Ascension in degrees
            dec: Declination in degrees
            radius_arcmin: Search radius in arcminutes
        """
        catalog_id = self.CATALOGS.get(catalog.lower(), catalog)
        
        adql = f"""
        SELECT TOP 100 *
        FROM "{catalog_id}"
        WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000),
                         CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60}))
        """
        
        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": adql
        }
        
        try:
            data = urllib.parse.urlencode(params).encode('utf-8')
            req = urllib.request.Request(self.tap_url, data=data)
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                columns = [col["name"] for col in result.get("metadata", [])]
                rows = result.get("data", [])
                
                data_list = []
                for row in rows[:50]:
                    data_list.append(dict(zip(columns, row)))
                
                return ToolResult(success=True, data={
                    "catalog": catalog_id,
                    "columns": columns,
                    "rows": data_list,
                    "total_rows": len(rows)
                })
                
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def query_2mass(self, ra: float, dec: float, radius_arcmin: float = 5) -> ToolResult:
        """Query 2MASS catalog for JHK photometry."""
        return self.query_catalog("2mass", ra, dec, radius_arcmin)
    
    def query_wise(self, ra: float, dec: float, radius_arcmin: float = 5) -> ToolResult:
        """Query AllWISE catalog for W1-W4 photometry."""
        return self.query_catalog("wise", ra, dec, radius_arcmin)


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        self.register(ADSSearchTool())
        self.register(ArxivSearchTool())
        self.register(GaiaQueryTool())
        self.register(VizierQueryTool())
    
    def register(self, tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str):
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """List all available tools with descriptions."""
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for LLM function calling."""
        definitions = []
        
        # ADS Search
        definitions.append({
            "name": "search_ads",
            "description": "Search NASA ADS for astronomy research papers. Returns titles, authors, abstracts, and citation counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'brown dwarf spectroscopy')"
                    },
                    "rows": {
                        "type": "integer",
                        "description": "Number of results (default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        })
        
        # arXiv Search
        definitions.append({
            "name": "search_arxiv",
            "description": "Search arXiv for recent astronomy preprints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        })
        
        # Gaia Query
        definitions.append({
            "name": "query_gaia",
            "description": "Query Gaia DR3 catalog using ADQL. Use for astrometry, photometry, and stellar data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "adql": {
                        "type": "string",
                        "description": "ADQL query string for Gaia DR3"
                    }
                },
                "required": ["adql"]
            }
        })
        
        # Gaia Cone Search
        definitions.append({
            "name": "gaia_cone_search",
            "description": "Search Gaia DR3 around specific coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ra": {"type": "number", "description": "Right Ascension in degrees"},
                    "dec": {"type": "number", "description": "Declination in degrees"},
                    "radius_arcmin": {"type": "number", "description": "Search radius in arcminutes", "default": 5}
                },
                "required": ["ra", "dec"]
            }
        })
        
        # Brown dwarf search
        definitions.append({
            "name": "gaia_brown_dwarf_candidates",
            "description": "Search Gaia DR3 for brown dwarf candidates based on color and absolute magnitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_distance_pc": {
                        "type": "number",
                        "description": "Maximum distance in parsecs",
                        "default": 50
                    }
                },
                "required": []
            }
        })
        
        # 2MASS Query
        definitions.append({
            "name": "query_2mass",
            "description": "Query 2MASS catalog for JHK infrared photometry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ra": {"type": "number", "description": "Right Ascension in degrees"},
                    "dec": {"type": "number", "description": "Declination in degrees"},
                    "radius_arcmin": {"type": "number", "description": "Search radius in arcminutes", "default": 5}
                },
                "required": ["ra", "dec"]
            }
        })
        
        # WISE Query
        definitions.append({
            "name": "query_wise",
            "description": "Query AllWISE catalog for W1-W4 mid-infrared photometry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ra": {"type": "number", "description": "Right Ascension in degrees"},
                    "dec": {"type": "number", "description": "Declination in degrees"},
                    "radius_arcmin": {"type": "number", "description": "Search radius in arcminutes", "default": 5}
                },
                "required": ["ra", "dec"]
            }
        })
        
        return definitions
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with given parameters."""
        tool = self.tools.get(tool_name)
        
        if not tool:
            return ToolResult(success=False, data=None, error=f"Unknown tool: {tool_name}")
        
        # Map tool names to methods
        if tool_name == "search_ads":
            return tool.search(**kwargs)
        elif tool_name == "search_arxiv":
            return tool.search(**kwargs)
        elif tool_name == "query_gaia":
            return tool.query(**kwargs)
        elif tool_name == "gaia_cone_search":
            return self.tools["query_gaia"].cone_search(**kwargs)
        elif tool_name == "gaia_brown_dwarf_candidates":
            return self.tools["query_gaia"].brown_dwarf_candidates(**kwargs)
        elif tool_name == "query_2mass":
            return self.tools["query_vizier"].query_2mass(**kwargs)
        elif tool_name == "query_wise":
            return self.tools["query_vizier"].query_wise(**kwargs)
        else:
            return ToolResult(success=False, data=None, error=f"Tool method not implemented: {tool_name}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing AstroLlama Agent Tools")
    print("=" * 60)
    
    registry = ToolRegistry()
    
    # Test ADS
    print("\n1. Testing ADS Search...")
    ads = ADSSearchTool()
    result = ads.search("brown dwarf spectroscopy", rows=3)
    if result.success:
        print(f"   Found {len(result.data)} papers")
        for p in result.data:
            print(f"   - [{p['year']}] {p['title'][:50]}...")
    else:
        print(f"   Error: {result.error}")
    
    # Test arXiv
    print("\n2. Testing arXiv Search...")
    arxiv = ArxivSearchTool()
    result = arxiv.search("T dwarf", max_results=3)
    if result.success:
        print(f"   Found {len(result.data)} papers")
        for p in result.data:
            print(f"   - [{p['published']}] {p['title'][:50]}...")
    else:
        print(f"   Error: {result.error}")
    
    # Test Gaia
    print("\n3. Testing Gaia Query...")
    gaia = GaiaQueryTool()
    result = gaia.cone_search(ra=180.0, dec=45.0, radius_arcmin=1)
    if result.success:
        print(f"   Found {result.data['total_rows']} sources")
    else:
        print(f"   Error: {result.error}")
    
    # Test VizieR/2MASS
    print("\n4. Testing 2MASS Query...")
    vizier = VizierQueryTool()
    result = vizier.query_2mass(ra=180.0, dec=45.0, radius_arcmin=1)
    if result.success:
        print(f"   Found {result.data['total_rows']} sources")
    else:
        print(f"   Error: {result.error}")
    
    print("\n" + "=" * 60)
    print("Tool testing complete!")
