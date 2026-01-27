#!/usr/bin/env python3
"""
AstroLlama Extended Catalog Query Tools
Query all major astronomical archives and catalogs.

Supported Archives:
- Gaia (TAP)
- 2MASS, WISE (VizieR/IRSA)
- SDSS (CasJobs)
- Pan-STARRS (MAST)
- HST/JWST (MAST)
- Spitzer (IRSA)
- ESO Archive
- Simbad, NED
- Aladin (image cutouts)
"""

import os
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import base64


@dataclass
class QueryResult:
    """Result from a catalog query."""
    success: bool
    data: Any
    columns: List[str] = None
    total_rows: int = 0
    error: str = None
    
    def to_dict(self):
        return {
            "success": self.success,
            "data": self.data,
            "columns": self.columns,
            "total_rows": self.total_rows,
            "error": self.error
        }


# =============================================================================
# Base TAP Client
# =============================================================================

class TAPClient:
    """Generic TAP client for ADQL queries."""
    
    def __init__(self, tap_url: str):
        self.tap_url = tap_url
    
    def query(self, adql: str, timeout: int = 60) -> QueryResult:
        """Execute an ADQL query."""
        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": adql
        }
        
        try:
            data = urllib.parse.urlencode(params).encode('utf-8')
            req = urllib.request.Request(self.tap_url, data=data)
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                columns = [col["name"] for col in result.get("metadata", [])]
                rows = result.get("data", [])
                
                # Convert to list of dicts
                data_list = [dict(zip(columns, row)) for row in rows[:500]]
                
                return QueryResult(
                    success=True,
                    data=data_list,
                    columns=columns,
                    total_rows=len(rows)
                )
                
        except Exception as e:
            return QueryResult(success=False, data=None, error=str(e))


# =============================================================================
# Gaia Queries
# =============================================================================

class GaiaCatalog:
    """Query Gaia DR3 via TAP."""
    
    def __init__(self):
        self.tap = TAPClient("https://gea.esac.esa.int/tap-server/tap/sync")
    
    def cone_search(self, ra: float, dec: float, radius_arcmin: float = 5,
                    mag_limit: float = 21) -> QueryResult:
        """Cone search around coordinates."""
        adql = f"""
        SELECT TOP 500 source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
               bp_rp, radial_velocity
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(POINT('ICRS', ra, dec), 
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60})) = 1
          AND phot_g_mean_mag < {mag_limit}
        ORDER BY phot_g_mean_mag
        """
        return self.tap.query(adql)
    
    def brown_dwarf_candidates(self, max_distance_pc: float = 50,
                               min_bp_rp: float = 2.5) -> QueryResult:
        """Search for brown dwarf candidates."""
        min_parallax = 1000 / max_distance_pc
        
        adql = f"""
        SELECT TOP 500 source_id, ra, dec, parallax, parallax_error,
               pmra, pmdec, phot_g_mean_mag, bp_rp,
               phot_g_mean_mag + 5*LOG10(parallax/100) AS abs_g,
               SQRT(pmra*pmra + pmdec*pmdec) AS total_pm
        FROM gaiadr3.gaia_source
        WHERE parallax > {min_parallax}
          AND parallax_over_error > 10
          AND bp_rp > {min_bp_rp}
          AND phot_g_mean_mag + 5*LOG10(parallax/100) > 12
        ORDER BY bp_rp DESC
        """
        return self.tap.query(adql)
    
    def high_proper_motion(self, min_pm: float = 500, max_distance_pc: float = 100) -> QueryResult:
        """Search for high proper motion objects."""
        min_parallax = 1000 / max_distance_pc
        
        adql = f"""
        SELECT TOP 500 source_id, ra, dec, parallax, pmra, pmdec,
               SQRT(pmra*pmra + pmdec*pmdec) AS total_pm,
               phot_g_mean_mag, bp_rp
        FROM gaiadr3.gaia_source
        WHERE parallax > {min_parallax}
          AND parallax_over_error > 5
          AND SQRT(pmra*pmra + pmdec*pmdec) > {min_pm}
        ORDER BY total_pm DESC
        """
        return self.tap.query(adql)
    
    def query(self, adql: str) -> QueryResult:
        """Execute custom ADQL query."""
        return self.tap.query(adql)


# =============================================================================
# VizieR Queries (2MASS, WISE, etc.)
# =============================================================================

class VizieRCatalog:
    """Query VizieR catalogs via TAP."""
    
    CATALOGS = {
        "2mass": "II/246/out",
        "allwise": "II/328/allwise",
        "catwise": "II/365/catwise2",
        "sdss_dr12": "V/147/sdss12",
        "ps1": "II/349/ps1",
        "ukidss": "II/319/las9",
        "vhs": "II/367/vhs_dr5",
        "unwise": "II/363/unwise",
    }
    
    def __init__(self):
        self.tap = TAPClient("http://tapvizier.u-strasbg.fr/TAPVizieR/tap/sync")
    
    def cone_search(self, catalog: str, ra: float, dec: float, 
                    radius_arcmin: float = 5) -> QueryResult:
        """Cone search in a VizieR catalog."""
        catalog_id = self.CATALOGS.get(catalog.lower(), catalog)
        
        adql = f"""
        SELECT TOP 500 *
        FROM "{catalog_id}"
        WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000),
                         CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60}))
        """
        return self.tap.query(adql)
    
    def query_2mass(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Query 2MASS for JHK photometry."""
        return self.cone_search("2mass", ra, dec, radius_arcmin)
    
    def query_wise(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Query AllWISE for W1-W4 photometry."""
        return self.cone_search("allwise", ra, dec, radius_arcmin)
    
    def query_catwise(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Query CatWISE2020 for improved WISE photometry."""
        return self.cone_search("catwise", ra, dec, radius_arcmin)
    
    def brown_dwarf_wise_colors(self, w1_w2_min: float = 0.8) -> QueryResult:
        """Search WISE for T/Y dwarf candidates by color."""
        adql = f"""
        SELECT TOP 500 AllWISE, RAJ2000, DEJ2000, 
               W1mag, W2mag, W3mag, W4mag,
               W1mag - W2mag AS W1_W2
        FROM "II/328/allwise"
        WHERE W1mag - W2mag > {w1_w2_min}
          AND W1mag < 17
          AND ccf = '0000'
        ORDER BY W1mag - W2mag DESC
        """
        return self.tap.query(adql)


# =============================================================================
# MAST Queries (HST, JWST, Pan-STARRS)
# =============================================================================

class MASTCatalog:
    """Query MAST archive for HST, JWST, and Pan-STARRS data."""
    
    def __init__(self):
        self.base_url = "https://mast.stsci.edu/api/v0"
    
    def _request(self, endpoint: str, params: Dict) -> QueryResult:
        """Make a request to MAST API."""
        try:
            url = f"{self.base_url}/{endpoint}"
            data = json.dumps(params).encode('utf-8')
            
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if "data" in result:
                    return QueryResult(
                        success=True,
                        data=result["data"],
                        columns=result.get("fields", []),
                        total_rows=len(result["data"])
                    )
                else:
                    return QueryResult(
                        success=True,
                        data=result,
                        total_rows=1
                    )
                    
        except Exception as e:
            return QueryResult(success=False, data=None, error=str(e))
    
    def cone_search(self, ra: float, dec: float, radius_arcmin: float = 5,
                    missions: List[str] = None) -> QueryResult:
        """Search MAST for observations."""
        params = {
            "service": "Mast.Caom.Cone",
            "params": {
                "ra": ra,
                "dec": dec,
                "radius": radius_arcmin / 60  # Convert to degrees
            },
            "format": "json"
        }
        
        return self._request("invoke", params)
    
    def search_jwst(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Search for JWST observations."""
        params = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": "*",
                "filters": [
                    {"paramName": "obs_collection", "values": ["JWST"]},
                    {"paramName": "s_ra", "values": [{"min": ra - radius_arcmin/60, "max": ra + radius_arcmin/60}]},
                    {"paramName": "s_dec", "values": [{"min": dec - radius_arcmin/60, "max": dec + radius_arcmin/60}]}
                ]
            },
            "format": "json"
        }
        return self._request("invoke", params)
    
    def search_hst(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Search for HST observations."""
        params = {
            "service": "Mast.Caom.Filtered",
            "params": {
                "columns": "*",
                "filters": [
                    {"paramName": "obs_collection", "values": ["HST"]},
                    {"paramName": "s_ra", "values": [{"min": ra - radius_arcmin/60, "max": ra + radius_arcmin/60}]},
                    {"paramName": "s_dec", "values": [{"min": dec - radius_arcmin/60, "max": dec + radius_arcmin/60}]}
                ]
            },
            "format": "json"
        }
        return self._request("invoke", params)
    
    def panstarrs_cone(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Query Pan-STARRS DR2 catalog."""
        params = {
            "service": "Mast.Catalogs.Panstarrs.Cone",
            "params": {
                "ra": ra,
                "dec": dec,
                "radius": radius_arcmin / 60
            },
            "format": "json"
        }
        return self._request("invoke", params)


# =============================================================================
# IRSA Queries (WISE, Spitzer, 2MASS)
# =============================================================================

class IRSACatalog:
    """Query IRSA for infrared survey data."""
    
    def __init__(self):
        self.base_url = "https://irsa.ipac.caltech.edu"
    
    def query_tap(self, adql: str) -> QueryResult:
        """Execute TAP query on IRSA."""
        tap = TAPClient(f"{self.base_url}/TAP/sync")
        return tap.query(adql)
    
    def wise_cone(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Query AllWISE via IRSA TAP."""
        adql = f"""
        SELECT TOP 500 designation, ra, dec, w1mpro, w2mpro, w3mpro, w4mpro,
               w1sigmpro, w2sigmpro, cc_flags, ext_flg, ph_qual
        FROM allwise_p3as_psd
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60})) = 1
        """
        return self.query_tap(adql)
    
    def spitzer_cone(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Search for Spitzer observations."""
        adql = f"""
        SELECT TOP 100 *
        FROM spitzer.seip_source
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60})) = 1
        """
        return self.query_tap(adql)


# =============================================================================
# Simbad Queries
# =============================================================================

class SimbadCatalog:
    """Query Simbad for object information."""
    
    def __init__(self):
        self.tap = TAPClient("https://simbad.u-strasbg.fr/simbad/sim-tap/sync")
    
    def resolve_name(self, name: str) -> QueryResult:
        """Resolve an object name to coordinates."""
        adql = f"""
        SELECT TOP 1 main_id, ra, dec, otype_txt, sp_type, plx_value, pmra, pmdec
        FROM basic
        WHERE main_id = '{name}' OR oid IN (SELECT oidref FROM ident WHERE id = '{name}')
        """
        return self.tap.query(adql)
    
    def cone_search(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Cone search in Simbad."""
        adql = f"""
        SELECT TOP 500 main_id, ra, dec, otype_txt, sp_type, plx_value, pmra, pmdec, rvz_radvel
        FROM basic
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin/60})) = 1
        ORDER BY DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec}))
        """
        return self.tap.query(adql)
    
    def search_by_type(self, object_type: str, limit: int = 100) -> QueryResult:
        """Search for objects by type (e.g., 'BrownD*' for brown dwarfs)."""
        adql = f"""
        SELECT TOP {limit} main_id, ra, dec, otype_txt, sp_type, plx_value, pmra, pmdec
        FROM basic
        WHERE otype_txt LIKE '%{object_type}%'
        ORDER BY plx_value DESC
        """
        return self.tap.query(adql)
    
    def brown_dwarfs(self, limit: int = 500) -> QueryResult:
        """Get known brown dwarfs from Simbad."""
        adql = f"""
        SELECT TOP {limit} main_id, ra, dec, sp_type, plx_value, pmra, pmdec
        FROM basic
        WHERE otype = 'BD*' OR sp_type LIKE 'L%' OR sp_type LIKE 'T%' OR sp_type LIKE 'Y%'
        ORDER BY plx_value DESC
        """
        return self.tap.query(adql)


# =============================================================================
# NED Queries
# =============================================================================

class NEDCatalog:
    """Query NASA/IPAC Extragalactic Database."""
    
    def __init__(self):
        self.base_url = "https://ned.ipac.caltech.edu"
    
    def cone_search(self, ra: float, dec: float, radius_arcmin: float = 5) -> QueryResult:
        """Cone search in NED."""
        url = f"{self.base_url}/cgi-bin/objsearch"
        params = {
            "search_type": "Near Position Search",
            "in_csys": "Equatorial",
            "in_equinox": "J2000.0",
            "lon": f"{ra}d",
            "lat": f"{dec}d",
            "radius": radius_arcmin,
            "out_csys": "Equatorial",
            "out_equinox": "J2000.0",
            "of": "json"
        }
        
        try:
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(full_url, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                return QueryResult(
                    success=True,
                    data=data,
                    total_rows=len(data) if isinstance(data, list) else 1
                )
        except Exception as e:
            return QueryResult(success=False, data=None, error=str(e))


# =============================================================================
# Aladin Image Cutouts
# =============================================================================

class AladinImages:
    """Get image cutouts via Aladin/HiPS."""
    
    SURVEYS = {
        "2mass_j": "CDS/P/2MASS/J",
        "2mass_h": "CDS/P/2MASS/H",
        "2mass_k": "CDS/P/2MASS/K",
        "wise_w1": "CDS/P/allWISE/W1",
        "wise_w2": "CDS/P/allWISE/W2",
        "dss": "CDS/P/DSS2/color",
        "panstarrs": "CDS/P/PanSTARRS/DR1/color",
        "sdss": "CDS/P/SDSS9/color",
        "galex": "CDS/P/GALEXGR6/AIS/color",
    }
    
    def __init__(self):
        self.base_url = "https://alasky.u-strasbg.fr/hips-image-services/hips2fits"
    
    def get_cutout(self, ra: float, dec: float, survey: str = "dss",
                   fov_arcmin: float = 5, width: int = 500) -> QueryResult:
        """Get an image cutout.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            survey: Survey name (see SURVEYS dict)
            fov_arcmin: Field of view in arcminutes
            width: Image width in pixels
        
        Returns:
            QueryResult with base64-encoded image
        """
        hips_id = self.SURVEYS.get(survey.lower(), survey)
        
        params = {
            "hips": hips_id,
            "ra": ra,
            "dec": dec,
            "fov": fov_arcmin / 60,  # Convert to degrees
            "width": width,
            "height": width,
            "format": "png"
        }
        
        try:
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=30) as response:
                image_data = response.read()
                
                return QueryResult(
                    success=True,
                    data={
                        "image_base64": base64.b64encode(image_data).decode('utf-8'),
                        "survey": survey,
                        "ra": ra,
                        "dec": dec,
                        "fov_arcmin": fov_arcmin,
                        "format": "png"
                    }
                )
        except Exception as e:
            return QueryResult(success=False, data=None, error=str(e))
    
    def get_multi_band(self, ra: float, dec: float, 
                       surveys: List[str] = None,
                       fov_arcmin: float = 5) -> QueryResult:
        """Get cutouts from multiple surveys."""
        surveys = surveys or ["dss", "2mass_j", "wise_w1"]
        
        results = {}
        for survey in surveys:
            result = self.get_cutout(ra, dec, survey, fov_arcmin)
            if result.success:
                results[survey] = result.data
        
        return QueryResult(
            success=len(results) > 0,
            data=results,
            total_rows=len(results)
        )


# =============================================================================
# Unified Catalog Interface
# =============================================================================

class CatalogHub:
    """Unified interface to all catalogs."""
    
    def __init__(self):
        self.gaia = GaiaCatalog()
        self.vizier = VizieRCatalog()
        self.mast = MASTCatalog()
        self.irsa = IRSACatalog()
        self.simbad = SimbadCatalog()
        self.ned = NEDCatalog()
        self.aladin = AladinImages()
    
    def query(self, catalog: str, method: str, **kwargs) -> QueryResult:
        """Query any catalog.
        
        Args:
            catalog: Catalog name (gaia, vizier, mast, etc.)
            method: Method to call (cone_search, brown_dwarf_candidates, etc.)
            **kwargs: Arguments for the method
        """
        catalog_obj = getattr(self, catalog.lower(), None)
        if not catalog_obj:
            return QueryResult(success=False, error=f"Unknown catalog: {catalog}")
        
        method_func = getattr(catalog_obj, method, None)
        if not method_func:
            return QueryResult(success=False, error=f"Unknown method: {method}")
        
        return method_func(**kwargs)
    
    def cross_match(self, ra: float, dec: float, radius_arcsec: float = 5) -> Dict[str, QueryResult]:
        """Cross-match position across multiple catalogs."""
        radius_arcmin = radius_arcsec / 60
        
        results = {}
        
        # Gaia
        results["gaia"] = self.gaia.cone_search(ra, dec, radius_arcmin)
        
        # 2MASS
        results["2mass"] = self.vizier.query_2mass(ra, dec, radius_arcmin)
        
        # WISE
        results["wise"] = self.vizier.query_wise(ra, dec, radius_arcmin)
        
        # Simbad
        results["simbad"] = self.simbad.cone_search(ra, dec, radius_arcmin)
        
        return results


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Catalog Query Tools")
    print("=" * 60)
    
    # Test coordinates (a known L dwarf)
    ra, dec = 170.0, 20.0
    
    hub = CatalogHub()
    
    # Test Gaia
    print("\n1. Gaia cone search...")
    result = hub.gaia.cone_search(ra, dec, radius_arcmin=2)
    print(f"   Found {result.total_rows} sources")
    
    # Test 2MASS
    print("\n2. 2MASS cone search...")
    result = hub.vizier.query_2mass(ra, dec, radius_arcmin=2)
    print(f"   Found {result.total_rows} sources")
    
    # Test WISE
    print("\n3. WISE cone search...")
    result = hub.vizier.query_wise(ra, dec, radius_arcmin=2)
    print(f"   Found {result.total_rows} sources")
    
    # Test Simbad
    print("\n4. Simbad cone search...")
    result = hub.simbad.cone_search(ra, dec, radius_arcmin=5)
    print(f"   Found {result.total_rows} sources")
    
    # Test brown dwarf search
    print("\n5. Gaia brown dwarf candidates...")
    result = hub.gaia.brown_dwarf_candidates(max_distance_pc=25)
    print(f"   Found {result.total_rows} candidates")
    
    # Test image cutout
    print("\n6. Aladin image cutout...")
    result = hub.aladin.get_cutout(ra, dec, survey="dss", fov_arcmin=3)
    if result.success:
        print(f"   Got image ({len(result.data['image_base64'])} bytes base64)")
    
    print("\n" + "=" * 60)
    print("Catalog testing complete!")
