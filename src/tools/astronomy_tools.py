"""
AstroLlama Tools Module
Agent tools for astronomical data access, analysis, and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Astronomy imports
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import astropy.units as u

# Astroquery for catalog access
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.mast import Observations, Catalogs
from astroquery.sdss import SDSS
from astroquery.nasa_ads import ADS
from astroquery.xmatch import XMatch

# LangChain tool decorator
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = os.environ.get("ASTRO_OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure Vizier for large queries
Vizier.ROW_LIMIT = 50000

# Configure ADS
ADS.TOKEN = os.environ.get("ADS_DEV_KEY", "")
ADS.NROWS = 20
ADS.ADS_FIELDS = ['bibcode', 'title', 'author', 'year', 'pub', 'abstract', 'citation_count']


# =============================================================================
# INPUT SCHEMAS (for structured tool inputs)
# =============================================================================

class CatalogQueryInput(BaseModel):
    """Input for catalog queries."""
    catalog: str = Field(description="Catalog name: 'gaia', 'sdss', 'vizier', '2mass', 'wise', 'ps1'")
    ra: float = Field(description="Right Ascension in degrees")
    dec: float = Field(description="Declination in degrees")
    radius: float = Field(default=5.0, description="Search radius in arcminutes")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional query filters")


class CrossMatchInput(BaseModel):
    """Input for catalog cross-matching."""
    catalog1: str = Field(description="First catalog or file path")
    catalog2: str = Field(description="Second catalog (VizieR ID or name)")
    max_sep: float = Field(default=2.0, description="Maximum separation in arcseconds")


class PlotCMDInput(BaseModel):
    """Input for color-magnitude diagram plotting."""
    data_source: str = Field(description="Data source: object name, coordinates, or file path")
    color_bands: str = Field(default="BP-RP", description="Color: 'BP-RP', 'g-r', 'B-V', 'J-K'")
    mag_band: str = Field(default="G", description="Magnitude band: 'G', 'r', 'V', 'K'")
    radius: float = Field(default=10.0, description="Search radius in arcminutes")


class LiteratureSearchInput(BaseModel):
    """Input for literature search."""
    query: str = Field(description="Search query for ADS")
    year_range: Optional[str] = Field(default=None, description="Year range, e.g., '2020-2025'")
    first_author: Optional[str] = Field(default=None, description="First author name")
    max_results: int = Field(default=10, description="Maximum number of results")


class ColorCutInput(BaseModel):
    """Input for applying color cuts."""
    data_file: str = Field(description="Path to data file (FITS or CSV)")
    color_col: str = Field(description="Column name for color")
    mag_col: str = Field(description="Column name for magnitude")
    color_min: float = Field(description="Minimum color value")
    color_max: float = Field(description="Maximum color value")
    mag_min: Optional[float] = Field(default=None, description="Minimum magnitude")
    mag_max: Optional[float] = Field(default=None, description="Maximum magnitude")


# =============================================================================
# CATALOG QUERY TOOLS
# =============================================================================

@tool(args_schema=CatalogQueryInput)
def query_catalog(catalog: str, ra: float, dec: float, radius: float = 5.0, 
                  filters: Optional[Dict] = None) -> str:
    """
    Query astronomical catalogs by position.
    
    Supported catalogs: gaia, sdss, 2mass, wise, ps1, vizier
    Returns a summary and saves full results to file.
    """
    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    radius_deg = radius / 60.0
    
    try:
        if catalog.lower() == 'gaia':
            # Gaia DR3 query
            query = f"""
            SELECT TOP 1000 source_id, ra, dec, phot_g_mean_mag, bp_rp, 
                   parallax, pmra, pmdec, ruwe
            FROM gaiadr3.gaia_source
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
            ) = 1
            AND phot_g_mean_mag IS NOT NULL
            ORDER BY phot_g_mean_mag
            """
            job = Gaia.launch_job(query)
            result = job.get_results()
            
        elif catalog.lower() == 'sdss':
            result = SDSS.query_region(
                coord, radius=radius*u.arcmin,
                photoobj_fields=['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'type']
            )
            
        elif catalog.lower() == '2mass':
            v = Vizier(columns=['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Kmag', 'Qflg'])
            result = v.query_region(coord, radius=radius*u.arcmin, catalog='II/246')[0]
            
        elif catalog.lower() == 'wise':
            v = Vizier(columns=['RAJ2000', 'DEJ2000', 'W1mag', 'W2mag', 'W3mag', 'W4mag'])
            result = v.query_region(coord, radius=radius*u.arcmin, catalog='II/328')[0]
            
        elif catalog.lower() == 'ps1':
            v = Vizier(columns=['RAJ2000', 'DEJ2000', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'])
            result = v.query_region(coord, radius=radius*u.arcmin, catalog='II/349')[0]
            
        else:
            # Generic VizieR query
            v = Vizier(columns=['**'])
            result = v.query_region(coord, radius=radius*u.arcmin, catalog=catalog)
            if result:
                result = result[0]
            else:
                return f"No results found in catalog {catalog}"
        
        if result is None or len(result) == 0:
            return f"No sources found within {radius} arcmin of ({ra}, {dec})"
        
        # Save results
        outfile = f"{OUTPUT_DIR}/{catalog}_query_{datetime.now():%Y%m%d_%H%M%S}.fits"
        result.write(outfile, overwrite=True)
        
        # Summary
        summary = f"""
Query Results: {catalog.upper()}
Position: RA={ra:.4f}, Dec={dec:.4f}
Radius: {radius} arcmin
Sources found: {len(result)}
Saved to: {outfile}

First 5 sources:
{result[:5]}
"""
        return summary
        
    except Exception as e:
        return f"Error querying {catalog}: {str(e)}"


@tool
def resolve_object_name(name: str) -> str:
    """
    Resolve an astronomical object name to coordinates using SIMBAD.
    
    Args:
        name: Object name (e.g., 'M31', 'NGC 1234', 'Vega')
    
    Returns:
        Coordinates and basic information about the object.
    """
    try:
        # Custom SIMBAD fields
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('otype', 'flux(V)', 'flux(B)', 'rv_value', 'z_value')
        
        result = custom_simbad.query_object(name)
        
        if result is None:
            return f"Object '{name}' not found in SIMBAD"
        
        coord = SkyCoord(ra=result['RA'][0], dec=result['DEC'][0], 
                         unit=(u.hourangle, u.deg))
        
        info = f"""
Object: {name}
SIMBAD ID: {result['MAIN_ID'][0]}
RA: {coord.ra.deg:.6f} deg ({result['RA'][0]})
Dec: {coord.dec.deg:.6f} deg ({result['DEC'][0]})
Type: {result['OTYPE'][0] if 'OTYPE' in result.colnames else 'Unknown'}
V mag: {result['FLUX_V'][0] if 'FLUX_V' in result.colnames and result['FLUX_V'][0] else 'N/A'}
"""
        return info
        
    except Exception as e:
        return f"Error resolving '{name}': {str(e)}"


@tool(args_schema=CrossMatchInput)
def crossmatch_catalogs(catalog1: str, catalog2: str, max_sep: float = 2.0) -> str:
    """
    Cross-match two catalogs or tables.
    
    Args:
        catalog1: Path to local file (FITS/CSV) with ra, dec columns
        catalog2: VizieR catalog ID (e.g., 'vizier:II/246' for 2MASS)
        max_sep: Maximum separation in arcseconds
    
    Returns:
        Summary of matches and saves results to file.
    """
    try:
        # Load local catalog
        if catalog1.endswith('.fits'):
            table1 = Table.read(catalog1)
        elif catalog1.endswith('.csv'):
            table1 = Table.read(catalog1, format='csv')
        else:
            return f"Unsupported file format for catalog1: {catalog1}"
        
        # Find RA/Dec columns
        ra_col = None
        dec_col = None
        for col in table1.colnames:
            if col.lower() in ['ra', 'raj2000', '_raj2000', 'ra_icrs']:
                ra_col = col
            if col.lower() in ['dec', 'dej2000', '_dej2000', 'de_icrs', 'decl']:
                dec_col = col
        
        if ra_col is None or dec_col is None:
            return f"Could not find RA/Dec columns in {catalog1}. Available: {table1.colnames}"
        
        # Cross-match using XMatch
        result = XMatch.query(
            cat1=table1,
            cat2=catalog2,
            max_distance=max_sep * u.arcsec,
            colRA1=ra_col, 
            colDec1=dec_col
        )
        
        # Save results
        outfile = f"{OUTPUT_DIR}/crossmatch_{datetime.now():%Y%m%d_%H%M%S}.fits"
        result.write(outfile, overwrite=True)
        
        match_rate = len(result) / len(table1) * 100
        
        return f"""
Cross-match Results:
Input catalog: {len(table1)} sources
Matched catalog: {catalog2}
Max separation: {max_sep} arcsec
Matches found: {len(result)} ({match_rate:.1f}%)
Saved to: {outfile}

First 5 matches:
{result[:5]}
"""
    except Exception as e:
        return f"Error in cross-match: {str(e)}"


# =============================================================================
# PLOTTING TOOLS
# =============================================================================

@tool(args_schema=PlotCMDInput)
def plot_color_magnitude_diagram(data_source: str, color_bands: str = "BP-RP", 
                                  mag_band: str = "G", radius: float = 10.0) -> str:
    """
    Create a color-magnitude diagram.
    
    Args:
        data_source: Object name, 'ra,dec' coordinates, or path to FITS file
        color_bands: Color to plot (BP-RP, g-r, B-V, J-K)
        mag_band: Magnitude band (G, r, V, K)
        radius: Search radius in arcminutes (for name/coord queries)
    
    Returns:
        Path to saved plot and summary statistics.
    """
    try:
        # Determine data source type
        if os.path.exists(data_source):
            # Load from file
            data = Table.read(data_source)
        elif ',' in data_source:
            # Coordinates
            ra, dec = map(float, data_source.split(','))
            data = _query_for_cmd(ra, dec, radius, color_bands)
        else:
            # Object name
            coord = SkyCoord.from_name(data_source)
            data = _query_for_cmd(coord.ra.deg, coord.dec.deg, radius, color_bands)
        
        # Extract color and magnitude columns
        color, mag, color_label, mag_label = _get_cmd_columns(data, color_bands, mag_band)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Plot with density coloring for large datasets
        if len(data) > 1000:
            from scipy.stats import gaussian_kde
            xy = np.vstack([color, mag])
            mask = np.isfinite(xy).all(axis=0)
            z = gaussian_kde(xy[:, mask])(xy[:, mask])
            idx = z.argsort()
            ax.scatter(color[mask][idx], mag[mask][idx], c=z[idx], s=1, cmap='viridis')
        else:
            ax.scatter(color, mag, s=5, alpha=0.5, c='blue')
        
        ax.set_xlabel(color_label, fontsize=12)
        ax.set_ylabel(mag_label, fontsize=12)
        ax.set_title(f'Color-Magnitude Diagram: {data_source}', fontsize=14)
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save
        outfile = f"{OUTPUT_DIR}/cmd_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Statistics
        stats = f"""
CMD Generated: {outfile}
Data source: {data_source}
Total sources: {len(data)}
Color range ({color_label}): [{np.nanmin(color):.2f}, {np.nanmax(color):.2f}]
Magnitude range ({mag_label}): [{np.nanmin(mag):.2f}, {np.nanmax(mag):.2f}]
"""
        return stats
        
    except Exception as e:
        return f"Error creating CMD: {str(e)}"


def _query_for_cmd(ra: float, dec: float, radius: float, color_bands: str) -> Table:
    """Helper to query appropriate catalog for CMD."""
    if 'BP' in color_bands or 'G' in color_bands.upper():
        # Gaia query
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, 
               phot_rp_mean_mag, bp_rp, parallax
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius/60})
        ) = 1
        AND phot_g_mean_mag IS NOT NULL
        AND bp_rp IS NOT NULL
        """
        job = Gaia.launch_job(query)
        return job.get_results()
    else:
        # SDSS query
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        return SDSS.query_region(coord, radius=radius*u.arcmin,
                                 photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z'])


def _get_cmd_columns(data: Table, color_bands: str, mag_band: str):
    """Extract appropriate columns for CMD."""
    color_map = {
        'BP-RP': ('bp_rp', 'BP - RP'),
        'G-RP': (lambda d: d['phot_g_mean_mag'] - d['phot_rp_mean_mag'], 'G - RP'),
        'g-r': (lambda d: d['g'] - d['r'], 'g - r'),
        'r-i': (lambda d: d['r'] - d['i'], 'r - i'),
        'u-g': (lambda d: d['u'] - d['g'], 'u - g'),
    }
    
    mag_map = {
        'G': ('phot_g_mean_mag', 'G [mag]'),
        'g': ('g', 'g [mag]'),
        'r': ('r', 'r [mag]'),
    }
    
    # Get color
    if color_bands in color_map:
        c = color_map[color_bands]
        if callable(c[0]):
            color = c[0](data)
        else:
            color = data[c[0]]
        color_label = c[1]
    else:
        color_label = color_bands
        color = data[color_bands]
    
    # Get magnitude
    if mag_band in mag_map:
        m = mag_map[mag_band]
        mag = data[m[0]]
        mag_label = m[1]
    else:
        mag = data[mag_band]
        mag_label = f"{mag_band} [mag]"
    
    return np.array(color), np.array(mag), color_label, mag_label


@tool
def plot_sky_positions(data_file: str, ra_col: str = "ra", dec_col: str = "dec",
                       projection: str = "aitoff") -> str:
    """
    Plot sky positions of sources on an all-sky map.
    
    Args:
        data_file: Path to FITS or CSV file with coordinates
        ra_col: Column name for RA
        dec_col: Column name for Dec
        projection: 'aitoff', 'mollweide', or 'hammer'
    
    Returns:
        Path to saved plot.
    """
    try:
        # Load data
        if data_file.endswith('.fits'):
            data = Table.read(data_file)
        else:
            data = Table.read(data_file, format='csv')
        
        ra = np.array(data[ra_col])
        dec = np.array(data[dec_col])
        
        # Convert to radians, wrap RA
        ra_rad = np.deg2rad(ra)
        ra_rad[ra_rad > np.pi] -= 2 * np.pi
        dec_rad = np.deg2rad(dec)
        
        # Create plot
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection=projection)
        
        ax.scatter(ra_rad, dec_rad, s=1, alpha=0.5, c='blue')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sky Distribution ({len(data)} sources)', fontsize=14)
        
        outfile = f"{OUTPUT_DIR}/sky_map_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"Sky map saved to: {outfile}"
        
    except Exception as e:
        return f"Error creating sky map: {str(e)}"


# =============================================================================
# LITERATURE TOOLS
# =============================================================================

@tool(args_schema=LiteratureSearchInput)
def search_literature(query: str, year_range: Optional[str] = None,
                      first_author: Optional[str] = None, max_results: int = 10) -> str:
    """
    Search NASA ADS for astronomical literature.
    
    Args:
        query: Search query (e.g., 'exoplanet atmosphere JWST')
        year_range: Optional year range (e.g., '2020-2025')
        first_author: Optional first author name
        max_results: Maximum number of results
    
    Returns:
        List of papers with titles, authors, and citations.
    """
    if not ADS.TOKEN:
        return "Error: ADS_DEV_KEY not configured. Set it in environment variables."
    
    try:
        # Build query
        full_query = query
        if year_range:
            full_query += f" year:{year_range}"
        if first_author:
            full_query += f" ^{first_author}"
        
        ADS.NROWS = max_results
        ADS.SORT = 'citation_count desc'
        
        results = ADS.query_simple(full_query)
        
        if results is None or len(results) == 0:
            return f"No papers found for query: {full_query}"
        
        # Format results
        output = [f"Found {len(results)} papers for: {full_query}\n"]
        output.append("=" * 60)
        
        for i, paper in enumerate(results):
            authors = paper['author']
            if len(authors) > 3:
                author_str = f"{authors[0]} et al."
            else:
                author_str = "; ".join(authors)
            
            output.append(f"""
{i+1}. {paper['title'][0]}
   Authors: {author_str}
   Year: {paper['year']}, {paper['pub']}
   Citations: {paper['citation_count']}
   Bibcode: {paper['bibcode']}
""")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error searching ADS: {str(e)}"


@tool
def generate_bibtex(bibcodes: str) -> str:
    """
    Generate BibTeX entries for papers.
    
    Args:
        bibcodes: Comma-separated list of ADS bibcodes
    
    Returns:
        BibTeX entries for all papers.
    """
    if not ADS.TOKEN:
        return "Error: ADS_DEV_KEY not configured."
    
    try:
        bibcode_list = [b.strip() for b in bibcodes.split(',')]
        
        all_bibtex = []
        for bibcode in bibcode_list:
            # Query for the paper
            ADS.NROWS = 1
            ADS.ADS_FIELDS = ['bibcode', 'title', 'author', 'year', 'pub', 'volume', 'page']
            
            # Get BibTeX export URL
            import requests
            url = f"https://api.adsabs.harvard.edu/v1/export/bibtex"
            headers = {"Authorization": f"Bearer {ADS.TOKEN}"}
            data = {"bibcode": [bibcode]}
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                all_bibtex.append(response.json().get('export', ''))
            else:
                all_bibtex.append(f"% Error getting BibTeX for {bibcode}")
        
        # Save to file
        outfile = f"{OUTPUT_DIR}/references_{datetime.now():%Y%m%d_%H%M%S}.bib"
        with open(outfile, 'w') as f:
            f.write("\n\n".join(all_bibtex))
        
        return f"BibTeX saved to: {outfile}\n\n" + "\n\n".join(all_bibtex)
        
    except Exception as e:
        return f"Error generating BibTeX: {str(e)}"


# =============================================================================
# DATA ANALYSIS TOOLS
# =============================================================================

@tool(args_schema=ColorCutInput)
def apply_color_cut(data_file: str, color_col: str, mag_col: str,
                    color_min: float, color_max: float,
                    mag_min: Optional[float] = None, 
                    mag_max: Optional[float] = None) -> str:
    """
    Apply color and magnitude cuts to select sources.
    
    Args:
        data_file: Path to input data file
        color_col: Column name for color
        mag_col: Column name for magnitude
        color_min, color_max: Color range
        mag_min, mag_max: Optional magnitude range
    
    Returns:
        Summary and path to filtered catalog.
    """
    try:
        # Load data
        if data_file.endswith('.fits'):
            data = Table.read(data_file)
        else:
            data = Table.read(data_file, format='csv')
        
        original_count = len(data)
        
        # Apply color cut
        mask = (data[color_col] >= color_min) & (data[color_col] <= color_max)
        
        # Apply magnitude cuts if specified
        if mag_min is not None:
            mask &= (data[mag_col] >= mag_min)
        if mag_max is not None:
            mask &= (data[mag_col] <= mag_max)
        
        filtered = data[mask]
        
        # Save filtered catalog
        outfile = f"{OUTPUT_DIR}/color_cut_{datetime.now():%Y%m%d_%H%M%S}.fits"
        filtered.write(outfile, overwrite=True)
        
        return f"""
Color Cut Applied:
Input: {original_count} sources
Selection: {color_min} < {color_col} < {color_max}
{"Magnitude: " + str(mag_min) + " < " + mag_col + " < " + str(mag_max) if mag_min or mag_max else ""}
Selected: {len(filtered)} sources ({100*len(filtered)/original_count:.1f}%)
Saved to: {outfile}
"""
    except Exception as e:
        return f"Error applying color cut: {str(e)}"


@tool
def compute_statistics(data_file: str, columns: str) -> str:
    """
    Compute statistics for columns in a data file.
    
    Args:
        data_file: Path to FITS or CSV file
        columns: Comma-separated list of column names
    
    Returns:
        Statistics for each column.
    """
    try:
        if data_file.endswith('.fits'):
            data = Table.read(data_file)
        else:
            data = Table.read(data_file, format='csv')
        
        col_list = [c.strip() for c in columns.split(',')]
        
        stats = ["Statistics:\n" + "=" * 50]
        
        for col in col_list:
            if col not in data.colnames:
                stats.append(f"\n{col}: Column not found")
                continue
            
            values = np.array(data[col])
            values = values[np.isfinite(values)]
            
            mean, median, std = sigma_clipped_stats(values, sigma=3.0)
            
            stats.append(f"""
{col}:
  N: {len(values)}
  Mean: {mean:.4f}
  Median: {median:.4f}
  Std: {std:.4f}
  Min: {np.min(values):.4f}
  Max: {np.max(values):.4f}
  25%: {np.percentile(values, 25):.4f}
  75%: {np.percentile(values, 75):.4f}
""")
        
        return "\n".join(stats)
        
    except Exception as e:
        return f"Error computing statistics: {str(e)}"


# =============================================================================
# IMAGE PROCESSING TOOLS
# =============================================================================

@tool
def extract_sources_from_image(fits_file: str, threshold: float = 5.0,
                                fwhm: float = 3.0) -> str:
    """
    Extract sources from a FITS image using photutils.
    
    Args:
        fits_file: Path to FITS image
        threshold: Detection threshold in sigma
        fwhm: Expected FWHM of sources in pixels
    
    Returns:
        Summary and path to source catalog.
    """
    try:
        from photutils.detection import DAOStarFinder
        from photutils.background import Background2D, MedianBackground
        
        # Load image
        with fits.open(fits_file) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
        
        # Background estimation
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                          bkg_estimator=MedianBackground())
        data_sub = data - bkg.background
        
        # Source detection
        mean, median, std = sigma_clipped_stats(data_sub, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
        sources = daofind(data_sub)
        
        if sources is None:
            return "No sources detected"
        
        # Add WCS coordinates if available
        try:
            wcs = WCS(header)
            coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
            sources['ra'] = coords.ra.deg
            sources['dec'] = coords.dec.deg
        except:
            pass
        
        # Save catalog
        outfile = f"{OUTPUT_DIR}/sources_{datetime.now():%Y%m%d_%H%M%S}.fits"
        sources.write(outfile, overwrite=True)
        
        return f"""
Source Extraction Complete:
Image: {fits_file}
Background: {bkg.background_median:.2f} +/- {bkg.background_rms_median:.2f}
Detection threshold: {threshold} sigma
Sources found: {len(sources)}
Saved to: {outfile}

Brightest 5 sources:
{sources[:5]}
"""
    except Exception as e:
        return f"Error extracting sources: {str(e)}"


@tool
def download_archive_image(target: str, archive: str = "mast",
                           radius: float = 1.0) -> str:
    """
    Download images from astronomical archives.
    
    Args:
        target: Object name or 'ra,dec' coordinates
        archive: Archive to query: 'mast', 'esa', 'sdss'
        radius: Search radius in arcminutes
    
    Returns:
        Download status and file paths.
    """
    try:
        # Parse target
        if ',' in target:
            ra, dec = map(float, target.split(','))
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        else:
            coord = SkyCoord.from_name(target)
        
        if archive.lower() == 'mast':
            # Query MAST
            obs = Observations.query_criteria(
                coordinates=coord,
                radius=radius * u.arcmin,
                obs_collection=['HST', 'JWST'],
                dataproduct_type='image',
                intentType='science'
            )
            
            if len(obs) == 0:
                return f"No images found for {target} in MAST"
            
            # Get data products for first observation
            products = Observations.get_product_list(obs[:1])
            drz = Observations.filter_products(
                products, 
                productSubGroupDescription=['DRZ', 'DRC', 'I2D']
            )
            
            if len(drz) == 0:
                return "No drizzled products available"
            
            # Download
            download_dir = f"{OUTPUT_DIR}/archive_data"
            os.makedirs(download_dir, exist_ok=True)
            
            manifest = Observations.download_products(
                drz[:1], download_dir=download_dir
            )
            
            return f"""
Download complete:
Target: {target}
Archive: {archive}
Observations found: {len(obs)}
Downloaded: {manifest['Local Path'][0]}
"""
        
        elif archive.lower() == 'sdss':
            # SDSS image cutout
            from astroquery.sdss import SDSS
            
            images = SDSS.get_images(coord, radius=radius*u.arcmin, band='gri')
            
            if images is None:
                return f"No SDSS images found for {target}"
            
            # Save first image
            outfile = f"{OUTPUT_DIR}/sdss_{target.replace(' ', '_')}.fits"
            images[0].writeto(outfile, overwrite=True)
            
            return f"SDSS image saved to: {outfile}"
        
        else:
            return f"Archive '{archive}' not supported. Use 'mast' or 'sdss'."
            
    except Exception as e:
        return f"Error downloading image: {str(e)}"


# =============================================================================
# TOOL REGISTRY
# =============================================================================

ALL_TOOLS = [
    query_catalog,
    resolve_object_name,
    crossmatch_catalogs,
    plot_color_magnitude_diagram,
    plot_sky_positions,
    search_literature,
    generate_bibtex,
    apply_color_cut,
    compute_statistics,
    extract_sources_from_image,
    download_archive_image,
]


def get_tools(tool_names: Optional[List[str]] = None):
    """Get a subset of tools by name, or all tools if no names specified."""
    if tool_names is None:
        return ALL_TOOLS
    
    name_to_tool = {t.name: t for t in ALL_TOOLS}
    return [name_to_tool[n] for n in tool_names if n in name_to_tool]
