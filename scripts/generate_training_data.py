#!/usr/bin/env python3
"""
AstroLlama - Training Data Generator
Generates high-quality astronomy Q&A pairs for fine-tuning.

Usage:
    python generate_training_data.py --output ./data/training/
    python generate_training_data.py --output ./data/training/ --count 1000
    python generate_training_data.py --output ./data/training/ --categories catalog,plotting

Categories:
    - catalog: Gaia, SDSS, 2MASS, VizieR queries
    - reduction: Photometry, calibration, astrometry
    - plotting: CMD, SED, light curves, sky maps
    - literature: ADS search, citations, paper summaries
    - selection: Star/galaxy classification, color cuts
    - concepts: Astronomy fundamentals, definitions
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are AstroLlama, an expert astronomy research assistant specializing in:
- Querying astronomical catalogs (Gaia, SDSS, 2MASS, WISE, Pan-STARRS)
- Data analysis and reduction techniques
- Creating publication-quality plots
- Literature search and citation management
- Stellar and galactic classification

You provide accurate, practical code examples using astropy, astroquery, and standard Python data science libraries. Your responses are concise but complete, with working code that researchers can use directly."""

# =============================================================================
# Training Data Templates
# =============================================================================

CATALOG_QUERIES = [
    # Gaia queries
    {
        "user": "How do I query Gaia DR3 for stars within 100 pc of the Sun?",
        "assistant": """To query Gaia DR3 for nearby stars within 100 pc, use the parallax criterion (parallax > 10 mas):

```python
from astroquery.gaia import Gaia

query = \"\"\"
SELECT source_id, ra, dec, parallax, parallax_error,
       phot_g_mean_mag, bp_rp, pmra, pmdec
FROM gaiadr3.gaia_source
WHERE parallax > 10 
  AND parallax_error/parallax < 0.1
  AND phot_g_mean_mag < 15
\"\"\"

job = Gaia.launch_job_async(query)
results = job.get_results()
print(f"Found {len(results)} stars within 100 pc")
```

Key points:
- `parallax > 10` means distance < 100 pc (d = 1000/parallax)
- Added quality cut on parallax error (<10%)
- Limited to G < 15 to avoid faint sources with poor astrometry"""
    },
    {
        "user": "Query Gaia for the Pleiades cluster",
        "assistant": """Here's how to query Gaia DR3 for Pleiades members:

```python
from astroquery.gaia import Gaia

# Pleiades center: RA=56.75°, Dec=24.12°, radius ~2°
query = \"\"\"
SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec), 
               CIRCLE('ICRS', 56.75, 24.12, 2)) = 1
  AND parallax BETWEEN 7.0 AND 7.8
  AND pmra BETWEEN 19 AND 21
  AND pmdec BETWEEN -46 AND -44
\"\"\"

job = Gaia.launch_job_async(query)
pleiades = job.get_results()
print(f"Found {len(pleiades)} Pleiades candidates")
```

The proper motion and parallax cuts isolate cluster members from field stars. Pleiades has:
- Distance ~136 pc (parallax ~7.4 mas)
- Proper motion: (μα*, μδ) ≈ (20, -45) mas/yr"""
    },
    {
        "user": "How do I cross-match Gaia with 2MASS?",
        "assistant": """Gaia DR3 includes pre-computed cross-matches with 2MASS. Use the `tmass_psc_xsc_best_neighbour` table:

```python
from astroquery.gaia import Gaia

query = \"\"\"
SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
       t.j_m, t.h_m, t.k_m, t.ph_qual
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm
  ON g.source_id = xm.source_id
JOIN gaiadr1.tmass_original_valid AS t
  ON xm.original_ext_source_id = t.tmass_oid
WHERE CONTAINS(POINT('ICRS', g.ra, g.dec),
               CIRCLE('ICRS', 180.0, 45.0, 1.0)) = 1
  AND t.ph_qual = 'AAA'
\"\"\"

job = Gaia.launch_job_async(query)
results = job.get_results()
```

The `ph_qual = 'AAA'` ensures high-quality JHK photometry. You can then compute colors like J-K or create optical-infrared CMDs."""
    },
    {
        "user": "Get SDSS photometry for a list of coordinates",
        "assistant": """Use astroquery to query SDSS for multiple coordinates:

```python
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Your coordinates
coords = SkyCoord(
    ra=[180.0, 180.5, 181.0],
    dec=[45.0, 45.1, 45.2],
    unit=(u.deg, u.deg)
)

# Query SDSS - searches within 2 arcsec by default
results = SDSS.query_crossid(coords, 
                              photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                                              'err_u', 'err_g', 'err_r', 'err_i', 'err_z',
                                              'type'])

# Filter for stars (type=6) or galaxies (type=3)
stars = results[results['type'] == 6]
print(f"Found {len(stars)} stars in SDSS")
```

For large lists (>1000 objects), consider using SDSS CasJobs for batch queries."""
    },
    {
        "user": "Query VizieR for a specific catalog",
        "assistant": """Here's how to query any VizieR catalog using astroquery:

```python
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

# Configure Vizier
Vizier.ROW_LIMIT = -1  # No row limit

# Query a specific catalog (e.g., APASS DR9 = II/336/apass9)
result = Vizier.query_region(
    SkyCoord(ra=180.0, dec=45.0, unit=(u.deg, u.deg)),
    radius=0.5*u.deg,
    catalog='II/336/apass9'
)

if result:
    apass = result[0]
    print(f"Found {len(apass)} sources")
    print(apass.colnames)  # See available columns

# Query multiple catalogs at once
catalogs = Vizier.query_region(
    SkyCoord(ra=180.0, dec=45.0, unit=(u.deg, u.deg)),
    radius=10*u.arcmin,
    catalog=['II/336/apass9', 'II/246/out']  # APASS + 2MASS
)
```

Find catalog IDs at: https://vizier.u-strasbg.fr/"""
    },
    {
        "user": "How do I query the WISE catalog?",
        "assistant": """Query WISE (AllWISE) through VizieR or IRSA:

```python
# Method 1: Via VizieR
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

Vizier.ROW_LIMIT = 10000

result = Vizier.query_region(
    SkyCoord(ra=180.0, dec=45.0, unit=(u.deg, u.deg)),
    radius=0.5*u.deg,
    catalog='II/328/allwise'  # AllWISE catalog
)

if result:
    wise = result[0]
    # Key columns: W1mag, W2mag, W3mag, W4mag
    print(wise['RAJ2000', 'DEJ2000', 'W1mag', 'W2mag'])

# Method 2: Via IRSA (for large queries)
from astroquery.ipac.irsa import Irsa

wise_data = Irsa.query_region(
    SkyCoord(ra=180.0, dec=45.0, unit='deg'),
    catalog='allwise_p3as_psd',
    radius=0.5*u.deg
)
```

WISE bands: W1 (3.4μm), W2 (4.6μm), W3 (12μm), W4 (22μm)."""
    },
    {
        "user": "Download Gaia data for a very large area",
        "assistant": """For large areas, use asynchronous queries and save results directly:

```python
from astroquery.gaia import Gaia
import time

# Login for larger queries (optional but recommended)
# Gaia.login(user='your_username', password='your_password')

# Large area query - use async
query = \"\"\"
SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, bp_rp, radial_velocity
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec),
               BOX('ICRS', 180, 0, 10, 10)) = 1
  AND phot_g_mean_mag < 18
  AND parallax IS NOT NULL
\"\"\"

# Launch async job
job = Gaia.launch_job_async(query, 
                            dump_to_file=True,
                            output_file='gaia_large_query.vot')

# Monitor progress
while job.is_finished() == False:
    print(f"Job status: {job.get_phase()}")
    time.sleep(30)

print(f"Job completed! Results in: gaia_large_query.vot")

# Load results
from astropy.table import Table
data = Table.read('gaia_large_query.vot')
```

For full-sky queries, consider using the Gaia bulk download or datalink service."""
    },
    {
        "user": "How to query Pan-STARRS catalog?",
        "assistant": """Query Pan-STARRS DR2 using the MAST archive:

```python
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
import astropy.units as u

# Query Pan-STARRS
coords = SkyCoord(ra=180.0, dec=45.0, unit='deg')

ps1 = Catalogs.query_region(
    coords,
    radius=0.1*u.deg,
    catalog='Panstarrs',
    data_release='dr2',
    table='stack'  # Stacked photometry
)

# Key columns: gMeanPSFMag, rMeanPSFMag, iMeanPSFMag, zMeanPSFMag, yMeanPSFMag
print(f"Found {len(ps1)} sources")
print(ps1['objID', 'raMean', 'decMean', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag'])

# For forced photometry on known positions
forced = Catalogs.query_region(
    coords,
    radius=0.1*u.deg,
    catalog='Panstarrs',
    data_release='dr2',
    table='forced_mean'
)
```

Pan-STARRS covers 3π steradians (Dec > -30°) in grizy bands."""
    },
    {
        "user": "Query Gaia for high proper motion stars",
        "assistant": """Find high proper motion stars (potential nearby stars or halo stars):

```python
from astroquery.gaia import Gaia
import numpy as np

query = \"\"\"
SELECT source_id, ra, dec, parallax, parallax_error,
       pmra, pmra_error, pmdec, pmdec_error,
       phot_g_mean_mag, bp_rp,
       SQRT(pmra*pmra + pmdec*pmdec) AS total_pm
FROM gaiadr3.gaia_source
WHERE SQRT(pmra*pmra + pmdec*pmdec) > 500
  AND parallax_over_error > 5
  AND phot_g_mean_mag < 18
ORDER BY total_pm DESC
\"\"\"

job = Gaia.launch_job_async(query)
hpm = job.get_results()

print(f"Found {len(hpm)} stars with PM > 500 mas/yr")

# Compute tangential velocity (km/s)
# v_t = 4.74 * PM(mas/yr) / parallax(mas)
hpm['v_tan'] = 4.74 * hpm['total_pm'] / hpm['parallax']
print(hpm['source_id', 'total_pm', 'parallax', 'v_tan'][:10])
```

Stars with PM > 500 mas/yr are typically within 50 pc and/or have high space velocities."""
    },
    {
        "user": "Query for RR Lyrae variables in Gaia",
        "assistant": """Gaia DR3 includes variability classifications. Query RR Lyrae stars:

```python
from astroquery.gaia import Gaia

# Query RR Lyrae from the variability tables
query = \"\"\"
SELECT g.source_id, g.ra, g.dec, g.parallax,
       g.phot_g_mean_mag, g.bp_rp,
       v.pf, v.p1_o, v.average_g_mag, v.peak_to_peak_g
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.vari_rrlyrae AS v
  ON g.source_id = v.source_id
WHERE g.parallax > 0.5
  AND g.parallax_over_error > 5
\"\"\"

job = Gaia.launch_job_async(query)
rrl = job.get_results()

print(f"Found {len(rrl)} RR Lyrae stars")

# Columns from vari_rrlyrae:
# - pf: fundamental period
# - p1_o: first overtone period  
# - peak_to_peak_g: G-band amplitude

# Calculate absolute magnitude using period-luminosity relation
# M_G ≈ -1.0 - 2.3*log10(P) for RRab (fundamental mode)
import numpy as np
rrl['M_G'] = -1.0 - 2.3 * np.log10(rrl['pf'])
```

You can also query `vari_cepheid`, `vari_eclipsing_binary`, etc."""
    },
]

PLOTTING_EXAMPLES = [
    {
        "user": "Create a color-magnitude diagram from Gaia data",
        "assistant": """Here's how to create a publication-quality CMD:

```python
import matplotlib.pyplot as plt
import numpy as np
from astroquery.gaia import Gaia

# Query data
query = \"\"\"
SELECT phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec),
               CIRCLE('ICRS', 56.75, 24.12, 1.5)) = 1
  AND phot_g_mean_mag < 20
  AND bp_rp IS NOT NULL
  AND parallax BETWEEN 7.0 AND 7.8
\"\"\"
job = Gaia.launch_job(query)
data = job.get_results()

# Create CMD
fig, ax = plt.subplots(figsize=(8, 10))

ax.scatter(data['bp_rp'], data['phot_g_mean_mag'], 
           s=2, alpha=0.5, c='black')

ax.set_xlabel('BP - RP [mag]', fontsize=12)
ax.set_ylabel('G [mag]', fontsize=12)
ax.set_title('Pleiades CMD (Gaia DR3)', fontsize=14)
ax.invert_yaxis()
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(20, 2)

plt.tight_layout()
plt.savefig('pleiades_cmd.png', dpi=300, bbox_inches='tight')
plt.show()
```"""
    },
    {
        "user": "Make a density plot CMD with hexbins",
        "assistant": """For large datasets, use hexbin or 2D histogram for better visualization:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Assuming 'data' has bp_rp and phot_g_mean_mag columns
color = data['bp_rp']
mag = data['phot_g_mean_mag']

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Hexbin plot
ax1 = axes[0]
hb = ax1.hexbin(color, mag, gridsize=100, cmap='viridis', 
                norm=LogNorm(), mincnt=1)
ax1.invert_yaxis()
ax1.set_xlabel('BP - RP [mag]')
ax1.set_ylabel('G [mag]')
ax1.set_title('CMD - Hexbin')
plt.colorbar(hb, ax=ax1, label='Count')

# 2D histogram with contours
ax2 = axes[1]
h, xedges, yedges = np.histogram2d(color, mag, bins=200)
extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]

im = ax2.imshow(h.T, extent=extent, aspect='auto', 
                cmap='magma', norm=LogNorm(), origin='upper')
ax2.set_xlabel('BP - RP [mag]')
ax2.set_ylabel('G [mag]')
ax2.set_title('CMD - 2D Histogram')
plt.colorbar(im, ax=ax2, label='Count')

plt.tight_layout()
plt.savefig('cmd_density.png', dpi=300)
plt.show()
```"""
    },
    {
        "user": "Plot proper motion diagram",
        "assistant": """Create a proper motion vector-point diagram:

```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming data has pmra, pmdec columns
pmra = data['pmra']
pmdec = data['pmdec']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Vector-point diagram (scatter)
ax1 = axes[0]
ax1.scatter(pmra, pmdec, s=1, alpha=0.3, c='blue')
ax1.set_xlabel(r'$\\mu_{\\alpha*}$ [mas/yr]', fontsize=12)
ax1.set_ylabel(r'$\\mu_\\delta$ [mas/yr]', fontsize=12)
ax1.set_title('Proper Motion Diagram')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)

# Density version
ax2 = axes[1]
hb = ax2.hexbin(pmra, pmdec, gridsize=100, cmap='hot_r', mincnt=1)
ax2.set_xlabel(r'$\\mu_{\\alpha*}$ [mas/yr]', fontsize=12)
ax2.set_ylabel(r'$\\mu_\\delta$ [mas/yr]', fontsize=12)
ax2.set_title('Proper Motion Density')
ax2.set_xlim(-50, 50)
ax2.set_ylim(-50, 50)
plt.colorbar(hb, ax=ax2, label='Count')

plt.tight_layout()
plt.savefig('proper_motion.png', dpi=300)
plt.show()
```"""
    },
    {
        "user": "Create a sky map in galactic coordinates",
        "assistant": """Plot sources on an all-sky Aitoff projection:

```python
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# Convert to galactic coordinates
coords = SkyCoord(ra=data['ra']*u.deg, dec=data['dec']*u.deg, frame='icrs')
galactic = coords.galactic

# Convert to radians for projection (-π to π for longitude)
l_rad = galactic.l.wrap_at(180*u.deg).radian
b_rad = galactic.b.radian

# Create Aitoff projection
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='aitoff')

# Scatter plot
sc = ax.scatter(l_rad, b_rad, s=1, alpha=0.3, c=data['phot_g_mean_mag'], 
                cmap='viridis_r', vmin=10, vmax=18)

ax.set_xlabel('Galactic Longitude')
ax.set_ylabel('Galactic Latitude')
ax.grid(True, alpha=0.3)
ax.set_title('Sky Distribution (Galactic Coordinates)')

cbar = plt.colorbar(sc, orientation='horizontal', pad=0.1, aspect=40)
cbar.set_label('G magnitude')

plt.tight_layout()
plt.savefig('sky_map_galactic.png', dpi=300)
plt.show()
```"""
    },
    {
        "user": "Plot a light curve",
        "assistant": """Here's how to plot a light curve with error bars:

```python
import matplotlib.pyplot as plt
import numpy as np

# Example: light curve data (time, magnitude, error)
time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # days
mag = np.array([15.0, 14.8, 14.5, 14.2, 14.0, 14.2, 14.5, 14.8, 15.0, 14.8, 14.5])
mag_err = np.array([0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03])

fig, ax = plt.subplots(figsize=(10, 6))

# Plot with error bars
ax.errorbar(time, mag, yerr=mag_err, fmt='o', markersize=6,
            capsize=3, elinewidth=1, color='blue', label='V-band')

# Invert y-axis (brighter = smaller magnitude)
ax.invert_yaxis()

ax.set_xlabel('Time [days]', fontsize=12)
ax.set_ylabel('Magnitude', fontsize=12)
ax.set_title('Variable Star Light Curve', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('light_curve.png', dpi=300)
plt.show()

# For phase-folded light curve
period = 3.5  # days
phase = (time % period) / period

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(phase, mag, yerr=mag_err, fmt='o', markersize=6, capsize=3)
ax.errorbar(phase + 1, mag, yerr=mag_err, fmt='o', markersize=6, 
            capsize=3, alpha=0.3)  # Repeat for clarity
ax.invert_yaxis()
ax.set_xlabel('Phase', fontsize=12)
ax.set_ylabel('Magnitude', fontsize=12)
ax.set_title(f'Phase-Folded Light Curve (P = {period} days)')
ax.set_xlim(0, 2)
plt.tight_layout()
plt.savefig('light_curve_phased.png', dpi=300)
```"""
    },
    {
        "user": "Create an SED plot",
        "assistant": """Plot a spectral energy distribution (SED):

```python
import matplotlib.pyplot as plt
import numpy as np

# Example photometry (wavelength in microns, flux in mJy)
# Typical filters: u, g, r, i, z, J, H, K, W1, W2
wavelengths = np.array([0.35, 0.48, 0.62, 0.76, 0.87, 1.24, 1.66, 2.16, 3.4, 4.6])
filter_names = ['u', 'g', 'r', 'i', 'z', 'J', 'H', 'K', 'W1', 'W2']
flux_mJy = np.array([0.5, 1.2, 2.1, 2.8, 3.2, 4.5, 5.1, 4.8, 3.2, 2.1])
flux_err = flux_mJy * 0.05  # 5% errors

# Convert to νFν (common SED representation)
# νFν = λFλ in same units
freq = 3e14 / wavelengths  # Hz (c/λ)
nu_fnu = freq * flux_mJy * 1e-26  # erg/s/cm²

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Flux vs wavelength
ax1 = axes[0]
ax1.errorbar(wavelengths, flux_mJy, yerr=flux_err, fmt='o', 
             markersize=8, capsize=3, color='blue')
for i, name in enumerate(filter_names):
    ax1.annotate(name, (wavelengths[i], flux_mJy[i]), 
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
ax1.set_xlabel('Wavelength [μm]', fontsize=12)
ax1.set_ylabel('Flux [mJy]', fontsize=12)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('SED: Flux vs Wavelength')
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: νFν vs frequency (standard SED plot)
ax2 = axes[1]
ax2.errorbar(freq, nu_fnu, fmt='o', markersize=8, capsize=3, color='red')
ax2.set_xlabel('Frequency [Hz]', fontsize=12)
ax2.set_ylabel(r'$\\nu F_\\nu$ [erg s$^{-1}$ cm$^{-2}$]', fontsize=12)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('SED: νFν vs Frequency')
ax2.grid(True, alpha=0.3, which='both')
ax2.invert_xaxis()  # Higher frequency (bluer) on left

plt.tight_layout()
plt.savefig('sed_plot.png', dpi=300)
plt.show()
```"""
    },
    {
        "user": "Make a 2D histogram of stellar parameters",
        "assistant": """Create a Kiel diagram (log g vs Teff) or similar parameter plot:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Example: Teff and log_g from spectroscopic survey
teff = np.random.normal(5500, 800, 10000)  # Example data
logg = np.random.normal(4.0, 0.8, 10000)
logg = np.clip(logg, 0, 5.5)

# Add some giants
teff = np.concatenate([teff, np.random.normal(4800, 400, 2000)])
logg = np.concatenate([logg, np.random.normal(2.5, 0.5, 2000)])

fig, ax = plt.subplots(figsize=(10, 8))

# 2D histogram
h = ax.hist2d(teff, logg, bins=100, cmap='viridis', norm=LogNorm())
plt.colorbar(h[3], ax=ax, label='Count')

# Invert x-axis (hotter stars on left, convention)
ax.invert_xaxis()
# Invert y-axis (giants at top)
ax.invert_yaxis()

ax.set_xlabel(r'$T_{\\rm eff}$ [K]', fontsize=12)
ax.set_ylabel(r'$\\log g$ [dex]', fontsize=12)
ax.set_title('Kiel Diagram', fontsize=14)

# Add annotations for regions
ax.annotate('Main Sequence', xy=(5500, 4.3), fontsize=10, color='white')
ax.annotate('Red Giants', xy=(4800, 2.5), fontsize=10, color='white')

plt.tight_layout()
plt.savefig('kiel_diagram.png', dpi=300)
plt.show()
```"""
    },
    {
        "user": "Plot spatial distribution with density contours",
        "assistant": """Combine scatter plot with kernel density contours:

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Example: RA, Dec positions
ra = data['ra']
dec = data['dec']

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
ax.scatter(ra, dec, s=1, alpha=0.3, c='gray', label='Sources')

# Compute and plot density contours
# Create 2D histogram for contours
xmin, xmax = ra.min(), ra.max()
ymin, ymax = dec.min(), dec.max()

# KDE for smooth contours
positions = np.vstack([ra, dec])
kernel = stats.gaussian_kde(positions)

# Create grid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions_grid = np.vstack([xx.ravel(), yy.ravel()])
z = kernel(positions_grid).reshape(xx.shape)

# Plot contours
levels = np.percentile(z, [50, 75, 90, 95, 99])
contours = ax.contour(xx, yy, z, levels=levels, colors='red', linewidths=1)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.1e')

ax.set_xlabel('RA [deg]', fontsize=12)
ax.set_ylabel('Dec [deg]', fontsize=12)
ax.set_title('Spatial Distribution with Density Contours')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('spatial_density.png', dpi=300)
plt.show()
```"""
    },
]

SELECTION_EXAMPLES = [
    {
        "user": "How do I select red giant stars from Gaia data?",
        "assistant": """Select red giants using color-magnitude cuts:

```python
import numpy as np

# Method 1: Simple CMD cuts
def select_red_giants(data):
    \"\"\"Select RGB stars using Gaia photometry.\"\"\"
    
    # Calculate absolute magnitude (if parallax available)
    parallax_mas = data['parallax']
    distance_pc = 1000 / parallax_mas
    M_G = data['phot_g_mean_mag'] - 5 * np.log10(distance_pc) + 5
    
    # RGB selection criteria
    color = data['bp_rp']
    
    # Typical RGB box:
    # - Color: 0.8 < BP-RP < 2.5
    # - Absolute mag: -3 < M_G < 4
    # - Above main sequence
    
    rgb_mask = (
        (color > 0.8) & (color < 2.5) &
        (M_G > -3) & (M_G < 4) &
        (M_G < 2.5 * color - 1)  # Above MS
    )
    
    return data[rgb_mask]

# Method 2: Using Gaia's own classifications
# Query the astrophysical_parameters table
query = \"\"\"
SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
       ap.teff_gspphot, ap.logg_gspphot, ap.mh_gspphot
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.astrophysical_parameters AS ap
  ON g.source_id = ap.source_id  
WHERE ap.logg_gspphot < 3.5
  AND ap.teff_gspphot BETWEEN 4000 AND 5500
  AND g.parallax_over_error > 5
\"\"\"
```

The log g < 3.5 cut isolates giants from dwarfs."""
    },
    {
        "user": "Select main sequence turnoff stars",
        "assistant": """Select MSTO stars for age determination:

```python
import numpy as np

def select_msto_stars(data, distance_pc, age_gyr=None):
    \"\"\"
    Select main sequence turnoff stars.
    MSTO location depends on cluster age.
    \"\"\"
    # Calculate absolute magnitude
    M_G = data['phot_g_mean_mag'] - 5 * np.log10(distance_pc) + 5
    color = data['bp_rp']
    
    # Approximate MSTO locations by age:
    # Young (100 Myr): M_G ~ 0-1, BP-RP ~ 0.0-0.3
    # 1 Gyr: M_G ~ 2-3, BP-RP ~ 0.3-0.5
    # 5 Gyr: M_G ~ 3.5-4.5, BP-RP ~ 0.5-0.7
    # 10 Gyr: M_G ~ 4-5, BP-RP ~ 0.6-0.8
    
    if age_gyr is None:
        # General MSTO region
        msto_mask = (
            (M_G > 0) & (M_G < 5) &
            (color > 0.0) & (color < 0.8) &
            (np.abs(M_G - (3.5 + 3 * (color - 0.4))) < 1)  # Near MS
        )
    else:
        # Age-specific selection
        if age_gyr < 0.5:
            msto_mask = (M_G > -1) & (M_G < 2) & (color > -0.1) & (color < 0.4)
        elif age_gyr < 2:
            msto_mask = (M_G > 1) & (M_G < 4) & (color > 0.2) & (color < 0.6)
        else:
            msto_mask = (M_G > 3) & (M_G < 5.5) & (color > 0.4) & (color < 0.9)
    
    return data[msto_mask], M_G[msto_mask]

# Usage
msto_stars, M_G = select_msto_stars(cluster_data, distance_pc=136, age_gyr=0.1)
print(f"Found {len(msto_stars)} MSTO candidates")
```"""
    },
    {
        "user": "Separate stars from galaxies in SDSS",
        "assistant": """Use morphological and color criteria to classify sources:

```python
import numpy as np

def classify_sdss_sources(data):
    \"\"\"
    Classify SDSS sources as stars or galaxies.
    Uses 'type' column and additional criteria.
    \"\"\"
    # SDSS type: 3 = galaxy, 6 = star
    sdss_stars = data['type'] == 6
    sdss_galaxies = data['type'] == 3
    
    # Additional quality cuts
    clean = (
        (data['r'] > 14) & (data['r'] < 22) &  # Reasonable magnitude range
        (data['err_r'] < 0.1) &  # Good photometry
        (data['g'] - data['r'] > -0.5) & (data['g'] - data['r'] < 2.5)  # Reasonable colors
    )
    
    # PSF-model magnitude difference (star/galaxy separator)
    # Stars: psfMag ≈ modelMag
    # Galaxies: psfMag > modelMag (extended)
    if 'psfMag_r' in data.colnames and 'modelMag_r' in data.colnames:
        psf_model_diff = data['psfMag_r'] - data['modelMag_r']
        point_source = np.abs(psf_model_diff) < 0.145  # Stars
        extended = psf_model_diff > 0.145  # Galaxies
    else:
        point_source = sdss_stars
        extended = sdss_galaxies
    
    return {
        'stars': data[clean & point_source],
        'galaxies': data[clean & extended],
        'stars_mask': clean & point_source,
        'galaxies_mask': clean & extended
    }

# Color-color selection for QSOs
def select_qso_candidates(data):
    \"\"\"Select QSO candidates using SDSS colors.\"\"\"
    u_g = data['u'] - data['g']
    g_r = data['g'] - data['r']
    r_i = data['r'] - data['i']
    i_z = data['i'] - data['z']
    
    # QSO color selection (simplified)
    qso_mask = (
        (u_g < 0.6) &
        (g_r < 0.5) &
        (r_i < 0.4) &
        (i_z < 0.4) &
        (data['type'] == 6)  # Point source
    )
    return data[qso_mask]
```"""
    },
    {
        "user": "Select white dwarf candidates",
        "assistant": """Select white dwarfs using Gaia CMD position:

```python
import numpy as np
from astroquery.gaia import Gaia

# Query Gaia for WD candidates
query = \"\"\"
SELECT source_id, ra, dec, parallax, parallax_error,
       phot_g_mean_mag, bp_rp, pmra, pmdec
FROM gaiadr3.gaia_source
WHERE parallax > 10
  AND parallax_over_error > 10
  AND phot_g_mean_mag + 5*LOG10(parallax/100) > 10
  AND bp_rp < 0.8
  AND bp_rp > -0.6
\"\"\"

job = Gaia.launch_job_async(query)
data = job.get_results()

def select_white_dwarfs(data):
    \"\"\"
    Select WD candidates from Gaia data.
    WDs occupy a distinct region in the CMD.
    \"\"\"
    # Calculate absolute magnitude
    M_G = data['phot_g_mean_mag'] + 5 * np.log10(data['parallax'] / 100)
    color = data['bp_rp']
    
    # WD selection box
    # WDs are blue (BP-RP < 0.8) and faint (M_G > 10)
    wd_mask = (
        (M_G > 10) & (M_G < 17) &
        (color > -0.6) & (color < 1.5) &
        # Below the WD cooling sequence upper bound
        (M_G > 3 * color + 10)
    )
    
    # Additional: proper motion cut for disk WDs
    total_pm = np.sqrt(data['pmra']**2 + data['pmdec']**2)
    high_pm = total_pm > 30  # mas/yr
    
    return {
        'all_wd': data[wd_mask],
        'disk_wd': data[wd_mask & high_pm],  # High-PM likely old disk WDs
        'M_G': M_G[wd_mask]
    }

wd_result = select_white_dwarfs(data)
print(f"Found {len(wd_result['all_wd'])} WD candidates")
```

For confirmed WDs, cross-match with the Montreal White Dwarf Database."""
    },
    {
        "user": "Select blue horizontal branch stars",
        "assistant": """Select BHB stars for distance determination:

```python
import numpy as np

def select_bhb_stars(data, has_parallax=True):
    \"\"\"
    Select Blue Horizontal Branch stars.
    BHB stars are excellent standard candles (M_g ~ 0.5).
    \"\"\"
    color = data['bp_rp']
    
    if has_parallax and 'parallax' in data.colnames:
        # Use absolute magnitude
        M_G = data['phot_g_mean_mag'] + 5 * np.log10(data['parallax'] / 100)
        
        # BHB selection in absolute CMD
        bhb_mask = (
            (color > -0.4) & (color < 0.2) &  # Blue
            (M_G > 0) & (M_G < 1.5) &  # HB luminosity
            (data['parallax_over_error'] > 5)
        )
    else:
        # Use color-color or apparent mag (less reliable)
        # Requires ugriz or JHK
        if 'u' in data.colnames and 'g' in data.colnames:
            u_g = data['u'] - data['g']
            g_r = data['g'] - data['r']
            
            # BHB color box (SDSS)
            bhb_mask = (
                (u_g > 0.9) & (u_g < 1.4) &
                (g_r > -0.4) & (g_r < 0.0)
            )
        else:
            # Simple Gaia color cut
            bhb_mask = (color > -0.4) & (color < 0.0)
    
    # BHB absolute magnitude: M_g ≈ 0.5 (can vary with metallicity)
    # Distance modulus: m - M = 5*log10(d/10pc)
    if 'g' in data.colnames:
        distance_pc = 10 ** ((data['g'][bhb_mask] - 0.5) / 5 + 1)
    else:
        distance_pc = 10 ** ((data['phot_g_mean_mag'][bhb_mask] - 0.5) / 5 + 1)
    
    return {
        'bhb_stars': data[bhb_mask],
        'distances_pc': distance_pc
    }
```

BHB stars are found in globular clusters and the halo field."""
    },
    {
        "user": "Select metal-poor stars",
        "assistant": """Select metal-poor candidates using photometry:

```python
import numpy as np
from astroquery.gaia import Gaia

# Method 1: Use Gaia's metallicity estimates
query = \"\"\"
SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
       ap.mh_gspphot, ap.teff_gspphot, ap.logg_gspphot
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.astrophysical_parameters AS ap
  ON g.source_id = ap.source_id
WHERE ap.mh_gspphot < -1.5
  AND ap.teff_gspphot BETWEEN 4500 AND 6500
  AND g.parallax_over_error > 5
\"\"\"

job = Gaia.launch_job_async(query)
mp_candidates = job.get_results()

# Method 2: Photometric metallicity estimate
def estimate_metallicity(data):
    \"\"\"
    Estimate metallicity from Gaia colors (rough).
    More accurate with ugriz or narrow-band photometry.
    \"\"\"
    # UV excess method (requires u-band)
    if 'u' in data.colnames:
        # δ(U-B) method for F/G stars
        u_g = data['u'] - data['g']
        g_r = data['g'] - data['r']
        
        # Expected u-g for solar metallicity (simplified)
        u_g_solar = 0.5 + 2.0 * g_r
        delta_ug = u_g - u_g_solar
        
        # Rough metallicity: Δ(u-g) ~ -0.3 * [Fe/H]
        feh_estimate = delta_ug / (-0.3)
        
        metal_poor = feh_estimate < -1.0
    else:
        # Use BP-RP with caution
        # Metal-poor giants are bluer at same temperature
        color = data['bp_rp']
        
        # This is very rough - should use spectroscopy
        metal_poor = color < 0.9  # For giants
    
    return metal_poor

# Method 3: Proper motion for halo membership
def select_halo_candidates(data):
    \"\"\"Halo stars have high velocities.\"\"\"
    total_pm = np.sqrt(data['pmra']**2 + data['pmdec']**2)
    
    # Combine proper motion with parallax to get tangential velocity
    v_tan = 4.74 * total_pm / data['parallax']  # km/s
    
    # Halo stars typically have v_tan > 200 km/s
    halo_candidate = v_tan > 200
    
    return data[halo_candidate], v_tan[halo_candidate]
```

For confirmed metallicities, cross-match with LAMOST, APOGEE, or GALAH."""
    },
]

DATA_REDUCTION_EXAMPLES = [
    {
        "user": "How do I perform aperture photometry?",
        "assistant": """Perform aperture photometry using photutils:

```python
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
import numpy as np

# Load image
hdu = fits.open('image.fits')
data = hdu[0].data
header = hdu[0].header

# Background estimation
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
print(f"Background: {median:.2f} ± {std:.2f}")

# Find sources
daofind = DAOStarFinder(fwhm=4.0, threshold=5.0 * std)
sources = daofind(data - median)
print(f"Found {len(sources)} sources")

# Define apertures
positions = np.transpose([sources['xcentroid'], sources['ycentroid']])
apertures = CircularAperture(positions, r=5.0)

# Perform photometry
phot_table = aperture_photometry(data - median, apertures)

# Add instrumental magnitude
phot_table['mag_inst'] = -2.5 * np.log10(phot_table['aperture_sum'])

# Aperture correction (if needed)
# Use curve of growth or PSF model
aperture_corr = 0.1  # Example value
phot_table['mag_corr'] = phot_table['mag_inst'] - aperture_corr

print(phot_table['id', 'xcenter', 'ycenter', 'aperture_sum', 'mag_corr'])
```

For better photometry, use PSF fitting (`photutils.psf`) or SExtractor."""
    },
    {
        "user": "How to calibrate magnitudes to a standard system?",
        "assistant": """Calibrate instrumental magnitudes using reference stars:

```python
import numpy as np
from scipy.optimize import curve_fit

def calibrate_photometry(inst_mag, ref_mag, airmass=None, color=None):
    \"\"\"
    Calibrate instrumental magnitudes to standard system.
    
    Standard transformation:
    m_std = m_inst + ZP + k*X + c*color
    
    where:
    - ZP: zero point
    - k: extinction coefficient
    - X: airmass
    - c: color term
    \"\"\"
    
    # Simple zero point (no extinction or color terms)
    if airmass is None and color is None:
        zp = np.median(ref_mag - inst_mag)
        zp_err = np.std(ref_mag - inst_mag) / np.sqrt(len(inst_mag))
        
        return {
            'zero_point': zp,
            'zp_error': zp_err,
            'calibrated_mag': inst_mag + zp
        }
    
    # Full transformation with extinction
    elif color is None:
        def model(X, zp, k):
            return inst_mag + zp + k * X
        
        popt, pcov = curve_fit(model, airmass, ref_mag)
        zp, k = popt
        
        return {
            'zero_point': zp,
            'extinction': k,
            'calibrated_mag': inst_mag + zp + k * airmass
        }
    
    # Full transformation with color term
    else:
        def model(params, zp, k, c):
            X, col = params
            return inst_mag + zp + k * X + c * col
        
        popt, pcov = curve_fit(model, (airmass, color), ref_mag)
        zp, k, c = popt
        
        return {
            'zero_point': zp,
            'extinction': k,
            'color_term': c,
            'calibrated_mag': inst_mag + zp + k * airmass + c * color
        }

# Example usage with Gaia reference stars
from astroquery.gaia import Gaia

# Query Gaia stars in your field
query = f\"\"\"
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec),
               CIRCLE('ICRS', {ra_center}, {dec_center}, 0.1)) = 1
  AND phot_g_mean_mag BETWEEN 14 AND 18
\"\"\"

gaia_ref = Gaia.launch_job(query).get_results()

# Cross-match with your sources (by position)
# Then calibrate
result = calibrate_photometry(my_inst_mag, gaia_ref['phot_g_mean_mag'])
print(f"Zero point: {result['zero_point']:.3f} ± {result['zp_error']:.3f}")
```"""
    },
    {
        "user": "How do I perform astrometric calibration?",
        "assistant": """Calibrate WCS using Gaia reference stars:

```python
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import numpy as np

# Method 1: Using astropy + manual plate solution
def calibrate_astrometry(image_file, detected_sources, approx_center):
    \"\"\"
    Calibrate WCS using Gaia reference stars.
    
    detected_sources: table with 'x', 'y' pixel positions
    approx_center: approximate (RA, Dec) of field center
    \"\"\"
    # Query Gaia for reference stars
    query = f\"\"\"
    SELECT source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {approx_center[0]}, {approx_center[1]}, 0.5)) = 1
      AND phot_g_mean_mag BETWEEN 12 AND 18
    \"\"\"
    gaia_ref = Gaia.launch_job(query).get_results()
    
    # Match detected sources to Gaia (need initial rough WCS)
    # ... matching code here ...
    
    return matched_sources, gaia_coords

# Method 2: Using astrometry.net (recommended)
# Install: sudo apt install astrometry.net astrometry-data-*
import subprocess

def solve_astrometry(image_file, ra=None, dec=None, radius=1.0):
    \"\"\"
    Solve astrometry using astrometry.net.
    Returns WCS-calibrated FITS file.
    \"\"\"
    cmd = ['solve-field', image_file,
           '--overwrite',
           '--no-plots',
           '--cpulimit', '60']
    
    if ra is not None and dec is not None:
        cmd.extend(['--ra', str(ra), '--dec', str(dec), 
                    '--radius', str(radius)])
    
    subprocess.run(cmd, check=True)
    
    # Result is in image.new (WCS-calibrated)
    solved_file = image_file.replace('.fits', '.new')
    return solved_file

# Method 3: Using astroalign for image registration
import astroalign as aa

def align_images(reference_image, target_image):
    \"\"\"Align target to reference image.\"\"\"
    ref_data = fits.getdata(reference_image)
    target_data = fits.getdata(target_image)
    
    # Find transformation
    aligned, footprint = aa.register(target_data, ref_data)
    
    return aligned

# Check astrometric accuracy
def check_astrometry(wcs, detected_x, detected_y, gaia_ra, gaia_dec):
    \"\"\"Calculate astrometric residuals.\"\"\"
    # Convert pixel to sky
    sky_coords = wcs.pixel_to_world(detected_x, detected_y)
    
    # Calculate offsets to Gaia
    gaia_coords = SkyCoord(ra=gaia_ra*u.deg, dec=gaia_dec*u.deg)
    sep = sky_coords.separation(gaia_coords)
    
    print(f"Median offset: {np.median(sep.arcsec):.2f} arcsec")
    print(f"RMS: {np.std(sep.arcsec):.2f} arcsec")
    
    return sep
```"""
    },
    {
        "user": "How do I stack multiple images?",
        "assistant": """Stack images using various methods:

```python
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clip
import numpy as np
from ccdproc import combine, wcs_project
import glob

def stack_images(file_list, method='median', sigma_clip_val=3.0):
    \"\"\"
    Stack multiple FITS images.
    
    Methods: 'median', 'average', 'sum'
    \"\"\"
    # Load images
    images = [CCDData.read(f, unit='adu') for f in file_list]
    
    # Combine with sigma clipping
    combined = combine(
        images,
        method=method,
        sigma_clip=True,
        sigma_clip_low_thresh=sigma_clip_val,
        sigma_clip_high_thresh=sigma_clip_val,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=np.ma.std
    )
    
    return combined

# For images that need alignment first
def align_and_stack(file_list, reference_idx=0):
    \"\"\"Align images to reference then stack.\"\"\"
    import astroalign as aa
    
    # Load reference
    ref_data = fits.getdata(file_list[reference_idx])
    
    aligned_images = [ref_data]
    
    for i, f in enumerate(file_list):
        if i == reference_idx:
            continue
        
        target_data = fits.getdata(f)
        try:
            aligned, _ = aa.register(target_data, ref_data)
            aligned_images.append(aligned)
        except:
            print(f"Warning: Could not align {f}")
    
    # Stack
    stack = np.stack(aligned_images, axis=0)
    combined = np.median(stack, axis=0)
    
    return combined

# For images with WCS
def reproject_and_stack(file_list, output_wcs, output_shape):
    \"\"\"Reproject images to common WCS and stack.\"\"\"
    from reproject import reproject_interp
    
    reprojected = []
    
    for f in file_list:
        hdu = fits.open(f)[0]
        array, footprint = reproject_interp(hdu, output_wcs, output_shape)
        reprojected.append(array)
    
    # Stack with sigma clipping
    stack = np.stack(reprojected, axis=0)
    combined = np.nanmedian(stack, axis=0)
    
    return combined

# Example usage
files = sorted(glob.glob('science_*.fits'))
stacked = stack_images(files, method='median')
stacked.write('stacked.fits', overwrite=True)
```"""
    },
    {
        "user": "How to do bias and flat field correction?",
        "assistant": """Perform standard CCD reduction:

```python
from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc as ccdp
from astropy.stats import mad_std
import numpy as np
import glob

def reduce_ccd_images(science_files, bias_files, flat_files, output_dir='reduced/'):
    \"\"\"
    Standard CCD reduction pipeline.
    
    1. Create master bias
    2. Create master flat
    3. Apply corrections to science frames
    \"\"\"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create master bias
    print("Creating master bias...")
    bias_list = [CCDData.read(f, unit='adu') for f in bias_files]
    master_bias = ccdp.combine(
        bias_list,
        method='median',
        sigma_clip=True,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std
    )
    master_bias.write(f'{output_dir}/master_bias.fits', overwrite=True)
    print(f"  Bias level: {np.median(master_bias):.1f} ADU")
    
    # 2. Create master flat
    print("Creating master flat...")
    flat_list = []
    for f in flat_files:
        flat = CCDData.read(f, unit='adu')
        # Subtract bias
        flat_bias_sub = ccdp.subtract_bias(flat, master_bias)
        # Normalize
        flat_normalized = flat_bias_sub.divide(np.median(flat_bias_sub))
        flat_list.append(flat_normalized)
    
    master_flat = ccdp.combine(
        flat_list,
        method='median',
        sigma_clip=True
    )
    master_flat.write(f'{output_dir}/master_flat.fits', overwrite=True)
    
    # 3. Reduce science frames
    print("Reducing science frames...")
    for sci_file in science_files:
        sci = CCDData.read(sci_file, unit='adu')
        
        # Subtract bias
        sci_bias_sub = ccdp.subtract_bias(sci, master_bias)
        
        # Divide by flat
        sci_reduced = ccdp.flat_correct(sci_bias_sub, master_flat)
        
        # Optional: cosmic ray removal
        sci_clean = ccdp.cosmicray_lacosmic(sci_reduced, sigclip=5)
        
        # Save
        basename = os.path.basename(sci_file).replace('.fits', '_reduced.fits')
        sci_clean.write(f'{output_dir}/{basename}', overwrite=True)
        print(f"  Reduced: {basename}")
    
    return master_bias, master_flat

# Example usage
bias_files = sorted(glob.glob('bias_*.fits'))
flat_files = sorted(glob.glob('flat_*.fits'))
science_files = sorted(glob.glob('science_*.fits'))

reduce_ccd_images(science_files, bias_files, flat_files)
```"""
    },
]

LITERATURE_EXAMPLES = [
    {
        "user": "How do I search NASA ADS for papers?",
        "assistant": """Query NASA ADS using the ads Python library:

```python
import ads

# Set your API token (get from https://ui.adsabs.harvard.edu/user/settings/token)
ads.config.token = 'your-api-token'

# Basic search
papers = ads.SearchQuery(
    q="Gaia DR3 stellar parameters",
    sort="citation_count desc",
    rows=20
)

for paper in papers:
    print(f"{paper.year} | {paper.bibcode}")
    print(f"  {paper.title[0]}")
    print(f"  Citations: {paper.citation_count}")
    print()

# Search by author
author_papers = ads.SearchQuery(
    author="^Gaia Collaboration",
    year="2020-2024",
    sort="date desc"
)

# Search by bibcode
specific_paper = list(ads.SearchQuery(bibcode="2023A&A...674A...1G"))[0]

# Full-text search
fulltext = ads.SearchQuery(
    q="full:\"red giant branch\" AND full:\"asteroseismology\"",
    fq="property:refereed",  # Only refereed papers
    rows=50
)

# Get abstracts
for paper in ads.SearchQuery(q="exoplanet atmospheres", rows=5):
    print(f"Title: {paper.title[0]}")
    print(f"Abstract: {paper.abstract[:200]}...")
    print()
```"""
    },
    {
        "user": "Generate a BibTeX citation",
        "assistant": """Export BibTeX citations from ADS:

```python
import ads
import requests

def get_bibtex(bibcodes, token):
    \"\"\"
    Get BibTeX entries for a list of bibcodes.
    \"\"\"
    if isinstance(bibcodes, str):
        bibcodes = [bibcodes]
    
    # Use ADS export API
    url = 'https://api.adsabs.harvard.edu/v1/export/bibtex'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    payload = {'bibcode': bibcodes}
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['export']
    else:
        raise Exception(f"Error: {response.status_code}")

# Example usage
token = 'your-api-token'

# Single paper
bibtex = get_bibtex('2023A&A...674A...1G', token)
print(bibtex)

# Multiple papers
bibcodes = [
    '2023A&A...674A...1G',  # Gaia DR3
    '2020AJ....160...71S',  # SDSS DR16
    '2006AJ....131.1163S',  # 2MASS
]
all_bibtex = get_bibtex(bibcodes, token)

# Save to file
with open('references.bib', 'w') as f:
    f.write(all_bibtex)
print("Saved to references.bib")

# Alternative: Use ads library directly
paper = list(ads.SearchQuery(bibcode="2023A&A...674A...1G"))[0]
print(f\"\"\"
@ARTICLE{{{paper.bibcode},
    author = {{{{{' and '.join(paper.author[:3])}{' et al.' if len(paper.author) > 3 else ''}}}}},
    title = {{{{{paper.title[0]}}}}},
    journal = {{{paper.pub}}},
    year = {paper.year},
    volume = {{{paper.volume}}},
    pages = {{{paper.page[0] if paper.page else ''}}},
}}
\"\"\")
```"""
    },
    {
        "user": "Find papers citing a specific paper",
        "assistant": """Find citations and references using ADS:

```python
import ads

def get_citations(bibcode, token):
    \"\"\"Get papers that cite a given paper.\"\"\"
    ads.config.token = token
    
    # Search for citing papers
    citing = ads.SearchQuery(
        q=f'citations(bibcode:{bibcode})',
        sort='citation_count desc',
        rows=100,
        fl=['bibcode', 'title', 'author', 'year', 'citation_count']
    )
    
    citing_papers = list(citing)
    print(f"Found {len(citing_papers)} citing papers")
    
    return citing_papers

def get_references(bibcode, token):
    \"\"\"Get papers referenced by a given paper.\"\"\"
    ads.config.token = token
    
    refs = ads.SearchQuery(
        q=f'references(bibcode:{bibcode})',
        rows=200
    )
    
    return list(refs)

def citation_network(seed_bibcode, token, depth=1):
    \"\"\"Build a citation network starting from a seed paper.\"\"\"
    network = {seed_bibcode: {'level': 0, 'citations': []}}
    
    for level in range(depth):
        current_papers = [b for b, info in network.items() 
                         if info['level'] == level]
        
        for bibcode in current_papers:
            citing = get_citations(bibcode, token)
            for paper in citing[:10]:  # Limit to top 10
                if paper.bibcode not in network:
                    network[paper.bibcode] = {
                        'level': level + 1,
                        'title': paper.title[0] if paper.title else '',
                        'citations': paper.citation_count
                    }
                network[bibcode]['citations'].append(paper.bibcode)
    
    return network

# Example: Find who cited the Gaia DR3 paper
token = 'your-api-token'
gaia_dr3 = '2023A&A...674A...1G'

citing_papers = get_citations(gaia_dr3, token)

print("Top 10 papers citing Gaia DR3:")
for paper in citing_papers[:10]:
    print(f"  [{paper.citation_count}] {paper.title[0][:60]}...")
    print(f"      {paper.bibcode}")
```"""
    },
    {
        "user": "Search arXiv for recent preprints",
        "assistant": """Query arXiv for recent astronomy papers:

```python
import arxiv
from datetime import datetime, timedelta

def search_arxiv(query, max_results=20, category='astro-ph'):
    \"\"\"
    Search arXiv for papers.
    
    Categories:
    - astro-ph: all astrophysics
    - astro-ph.SR: solar/stellar
    - astro-ph.GA: galaxies
    - astro-ph.CO: cosmology
    - astro-ph.EP: exoplanets
    - astro-ph.HE: high energy
    - astro-ph.IM: instrumentation
    \"\"\"
    search = arxiv.Search(
        query=f'cat:{category} AND {query}',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = []
    for paper in search.results():
        results.append({
            'title': paper.title,
            'authors': [a.name for a in paper.authors],
            'abstract': paper.summary,
            'arxiv_id': paper.entry_id.split('/')[-1],
            'submitted': paper.published,
            'pdf_url': paper.pdf_url,
            'categories': paper.categories
        })
    
    return results

# Search recent preprints
papers = search_arxiv('Gaia radial velocity', max_results=10)

for p in papers:
    print(f"[{p['arxiv_id']}] {p['title']}")
    print(f"  Authors: {', '.join(p['authors'][:3])}...")
    print(f"  Submitted: {p['submitted'].strftime('%Y-%m-%d')}")
    print()

# Download PDFs
def download_arxiv_pdf(arxiv_id, output_dir='./papers'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    paper = next(arxiv.Search(id_list=[arxiv_id]).results())
    paper.download_pdf(dirpath=output_dir)
    print(f"Downloaded: {paper.title}")

# Recent papers in the last week
def get_recent_papers(category='astro-ph.SR', days=7):
    since = datetime.now() - timedelta(days=days)
    
    search = arxiv.Search(
        query=f'cat:{category}',
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    recent = []
    for paper in search.results():
        if paper.published.replace(tzinfo=None) > since:
            recent.append(paper)
    
    return recent

recent = get_recent_papers('astro-ph.SR', days=3)
print(f"Found {len(recent)} stellar papers in last 3 days")
```"""
    },
]

CONCEPT_EXAMPLES = [
    {
        "user": "What is the difference between apparent and absolute magnitude?",
        "assistant": """**Apparent magnitude (m)** is how bright an object appears from Earth, while **absolute magnitude (M)** is its intrinsic brightness at a standard distance of 10 parsecs.

**Relationship:**
```
m - M = 5 * log10(d/10)
```
where d is distance in parsecs.

The quantity (m - M) is called the **distance modulus**.

**Examples:**
- Sun: m = -26.7 (very bright), M = +4.8 (average star)
- Vega: m = +0.03, M = +0.6
- Betelgeuse: m = +0.5, M = -5.6 (very luminous supergiant)

**In code:**
```python
import numpy as np

def apparent_to_absolute(m, distance_pc):
    return m - 5 * np.log10(distance_pc / 10)

def absolute_to_apparent(M, distance_pc):
    return M + 5 * np.log10(distance_pc / 10)

def distance_modulus(m, M):
    return m - M

def distance_from_modulus(m, M):
    return 10 ** ((m - M + 5) / 5)

# Example: Star with m=15 at 1000 pc
m = 15
d = 1000  # pc
M = apparent_to_absolute(m, d)
print(f"Absolute magnitude: {M:.2f}")
```

For Gaia data, use parallax to get distance:
```python
distance_pc = 1000 / parallax_mas
M_G = phot_g_mean_mag - 5 * np.log10(distance_pc / 10)
```"""
    },
    {
        "user": "Explain the Hertzsprung-Russell diagram",
        "assistant": """The **Hertzsprung-Russell (HR) diagram** plots stellar luminosity vs. temperature, revealing distinct evolutionary stages:

**Axes:**
- Y-axis: Luminosity (or absolute magnitude) — brighter at top
- X-axis: Temperature (or color) — hotter stars on LEFT (blue), cooler on RIGHT (red)

**Key Features:**

1. **Main Sequence (MS)**: Diagonal band from upper-left (hot, luminous) to lower-right (cool, faint). Stars burning hydrogen in their cores spend ~90% of their lives here.

2. **Red Giant Branch (RGB)**: Upper-right region. Stars after hydrogen exhaustion, with expanding envelopes.

3. **Horizontal Branch (HB)**: Stars burning helium in their cores.

4. **Asymptotic Giant Branch (AGB)**: Late evolutionary stage with shell burning.

5. **White Dwarf Sequence**: Lower-left. Stellar remnants cooling over time.

**Create an HR diagram with Gaia:**
```python
import matplotlib.pyplot as plt
import numpy as np
from astroquery.gaia import Gaia

# Query nearby stars with good parallax
query = \"\"\"
SELECT phot_g_mean_mag, bp_rp, parallax
FROM gaiadr3.gaia_source
WHERE parallax > 20 AND parallax_over_error > 10
  AND bp_rp IS NOT NULL
\"\"\"
data = Gaia.launch_job(query).get_results()

# Calculate absolute magnitude
M_G = data['phot_g_mean_mag'] + 5 * np.log10(data['parallax'] / 100)

# Plot HR diagram (CMD as proxy)
plt.figure(figsize=(8, 10))
plt.scatter(data['bp_rp'], M_G, s=1, alpha=0.1)
plt.xlabel('BP - RP (color)')
plt.ylabel('M_G (absolute magnitude)')
plt.gca().invert_yaxis()  # Brighter at top
plt.title('HR Diagram from Gaia')
plt.show()
```

The HR diagram is fundamental for understanding stellar evolution, determining ages, and identifying stellar populations."""
    },
    {
        "user": "What is interstellar extinction and how do I correct for it?",
        "assistant": """**Interstellar extinction** is the dimming and reddening of starlight by dust and gas between the star and observer.

**Key quantities:**
- **A_V**: Total extinction in V-band (magnitudes)
- **E(B-V)**: Color excess (reddening)
- **R_V**: Ratio of total to selective extinction, typically R_V ≈ 3.1

**Relationships:**
```
A_V = R_V × E(B-V)
A_λ = A_V × (λ_eff / λ_V)^α
```

**Extinction correction:**
```python
import numpy as np
from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery
from astropy.coordinates import SkyCoord
import astropy.units as u

# Method 1: Using dust maps (SFD, Bayestar)
def get_extinction_sfd(ra, dec):
    \"\"\"Get E(B-V) from SFD dust map.\"\"\"
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coords)
    return ebv

# Method 2: Using 3D dust map (Bayestar, requires distance)
def get_extinction_3d(ra, dec, distance_pc):
    \"\"\"Get E(B-V) from 3D dust map.\"\"\"
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, 
                      distance=distance_pc*u.pc, frame='icrs')
    bayestar = BayestarQuery()
    ebv = bayestar(coords)
    return ebv

# Convert E(B-V) to extinction in different bands
def extinction_in_band(ebv, band='V', Rv=3.1):
    \"\"\"
    Calculate extinction A_band from E(B-V).
    Coefficients from Cardelli et al. (1989) or Fitzpatrick (1999).
    \"\"\"
    # Approximate coefficients
    coefficients = {
        'U': 1.531,
        'B': 1.324,
        'V': 1.000,
        'R': 0.748,
        'I': 0.482,
        'J': 0.282,
        'H': 0.175,
        'K': 0.112,
        'G': 0.789,  # Gaia G-band
        'BP': 1.002,  # Gaia BP
        'RP': 0.589,  # Gaia RP
    }
    
    A_V = Rv * ebv
    return A_V * coefficients.get(band, 1.0)

# Apply correction
def deredden_magnitude(mag, ebv, band='V', Rv=3.1):
    \"\"\"Correct observed magnitude for extinction.\"\"\"
    A_band = extinction_in_band(ebv, band, Rv)
    return mag - A_band

# Example: Correct Gaia magnitudes
ebv = 0.3  # Example E(B-V)
G_obs = 15.0
G_dered = deredden_magnitude(G_obs, ebv, band='G')
print(f"Observed G: {G_obs}, Dereddened G: {G_dered:.2f}")
```

For Gaia, extinction coefficients depend on stellar color. Use `GaiaXPy` for accurate corrections."""
    },
    {
        "user": "What is proper motion and how is it measured?",
        "assistant": """**Proper motion** is the angular motion of a star across the sky, measured in milliarcseconds per year (mas/yr).

**Components:**
- **μα***: Motion in Right Ascension (includes cos(δ) factor)
- **μδ**: Motion in Declination
- **Total PM**: √(μα*² + μδ²)

**Physical meaning:**
Proper motion depends on the star's true space velocity AND its distance:
```
PM (mas/yr) = v_tan (km/s) × 1000 / (4.74 × d (pc))
```

Or equivalently:
```
v_tan (km/s) = 4.74 × PM (mas/yr) × d (pc) / 1000
```

**Working with proper motion:**
```python
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

def proper_motion_analysis(data):
    \"\"\"Analyze proper motion from Gaia data.\"\"\"
    pmra = data['pmra']  # mas/yr (already includes cos(dec))
    pmdec = data['pmdec']  # mas/yr
    parallax = data['parallax']  # mas
    
    # Total proper motion
    total_pm = np.sqrt(pmra**2 + pmdec**2)
    
    # Position angle of motion (North through East)
    pa = np.degrees(np.arctan2(pmra, pmdec))
    
    # Tangential velocity (km/s)
    distance_pc = 1000 / parallax
    v_tan = 4.74 * total_pm * distance_pc / 1000
    
    return {
        'total_pm': total_pm,
        'position_angle': pa,
        'v_tangential': v_tan
    }

# Convert to Galactic velocity components
def pm_to_galactic_velocity(ra, dec, pmra, pmdec, parallax, rv=0):
    \"\"\"
    Convert proper motion to Galactic UVW velocities.
    Requires radial velocity for full 3D.
    \"\"\"
    coords = SkyCoord(
        ra=ra*u.deg, dec=dec*u.deg,
        distance=(1000/parallax)*u.pc,
        pm_ra_cosdec=pmra*u.mas/u.yr,
        pm_dec=pmdec*u.mas/u.yr,
        radial_velocity=rv*u.km/u.s
    )
    
    galcen = coords.galactocentric
    
    return {
        'U': galcen.v_x.value,  # Toward Galactic center
        'V': galcen.v_y.value,  # Direction of rotation
        'W': galcen.v_z.value   # Toward North Galactic Pole
    }

# High proper motion stars are nearby
# Example: Barnard's Star has PM ≈ 10,000 mas/yr (10 arcsec/yr!)
```

**Applications:**
- Identify nearby stars (high PM)
- Cluster membership (common PM)
- Kinematic populations (thin/thick disk, halo)
- Binary detection (PM anomaly)"""
    },
    {
        "user": "What are the different Gaia data releases and their differences?",
        "assistant": """**Gaia Data Releases Overview:**

| Release | Date | Sources | Key Features |
|---------|------|---------|--------------|
| DR1 | Sep 2016 | 1.1B | Positions only, TGAS subset with PM |
| DR2 | Apr 2018 | 1.7B | Full 5-param astrometry, 3-band photometry |
| EDR3 | Dec 2020 | 1.8B | Improved astrometry, same photometry as DR2 |
| **DR3** | Jun 2022 | 1.8B | Spectra, RVs, stellar params, variability |
| DR4 | ~2026 | ~2B | Improved everything, more epochs |
| DR5 | ~2030 | ~2B | Final release, 10-year baseline |

**DR3 Tables (most commonly used):**

```python
from astroquery.gaia import Gaia

# Main source table
# gaiadr3.gaia_source - 1.8 billion sources
# Columns: source_id, ra, dec, parallax, pmra, pmdec, phot_g/bp/rp_mean_mag

# Cross-matches
# gaiadr3.tmass_psc_xsc_best_neighbour - 2MASS
# gaiadr3.allwise_best_neighbour - WISE
# gaiadr3.panstarrs1_best_neighbour - Pan-STARRS

# Astrophysical parameters (470M stars)
# gaiadr3.astrophysical_parameters - Teff, logg, [Fe/H], A_G

# Radial velocities (33M stars)
# gaiadr3.gaia_source (radial_velocity column)

# Variability
# gaiadr3.vari_summary - 10M variable sources
# gaiadr3.vari_rrlyrae, vari_cepheid, vari_eclipsing_binary, etc.

# XP Spectra (220M sources)
# gaiadr3.xp_continuous_mean_spectrum
# gaiadr3.xp_sampled_mean_spectrum

# Example: Query with multiple tables
query = \"\"\"
SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
       g.parallax, g.radial_velocity,
       ap.teff_gspphot, ap.logg_gspphot, ap.mh_gspphot
FROM gaiadr3.gaia_source AS g
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
  ON g.source_id = ap.source_id
WHERE g.parallax > 10
  AND g.parallax_over_error > 10
\"\"\"
```

**Key improvements DR2 → DR3:**
- 30% more radial velocities
- Stellar parameters for 470M stars
- BP/RP spectra for 220M sources
- Extended variability classification
- Non-single star solutions (binaries)

**Data Quality Flags:**
```python
# Good astrometry cuts:
# - ruwe < 1.4 (renormalized unit weight error)
# - parallax_over_error > 5
# - visibility_periods_used > 8

# Good photometry:
# - phot_bp_rp_excess_factor in expected range
# - phot_g_mean_flux_over_error > 50
```"""
    },
]

# =============================================================================
# Data Generation
# =============================================================================

def generate_variations(examples: List[Dict], count_per_example: int = 3) -> List[Dict]:
    """Generate variations of examples with slight modifications."""
    variations = []
    
    # Object names for variation
    objects = [
        "M31", "M33", "NGC 6397", "NGC 104", "Omega Centauri", "M13",
        "M67", "Hyades", "Praesepe", "NGC 752", "M35", "M45",
        "Orion Nebula", "Carina Nebula", "Large Magellanic Cloud",
        "Sagittarius Dwarf", "Sculptor Dwarf", "Draco Dwarf",
        "Cygnus X-1", "SS 433", "GW Ori", "TW Hya"
    ]
    
    # Coordinates for variation
    coords = [
        (180.0, 45.0), (10.5, -23.4), (266.4, -29.0), (83.8, -5.4),
        (56.75, 24.12), (201.3, -43.0), (187.5, 12.4), (132.8, 11.8)
    ]
    
    for example in examples:
        variations.append(example)  # Keep original
        
        # Create variations
        for _ in range(count_per_example - 1):
            new_user = example["user"]
            new_assistant = example["assistant"]
            
            # Random coordinate substitution
            if "180.0" in new_assistant or "45.0" in new_assistant:
                ra, dec = random.choice(coords)
                new_assistant = new_assistant.replace("180.0", str(ra))
                new_assistant = new_assistant.replace("45.0", str(dec))
            
            # Random object substitution
            for obj in objects[:5]:
                if obj in new_user or obj in new_assistant:
                    new_obj = random.choice(objects)
                    new_user = new_user.replace(obj, new_obj)
                    new_assistant = new_assistant.replace(obj, new_obj)
                    break
            
            variations.append({
                "user": new_user,
                "assistant": new_assistant
            })
    
    return variations


def create_training_example(user: str, assistant: str) -> Dict[str, Any]:
    """Create a training example in the correct format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }


def generate_all_training_data(target_count: int = 700, categories: List[str] = None) -> List[Dict]:
    """Generate all training data."""
    
    all_examples = []
    
    category_data = {
        "catalog": CATALOG_QUERIES,
        "plotting": PLOTTING_EXAMPLES,
        "selection": SELECTION_EXAMPLES,
        "reduction": DATA_REDUCTION_EXAMPLES,
        "literature": LITERATURE_EXAMPLES,
        "concepts": CONCEPT_EXAMPLES,
    }
    
    if categories:
        category_data = {k: v for k, v in category_data.items() if k in categories}
    
    # Calculate variations needed
    total_base = sum(len(v) for v in category_data.values())
    variations_per_example = max(3, target_count // total_base)
    
    print(f"Base examples: {total_base}")
    print(f"Variations per example: {variations_per_example}")
    
    for category, examples in category_data.items():
        print(f"Processing {category}: {len(examples)} base examples")
        varied = generate_variations(examples, variations_per_example)
        
        for ex in varied:
            all_examples.append(create_training_example(ex["user"], ex["assistant"]))
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Trim to target
    if len(all_examples) > target_count:
        all_examples = all_examples[:target_count]
    
    return all_examples


def save_training_data(examples: List[Dict], output_dir: str, split_ratio: float = 0.9):
    """Save training data to JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split into train/validation
    split_idx = int(len(examples) * split_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Save training data
    train_file = output_path / "train.jsonl"
    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(train_examples)} training examples to {train_file}")
    
    # Save validation data
    val_file = output_path / "validation.jsonl"
    with open(val_file, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(val_examples)} validation examples to {val_file}")
    
    # Save combined for reference
    all_file = output_path / "all_examples.jsonl"
    with open(all_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} total examples to {all_file}")
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total examples: {len(examples)}")
    print(f"Training: {len(train_examples)}")
    print(f"Validation: {len(val_examples)}")
    print(f"Output directory: {output_path}")
    
    return {
        "train_file": str(train_file),
        "val_file": str(val_file),
        "train_count": len(train_examples),
        "val_count": len(val_examples)
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate AstroLlama training data")
    parser.add_argument("--output", "-o", default="./data/training/",
                        help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=700,
                        help="Target number of examples")
    parser.add_argument("--categories", "-c", type=str, default=None,
                        help="Comma-separated list of categories")
    parser.add_argument("--split", "-s", type=float, default=0.9,
                        help="Train/validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
        print(f"Selected categories: {categories}")
    
    # Generate data
    print(f"\nGenerating {args.count} training examples...")
    examples = generate_all_training_data(args.count, categories)
    
    # Save
    result = save_training_data(examples, args.output, args.split)
    
    print(f"\n✓ Training data generation complete!")
    print(f"\nNext steps:")
    print(f"1. Review examples in {args.output}")
    print(f"2. Upload to S3:")
    print(f"   aws s3 cp {result['train_file']} s3://$ASTROLLAMA_BUCKET/training-data/")
    print(f"   aws s3 cp {result['val_file']} s3://$ASTROLLAMA_BUCKET/validation-data/")
    print(f"3. Start fine-tuning:")
    print(f"   python scripts/bedrock_finetune.py train --job-name astro-llama-v1")


if __name__ == "__main__":
    main()
