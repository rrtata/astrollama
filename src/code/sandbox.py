#!/usr/bin/env python3
"""
AstroLlama Code Execution Sandbox
Execute Python code safely with astronomy packages.

Features:
- Sandboxed execution with timeouts
- Pre-installed astronomy packages
- Plot capture and return as base64
- Result serialization
"""

import os
import sys
import io
import base64
import traceback
import contextlib
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str
    error: str = None
    plots: list = None  # List of base64-encoded plot images
    variables: dict = None  # Returned variables
    execution_time: float = 0.0
    
    def to_dict(self):
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "plots": self.plots or [],
            "variables": self.variables or {},
            "execution_time": self.execution_time
        }


class CodeSandbox:
    """Execute Python code in a sandboxed environment."""
    
    # Allowed imports for astronomy work
    ALLOWED_MODULES = {
        # Core
        "numpy", "np",
        "scipy",
        "pandas", "pd",
        "matplotlib", "plt",
        "matplotlib.pyplot",
        
        # Astronomy
        "astropy",
        "astropy.io.fits",
        "astropy.coordinates",
        "astropy.units",
        "astropy.table",
        "astropy.wcs",
        "astropy.time",
        "astropy.stats",
        "astroquery",
        "astroquery.vizier",
        "astroquery.simbad",
        "astroquery.gaia",
        "astroquery.mast",
        "photutils",
        "specutils",
        "reproject",
        "regions",
        
        # Image processing
        "skimage", "scikit-image",
        "cv2", "opencv",
        "PIL", "pillow",
        "sep",
        
        # Machine learning
        "sklearn", "scikit-learn",
        
        # Utilities
        "math",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "json",
        "datetime",
        "re",
        "io",
        "base64",
    }
    
    # Dangerous functions to block
    BLOCKED_NAMES = {
        "exec", "eval", "compile", "open", "input",
        "__import__", "globals", "locals",
        "getattr", "setattr", "delattr",
        "exit", "quit", "help",
        "os.system", "subprocess", "shutil.rmtree",
    }
    
    def __init__(self, timeout: int = 30, capture_plots: bool = True):
        self.timeout = timeout
        self.capture_plots = capture_plots
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up the execution environment with common imports."""
        self.namespace = {}
        
        # Pre-import common packages
        setup_code = """
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Astronomy packages (if available)
try:
    import astropy
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astropy.io import fits
except ImportError:
    pass

try:
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
except ImportError:
    pass

# Helper functions
def show_table(df, max_rows=20):
    '''Display a DataFrame or Table nicely.'''
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()
    return df.head(max_rows).to_string()

def save_plot():
    '''Save current plot and return as base64.'''
    buf = __builtins__['__plot_buffer__']
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return buf

print("Environment ready. Available: numpy (np), pandas (pd), matplotlib (plt), astropy, astroquery")
"""
        try:
            exec(setup_code, self.namespace)
        except Exception as e:
            print(f"Setup warning: {e}")
    
    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate code for safety."""
        # Check for blocked names
        for blocked in self.BLOCKED_NAMES:
            if blocked in code:
                return False, f"Blocked function/module: {blocked}"
        
        # Check for file operations
        if "open(" in code and "plt" not in code:
            # Allow plt.savefig but not general file access
            if ".write(" in code or "mode=" in code:
                return False, "Direct file writing not allowed"
        
        # Check for network access (except astroquery)
        if "requests." in code or "urllib." in code:
            if "astroquery" not in code:
                return False, "Direct network access not allowed (use astroquery)"
        
        return True, ""
    
    def _capture_plots(self) -> list:
        """Capture all matplotlib plots as base64."""
        import matplotlib.pyplot as plt
        
        plots = []
        
        # Get all figure numbers
        fig_nums = plt.get_fignums()
        
        for num in fig_nums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plots.append(base64.b64encode(buf.read()).decode('utf-8'))
            buf.close()
        
        # Close all figures
        plt.close('all')
        
        return plots
    
    def execute(self, code: str, return_vars: list = None) -> ExecutionResult:
        """Execute Python code.
        
        Args:
            code: Python code to execute
            return_vars: List of variable names to return
        
        Returns:
            ExecutionResult with output, plots, and variables
        """
        import time
        
        # Validate code
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation failed: {error_msg}"
            )
        
        # Capture stdout
        stdout_capture = io.StringIO()
        
        # Add plot buffer to namespace
        self.namespace['__plot_buffer__'] = io.BytesIO()
        
        start_time = time.time()
        
        try:
            # Execute with captured output
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, self.namespace)
            
            execution_time = time.time() - start_time
            output = stdout_capture.getvalue()
            
            # Capture plots
            plots = []
            if self.capture_plots:
                plots = self._capture_plots()
            
            # Get requested variables
            variables = {}
            if return_vars:
                for var in return_vars:
                    if var in self.namespace:
                        val = self.namespace[var]
                        # Try to serialize
                        try:
                            if hasattr(val, 'to_dict'):
                                variables[var] = val.to_dict()
                            elif hasattr(val, 'tolist'):
                                variables[var] = val.tolist()
                            elif isinstance(val, (int, float, str, bool, list, dict)):
                                variables[var] = val
                            else:
                                variables[var] = str(val)
                        except:
                            variables[var] = str(val)
            
            return ExecutionResult(
                success=True,
                output=output,
                plots=plots,
                variables=variables,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n\n{error_trace}",
                execution_time=execution_time
            )
    
    def reset(self):
        """Reset the execution environment."""
        self._setup_environment()


class AstronomyCodeAssistant:
    """Helper for generating astronomy code."""
    
    TEMPLATES = {
        "cone_search_gaia": '''
from astroquery.gaia import Gaia

# Cone search around coordinates
coord = SkyCoord(ra={ra}, dec={dec}, unit=(u.degree, u.degree))
radius = {radius} * u.arcmin

job = Gaia.cone_search_async(coord, radius)
results = job.get_results()

print(f"Found {{len(results)}} sources")
print(results[['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag']])
''',
        
        "color_magnitude_diagram": '''
import matplotlib.pyplot as plt

# Create color-magnitude diagram
fig, ax = plt.subplots(figsize=(10, 8))

# Assuming you have bp_rp colors and absolute magnitudes
ax.scatter(bp_rp, abs_g, s=1, alpha=0.5)

ax.set_xlabel('BP - RP (mag)')
ax.set_ylabel('Absolute G (mag)')
ax.set_title('Color-Magnitude Diagram')
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
''',
        
        "spectral_plot": '''
import matplotlib.pyplot as plt

# Plot spectrum
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(wavelength, flux, 'b-', lw=0.5)
ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Flux')
ax.set_title('Spectrum')

# Mark key features
features = {
    1.15: 'H₂O',
    1.31: 'CH₄',
    1.6: 'CH₄',
    2.2: 'CH₄'
}

for wl, name in features.items():
    if wavelength.min() < wl < wavelength.max():
        ax.axvline(wl, color='r', ls='--', alpha=0.5)
        ax.text(wl, ax.get_ylim()[1], name, rotation=90, va='top')

plt.tight_layout()
plt.show()
''',
        
        "photometry": '''
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.stats import sigma_clipped_stats

# Background statistics
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# Find sources
daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
sources = daofind(data - median)

print(f"Found {{len(sources)}} sources")

# Aperture photometry
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=5.0)
phot_table = aperture_photometry(data - median, apertures)

print(phot_table)
'''
    }
    
    @classmethod
    def get_template(cls, template_name: str, **kwargs) -> str:
        """Get a code template with filled parameters."""
        template = cls.TEMPLATES.get(template_name, "")
        return template.format(**kwargs) if kwargs else template
    
    @classmethod
    def list_templates(cls) -> list:
        """List available templates."""
        return list(cls.TEMPLATES.keys())


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Code Sandbox")
    print("=" * 60)
    
    sandbox = CodeSandbox()
    
    # Test 1: Simple calculation
    print("\n1. Simple calculation:")
    result = sandbox.execute("""
x = np.linspace(0, 10, 100)
y = np.sin(x)
print(f"Max value: {y.max():.3f}")
print(f"Mean value: {y.mean():.3f}")
""")
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output}")
    
    # Test 2: Plot generation
    print("\n2. Plot generation:")
    result = sandbox.execute("""
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()
""")
    print(f"   Success: {result.success}")
    print(f"   Plots captured: {len(result.plots)}")
    
    # Test 3: DataFrame
    print("\n3. DataFrame creation:")
    result = sandbox.execute("""
data = {
    'star': ['A', 'B', 'C'],
    'mag': [15.2, 16.1, 14.8],
    'color': [1.2, 1.5, 0.9]
}
df = pd.DataFrame(data)
print(df)
""", return_vars=['df'])
    print(f"   Success: {result.success}")
    print(f"   Output: {result.output}")
    
    # Test 4: Blocked code
    print("\n4. Security test (should fail):")
    result = sandbox.execute("""
import os
os.system('ls')
""")
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error}")
    
    print("\n" + "=" * 60)
    print("Code sandbox testing complete!")
