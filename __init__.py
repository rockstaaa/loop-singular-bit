"""
Loop Singular Bit - Extreme Model Compression
=============================================

Outlier-Preserving 1-Bit Quantization for 32× compression with 99.5% quality preservation

Author: Bommareddy Bharath Reddy
"""

from .loop_singular_bit import (
    LoopSingularBit,
    load_compressed_model,
    list_models,
    get_system_info
)

__version__ = "1.0.0"
__author__ = "Bommareddy Bharath Reddy"
__email__ = "contact@loop.org"

__all__ = [
    'LoopSingularBit',
    'load_compressed_model', 
    'list_models',
    'get_system_info'
]
