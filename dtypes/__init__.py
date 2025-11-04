"""
Custom data types module for GPT-OSS.

This module provides:
- bfloat16: BFloat16 array operations and conversions
- linear_layer: Optimized linear layer implementations using oneDNN
- linear_layer_torch: (Optional) PyTorch-based linear layer with BF16 support
"""

# Import core modules
try:
    from . import bfloat16
    from . import linear_layer
    _CORE_AVAILABLE = True
except ImportError as e:
    # Extensions not built yet
    _CORE_AVAILABLE = False
    import warnings
    warnings.warn(f"Core dtypes extensions not available: {e}")

# Optionally import torch module if available
try:
    from . import linear_layer_torch
    _TORCH_AVAILABLE = True
except ImportError:
    # Torch module not built (PyTorch not installed during build) or runtime linking issue
    _TORCH_AVAILABLE = False
    linear_layer_torch = None

__all__ = ['bfloat16', 'linear_layer', 'linear_layer_torch']

def has_torch_support():
    """Check if PyTorch-based linear layer is available."""
    return _TORCH_AVAILABLE

