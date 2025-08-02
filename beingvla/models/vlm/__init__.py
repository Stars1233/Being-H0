# VLM adapter modules

from .base import BaseVLMAdapter
from .registry import VLMAdapterRegistry, register_vlm_adapter, create_vlm_adapter, list_vlm_adapters

# Import adapters to register them with protection against circular imports
def _safe_import_adapters():
    """Safely import and register VLM adapters with error handling."""
    try:
        # Import built-in adapters to trigger their registration
        from . import internvl_adapter  # This will trigger the registration - ignore Pylance warning
    except ImportError as e:
        # Log the error but don't fail the entire module import
        import warnings
        warnings.warn(f"Failed to import built-in VLM adapter: {e}")

# Perform safe adapter registration
_safe_import_adapters()

__all__ = [
    'BaseVLMAdapter',
    'VLMAdapterRegistry',
    'register_vlm_adapter', 
    'create_vlm_adapter',
    'list_vlm_adapters',
]