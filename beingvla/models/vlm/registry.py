# VLM Adapter Registry System
#
# This module provides a registry pattern for VLM adapters, allowing new VLMs to be added
# without modifying core VLA code. Each VLM adapter registers itself with the registry.

from typing import Dict, Type, Optional, Any
from .base import BaseVLMAdapter


class VLMAdapterRegistry:
    """
    Registry for VLM adapters that allows dynamic registration and creation of adapters.
    
    This registry pattern enables adding new VLM adapters without modifying the core
    VLA model code. Each adapter type registers itself with a unique name.
    """
    
    _registry: Dict[str, Type[BaseVLMAdapter]] = {}
    _config_converters: Dict[str, callable] = {}
    _registration_lock = False  # Import protection flag
    
    @classmethod
    def register(
        cls, 
        adapter_type: str, 
        adapter_class: Type[BaseVLMAdapter],
        config_converter: Optional[callable] = None
    ) -> None:
        """
        Register a VLM adapter class with the registry.
        
        Args:
            adapter_type: Unique identifier for the adapter (e.g., 'internvl', 'llava')
            adapter_class: The adapter class that implements BaseVLMAdapter
            config_converter: Optional function to convert legacy configs to BeingVLAConfig
        """
        # Import protection: prevent registration during circular imports
        if cls._registration_lock:
            import warnings
            warnings.warn(f"Skipping registration of '{adapter_type}' due to import protection")
            return
            
        if adapter_type in cls._registry:
            # Allow re-registration with warning instead of error for robustness
            import warnings
            warnings.warn(f"Adapter type '{adapter_type}' is already registered, overwriting")
        
        cls._registry[adapter_type] = adapter_class
        if config_converter:
            cls._config_converters[adapter_type] = config_converter
    
    @classmethod
    def create_adapter(
        cls, 
        adapter_type: str, 
        config: Any,
        **kwargs
    ) -> BaseVLMAdapter:
        """
        Create a VLM adapter instance based on the adapter type.
        
        Args:
            adapter_type: The type of adapter to create
            config: Configuration object for the adapter
            **kwargs: Additional arguments to pass to the adapter constructor
            
        Returns:
            Instance of the requested VLM adapter
            
        Raises:
            ValueError: If the adapter type is not registered or creation fails
            RuntimeError: If adapter initialization fails
        """
        if adapter_type not in cls._registry:
            available = ', '.join(cls._registry.keys()) if cls._registry else 'none'
            raise ValueError(
                f"Unknown VLM adapter type '{adapter_type}'. "
                f"Available types: {available}. "
                f"Make sure the adapter module is imported and registered."
            )
        
        adapter_class = cls._registry[adapter_type]
        
        try:
            # Validate config before adapter creation
            if config is None:
                raise ValueError(f"Config cannot be None for adapter type '{adapter_type}'")
            
            # Create adapter with enhanced error context
            return adapter_class(config, **kwargs)
            
        except Exception as e:
            # Enhanced error reporting with context
            error_msg = (
                f"Failed to create {adapter_type} adapter. "
                f"Error: {str(e)}. "
                f"Config type: {type(config).__name__}. "
                f"Available kwargs: {list(kwargs.keys())}"
            )
            
            # Re-raise with better context
            if isinstance(e, (ValueError, TypeError)):
                raise ValueError(error_msg) from e
            else:
                raise RuntimeError(error_msg) from e
    
    @classmethod
    def get_config_converter(cls, adapter_type: str) -> Optional[callable]:
        """
        Get the config converter function for a specific adapter type.
        
        Args:
            adapter_type: The adapter type
            
        Returns:
            Config converter function if registered, None otherwise
        """
        return cls._config_converters.get(adapter_type)
    
    @classmethod
    def list_adapters(cls) -> list[str]:
        """
        List all registered adapter types.
        
        Returns:
            List of registered adapter type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, adapter_type: str) -> bool:
        """
        Check if an adapter type is registered.
        
        Args:
            adapter_type: The adapter type to check
            
        Returns:
            True if the adapter type is registered, False otherwise
        """
        return adapter_type in cls._registry
    
    @classmethod
    def set_registration_lock(cls, locked: bool) -> None:
        """
        Set the registration lock to prevent circular import issues.
        
        Args:
            locked: Whether to lock registration
        """
        cls._registration_lock = locked


# Convenience functions for the registry
def register_vlm_adapter(
    adapter_type: str, 
    adapter_class: Type[BaseVLMAdapter],
    config_converter: Optional[callable] = None
) -> None:
    """
    Register a VLM adapter with the global registry.
    
    This function is typically used as a decorator to automatically register
    VLM adapters when their module is imported.
    
    Args:
        adapter_type: Unique identifier for the adapter (e.g., 'internvl', 'llava')
        adapter_class: The adapter class that implements BaseVLMAdapter
        config_converter: Optional function to convert legacy configs to BeingVLAConfig format
    
    Example:
        >>> @register_vlm_adapter('my_vlm', config_converter=convert_my_vlm_config)
        >>> class MyVLMAdapter(BaseVLMAdapter):
        ...     def __init__(self, config):
        ...         # Implementation
        ...         pass
    
    Note:
        The adapter will be automatically available for use with BeingVLAModel
        once its module is imported. No additional registration is needed.
    """
    VLMAdapterRegistry.register(adapter_type, adapter_class, config_converter)


def create_vlm_adapter(adapter_type: str, config: Any, **kwargs) -> BaseVLMAdapter:
    """Create a VLM adapter from the global registry."""
    return VLMAdapterRegistry.create_adapter(adapter_type, config, **kwargs)


def list_vlm_adapters() -> list[str]:
    """List all registered VLM adapter types."""
    return VLMAdapterRegistry.list_adapters()