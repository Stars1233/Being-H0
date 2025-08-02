# BeingVLA Configuration System
#
# This module provides a unified configuration system for BeingVLA models that can handle
# different VLM and Motion adapter combinations while maintaining backward compatibility.

import copy
from typing import Dict, List, Optional, Union, Any
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

from ..vlm.registry import VLMAdapterRegistry

logger = logging.get_logger(__name__)


class BeingVLAConfig(PretrainedConfig):
    """
    Unified configuration for BeingVLA models.
    
    This configuration class supports different VLM and Motion adapter combinations
    while maintaining backward compatibility with existing InternVLMotionConfig.
    
    Args:
        vlm_type (str): Type of VLM adapter to use. Options: 'internvl', 'llava', 'qwen_vl'
        motion_type (str): Type of motion adapter. Options: 'mano' for hand motion, None to disable
        vlm_config (Dict): Configuration for the VLM adapter
        motion_config (Union[Dict, List[Dict]]): Configuration for motion models
        proprio_dim (int): Dimension of proprioception input for robot control
        action_dim (int): Dimension of action output for robot control
        action_chunk_length (int): Number of future actions to predict in robot control
        loss_func (str): Loss function for robot alignment. Options: 'l1', 'l2', 'smooth_l1'
        gen_action_type (str): Action generation strategy. Options: 'action_token', 'prop_hidden', 'last_hidden'
        enable_robot_alignment (bool): Whether to enable robot control modules
    
    Example:
        >>> # Motion generation configuration
        >>> config = BeingVLAConfig(
        ...     vlm_type='internvl',
        ...     motion_type='mano',
        ...     motion_config=[{
        ...         'use_part': 'wrist',
        ...         'quantizer': 'grvq',
        ...         'n_codes': 4096
        ...     }]
        ... )
        
        >>> # Robot control configuration
        >>> config = BeingVLAConfig(
        ...     vlm_type='internvl',
        ...     enable_robot_alignment=True,
        ...     proprio_dim=13,
        ...     action_dim=13,
        ...     gen_action_type='action_token'
        ... )
    """
    
    model_type = 'beingvla'
    is_composition = True
    
    def __init__(
        self,
        vlm_type: str = 'internvl',
        motion_type: str = 'mano',
        vlm_config: Optional[Dict] = None,
        motion_config: Optional[Union[Dict, List[Dict]]] = None,
        # Robot alignment specific parameters
        proprio_dim: int = 13,
        action_dim: int = 13,
        action_chunk_length: int = 16,
        loss_func: str = "l1",
        gen_action_type: str = "prop_hidden",  # action_token, prop_hidden, last_hidden
        enable_robot_alignment: bool = False,  # Explicitly enable robot alignment modules
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Adapter type configuration
        self.vlm_type = vlm_type
        self.motion_type = motion_type
        
        # VLM configuration (standard nested format)
        self.vlm_config = vlm_config if vlm_config is not None else {}
        
        # Robot alignment specific parameters
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_chunk_length = action_chunk_length
        self.loss_func = loss_func
        self.gen_action_type = gen_action_type
        self.enable_robot_alignment = enable_robot_alignment
        
        # Motion configuration
        self.motion_config = self._validate_motion_config(motion_config)
    
    def _validate_motion_config(self, motion_config):
        """Validate and normalize motion configuration."""
        if motion_config is None:
            return {}
        
        if isinstance(motion_config, dict):
            return motion_config
        elif isinstance(motion_config, (list, tuple)):
            flag = False
            for item in motion_config:
                if not isinstance(item, dict):
                    flag = True
                    break
            if flag:
                logger.warning("motion_config should be a list of dicts. Initializing as an empty dict.")
                return {}
            else:
                return motion_config
        else:
            logger.warning("motion_config should be a dict or list of dicts. Initializing as an empty dictionary.")
            return {}
    
    @classmethod
    def from_legacy_config(cls, legacy_config: Any, vlm_type: Optional[str] = None) -> 'BeingVLAConfig':
        """
        Create BeingVLAConfig from a legacy VLM-specific config using the adapter registry.
        
        Args:
            legacy_config: Legacy configuration object (e.g., InternVLMotionConfig)
            vlm_type: Optional VLM type override. If not provided, will try to infer from config.
            
        Returns:
            BeingVLAConfig instance with equivalent configuration
        """
        # Try to infer VLM type from config if not provided
        if vlm_type is None:
            # Check config type name for hints
            config_type_name = type(legacy_config).__name__.lower()
            if 'internvl' in config_type_name:
                vlm_type = 'internvl'
            elif 'llava' in config_type_name:
                vlm_type = 'llava'
            else:
                # Default to internvl for backward compatibility
                vlm_type = 'internvl'
                logger.warning(f"Could not infer VLM type from config {type(legacy_config).__name__}, defaulting to 'internvl'")
        
        # Get the config converter from the registry
        converter = VLMAdapterRegistry.get_config_converter(vlm_type)
        if converter is None:
            raise ValueError(f"No config converter registered for VLM type '{vlm_type}'")
        
        # Convert the legacy config
        config_dict = converter(legacy_config)
        return cls(**config_dict)
    
    def get_vlm_config(self) -> Dict:
        """Get configuration for VLM adapter."""
        return self.vlm_config
    
    def get_motion_config(self) -> Union[Dict, List[Dict]]:
        """Get configuration for Motion adapter."""
        return self.motion_config
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        
        Returns:
            Dict[str, any]: Dictionary of all attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output['model_type'] = self.__class__.model_type
        
        # Remove private attributes
        output = {k: v for k, v in output.items() if not k.startswith('_')}
        
        return output
    
    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs):
        """
        Instantiates a VLAConfig from a Python dictionary of parameters.
        
        Args:
            config_dict: Dictionary of configuration parameters
            **kwargs: Additional parameters passed to the constructor
            
        Returns:
            VLAConfig: The configuration instance
        """
        config_dict = config_dict.copy()
        
        # InternVL-specific attributes that belong in vlm_config
        internvl_attrs = {
            'force_image_size', 'downsample_ratio', 'use_thumbnail', 
            'ps_version', 'template', 'select_layer'
        }
        
        # Migrate any top-level InternVL attributes to vlm_config for proper encapsulation
        if any(attr in config_dict for attr in internvl_attrs):
            if 'vlm_config' not in config_dict:
                config_dict['vlm_config'] = {}
            elif not isinstance(config_dict['vlm_config'], dict):
                config_dict['vlm_config'] = {}
            
            for attr in internvl_attrs:
                if attr in config_dict:
                    config_dict['vlm_config'][attr] = config_dict.pop(attr)
                    logger.debug(f"Migrated InternVL attribute '{attr}' to vlm_config")
        
        # Handle kwargs that may conflict with config_dict
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in config_dict:
                filtered_kwargs[key] = value
            else:
                logger.debug(f"Parameter '{key}' specified in both config_dict and kwargs. Using config_dict value.")
        
        return cls(**config_dict, **filtered_kwargs)


# ManoAdapterConfig removed - motion configuration is now handled directly in BeingVLA config
# Motion adapter configuration is integrated into the main config system for better consistency