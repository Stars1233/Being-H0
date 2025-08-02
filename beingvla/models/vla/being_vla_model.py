# BeingVLA Model Orchestrator
#
# This module implements the main BeingVLAModel class that orchestrates VLM and Motion adapters
# to create a modular, extensible BeingVLA (Vision-Language-Action) system.

import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import GenerationConfig, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .config import BeingVLAConfig
from ..vlm.base import BaseVLMAdapter
from ..vlm.registry import create_vlm_adapter
from ..motion.base import BaseMotionAdapter
from ..motion.mano_adapter import ManoMotionAdapter
from .training_mixins import BeingVLATrainingMixin
from .generation_mixins import BeingVLAGenerationMixin
from ...utils.constants import DEEPSPEED_NO_SPLIT_MODULES

logger = logging.get_logger(__name__)


class BeingVLAModel(PreTrainedModel, BeingVLATrainingMixin, BeingVLAGenerationMixin):
    """
    Modular BeingVLA (Vision-Language-Action) Model Orchestrator.
    
    This class serves as a clean orchestrator that combines VLM and Motion adapters
    with specialized training and generation mixins to create a unified BeingVLA system.
    
    The modular design allows easy replacement of VLM backends (InternVL, LLaVA, QwenVL)
    and motion models (MANO, SMPL) while maintaining backward compatibility with the
    original InternVLMotionModel interface.
    
    Architecture:
        BeingVLAModel (orchestrator)
        ├── VLMAdapter (vision + language processing)  
        ├── MotionAdapter (motion processing)
        ├── BeingVLATrainingMixin (training strategies)
        └── BeingVLAGenerationMixin (generation strategies)
    """
    
    config_class = BeingVLAConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'  # Match original for DeepSpeed compatibility
    _no_split_modules = DEEPSPEED_NO_SPLIT_MODULES  # Match original exactly
    
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    
    def __init__(
        self, 
        config: BeingVLAConfig, 
        vlm_adapter: Optional[BaseVLMAdapter] = None,
        motion_adapter: Optional[BaseMotionAdapter] = None,
        use_flash_attn: bool = True,
        build_motion_model: bool = False
    ):
        super().__init__(config)
        
        # Store key attributes directly like the original (for DeepSpeed compatibility)
        # Handle vision_config - check multiple possible locations
        vision_config = None
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
        elif hasattr(config, 'vlm_config') and isinstance(config.vlm_config, dict):
            vision_config = config.vlm_config.get('vision_config')
        
        if vision_config is None:
            raise ValueError("No vision_config found in the configuration")
            
        if isinstance(vision_config, dict):
            image_size = getattr(config, 'force_image_size', None) or vision_config.get('image_size', 448)
            patch_size = vision_config.get('patch_size', 14)
        else:
            image_size = getattr(config, 'force_image_size', None) or vision_config.image_size
            patch_size = vision_config.patch_size
        
        # Store patch_size for compatibility, but other InternVL-specific attributes
        # should be accessed through the VLM adapter
        self.patch_size = patch_size
        
        # Handle llm_config - check multiple possible locations
        llm_config = None
        if hasattr(config, 'llm_config'):
            llm_config = config.llm_config
        elif hasattr(config, 'vlm_config') and isinstance(config.vlm_config, dict):
            llm_config = config.vlm_config.get('llm_config')
            
        if llm_config is None:
            raise ValueError("No llm_config found in the configuration")
            
        if isinstance(llm_config, dict):
            self.llm_arch_name = llm_config.get('architectures', [''])[0]
        else:
            self.llm_arch_name = llm_config.architectures[0]
        
        # Initialize VLM adapter
        if vlm_adapter is not None:
            self.vlm_adapter = vlm_adapter
        else:
            self.vlm_adapter = self._create_vlm_adapter(config, use_flash_attn)
        
        # Initialize Motion adapter
        if motion_adapter is not None:
            self.motion_adapter = motion_adapter
        else:
            self.motion_adapter = self._create_motion_adapter(config, build_motion_model)
        
        # Initialize mixin attributes
        self.__init_training_attributes__()
        self.__init_generation_attributes__()
        
        # Initialize robot alignment modules if needed
        self._init_robot_alignment_modules(config)
        
        # Robot alignment specific attributes
        self.prop_context_token_id = None
        self.action_chunk_token_ids = []
    
    @property
    def num_image_token(self):
        """Get number of image tokens from the VLM adapter."""
        if hasattr(self.vlm_adapter, 'num_image_token'):
            return self.vlm_adapter.num_image_token
        # Fallback calculation if adapter doesn't provide it
        force_image_size = self.config.vlm_config.get('force_image_size', 448)
        return int((force_image_size // self.patch_size) ** 2 * 0.25)
    
    def _init_robot_alignment_modules(self, config: BeingVLAConfig):
        """Initialize robot alignment modules for post-training tasks."""
        # Only initialize if explicitly enabled
        if hasattr(config, 'enable_robot_alignment') and config.enable_robot_alignment:
            llm_hidden_size = self.vlm_adapter.language_model.config.hidden_size
            
            # Proprioception encoder
            self.proprio_encoder = nn.Sequential(
                nn.Linear(config.proprio_dim, llm_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(llm_hidden_size // 2, llm_hidden_size)
            )
            
            # Action decoder
            if config.gen_action_type == "action_token":
                self.action_decoder = nn.Sequential(
                    nn.Linear(llm_hidden_size, llm_hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(llm_hidden_size // 2, config.action_dim)
                )
            else:
                self.action_decoder = nn.Sequential(
                    nn.Linear(llm_hidden_size, llm_hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(llm_hidden_size // 2, config.action_dim * config.action_chunk_length)
                )
        else:
            self.proprio_encoder = None
            self.action_decoder = None
    
    def _create_vlm_adapter(self, config: BeingVLAConfig, use_flash_attn: bool) -> BaseVLMAdapter:
        """Create VLM adapter based on configuration using the adapter registry."""
        vlm_type = config.vlm_type.lower()
        
        try:
            # Use the registry to create the adapter
            return create_vlm_adapter(vlm_type, config, use_flash_attn=use_flash_attn)
        except Exception as e:
            # Enhanced error handling with fallback suggestions
            error_msg = (
                f"Failed to create VLM adapter of type '{vlm_type}'. "
                f"Original error: {str(e)}. "
                f"Please check that: "
                f"1) The adapter type is correct, "
                f"2) Required dependencies are installed, "
                f"3) Configuration is valid."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _create_motion_adapter(self, config: BeingVLAConfig, build_motion_model: bool) -> Optional[BaseMotionAdapter]:
        """Create Motion adapter based on configuration."""
        motion_type = config.motion_type.lower()
        
        if not build_motion_model or not config.motion_config:
            logger.warning('Motion model not built. Set build_motion_model=True to enable motion processing.')
            return None
        
        try:
            if motion_type == 'mano':
                # Validate motion config
                motion_config_data = config.get_motion_config()
                if not motion_config_data:
                    raise ValueError("Motion config is empty but motion model is requested")
                
                # Create MANO-compatible config for adapter
                motion_config = type('MotionConfig', (), {})()
                motion_config.motion_config = motion_config_data
                
                return ManoMotionAdapter(motion_config)
            else:
                available_types = ['mano']  # Can be extended
                raise NotImplementedError(
                    f"Motion adapter type '{motion_type}' is not implemented. "
                    f"Available types: {available_types}"
                )
        except Exception as e:
            error_msg = (
                f"Failed to create motion adapter of type '{motion_type}'. "
                f"Original error: {str(e)}. "
                f"Please check the motion configuration."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @property
    def vision_model(self):
        """Access to vision model for backward compatibility."""
        return self.vlm_adapter._vision_model
    
    @property
    def language_model(self):
        """Access to language model for backward compatibility."""
        return self.vlm_adapter._language_model
    
    @property
    def motion_model(self):
        """Access to motion model for backward compatibility."""
        return self.motion_adapter.motion_model if self.motion_adapter else None
    
    # num_image_token is now stored as a direct attribute like the original
    
    @property
    def img_context_token_id(self) -> Optional[int]:
        """Image context token ID."""
        return self.vlm_adapter.img_context_token_id
    
    @img_context_token_id.setter
    def img_context_token_id(self, value: int):
        """Set image context token ID."""
        self.vlm_adapter.img_context_token_id = value
    
    def extract_feature(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Extract visual features from pixel values."""
        return self.vlm_adapter.extract_feature(pixel_values)
    
    
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
        # Robot alignment specific parameters
        proprioception_values: Optional[torch.FloatTensor] = None,
        action_labels: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass combining VLM and motion processing, with optional robot alignment support."""
        
        # Handle proprioception encoding if in robot alignment mode
        proprio_embeds = None
        if proprioception_values is not None and self.proprio_encoder is not None:
            proprio_embeds = self.proprio_encoder(proprioception_values)
            # Pass proprio_embeds to VLM adapter via kwargs
            kwargs['proprio_embeds'] = proprio_embeds
            kwargs['prop_context_token_id'] = self.prop_context_token_id
            
        # For robot alignment mode, we need to get hidden states
        if action_labels is not None and self.action_decoder is not None:
            output_hidden_states = True
        
        # Standard VLM forward pass
        outputs = self.vlm_adapter.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels if action_labels is None else None,  # No language modeling loss in alignment mode
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            statistics=statistics,
            loss_weight=loss_weight,
            loss_reduction_all_gather=loss_reduction_all_gather,
            **kwargs
        )
        
        # Handle robot alignment mode - compute action loss
        if action_labels is not None and self.action_decoder is not None:
            outputs = self._compute_action_loss(outputs, labels, action_labels)
        # Apply motion-specific training logic if enabled
        elif labels is not None and self.optimize_rate > 0.0 and self.motion_adapter is not None:
            outputs = self._apply_motion_training_logic(outputs, labels)
        
        return outputs
    
    def _compute_action_loss(self, outputs, labels, action_labels):
        """Compute action prediction loss for robot alignment tasks."""
        from torch.nn import L1Loss, MSELoss
        
        # Get loss function based on config
        if hasattr(self.config, 'loss_func'):
            if self.config.loss_func == 'l1':
                loss_fct = L1Loss()
            elif self.config.loss_func == 'mse':
                loss_fct = MSELoss()
            else:
                loss_fct = L1Loss()  # default
        else:
            loss_fct = L1Loss()
        
        # Extract hidden states at action token positions
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
            
            # Create mask for action chunk tokens
            action_chunk_mask = torch.zeros_like(labels, dtype=torch.bool)
            for token_id in self.action_chunk_token_ids:
                action_chunk_mask |= (labels == token_id)
            
            # Get action hidden states
            action_hidden_states = last_hidden_state[action_chunk_mask]
            
            if action_hidden_states.shape[0] > 0:
                # Decode actions from hidden states
                predicted_actions = self.action_decoder(action_hidden_states)
                
                # Reshape target actions
                target_actions = action_labels.reshape(-1, self.config.action_dim)
                
                # Compute action loss
                action_loss = loss_fct(
                    predicted_actions,
                    target_actions.to(predicted_actions.device, dtype=predicted_actions.dtype)
                )
                
                # Replace the loss with action loss
                outputs.loss = action_loss
            else:
                # No action tokens found, set loss to 0
                outputs.loss = torch.tensor(0.0, device=outputs.logits.device)
        else:
            # No hidden states available
            outputs.loss = torch.tensor(0.0, device=outputs.logits.device)
            
        return outputs
    
    @torch.no_grad()
    def get_action(
        self,
        tokenizer,
        pixel_values: torch.FloatTensor,
        proprioception_values: torch.FloatTensor,
        task_description: str = "",
        action_chunk_length: int = 16,
        verbose: bool = False,
    ):
        """Get predicted actions for robot control (inference mode)."""
        if self.proprio_encoder is None or self.action_decoder is None:
            raise ValueError("Robot alignment modules not initialized. This model was not configured for robot control.")
        
        # Prepare the prompt with image and special tokens
        # Add image tokens first
        num_patches = pixel_values.size(0)
        image_tokens = f"<img>{'<IMG_CONTEXT>' * self.num_image_token * num_patches}</img>"
        prompt = f"{image_tokens} {task_description} <PROP_CONTEXT>"
        for i in range(action_chunk_length):
            prompt += f" <ACTION_CHUNK_{i}>"
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(pixel_values.device)
        attention_mask = inputs['attention_mask'].to(pixel_values.device)
        
        # Calculate image_flags for the pixel_values
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long, device=pixel_values.device)
        
        # Get model outputs with hidden states
        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            proprioception_values=proprioception_values.unsqueeze(0) if proprioception_values.dim() == 1 else proprioception_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract hidden states at action token positions
        last_hidden_state = outputs.hidden_states[-1]
        
        # Find action token positions
        action_chunk_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.action_chunk_token_ids:
            action_chunk_mask |= (input_ids == token_id)
        
        # Get action hidden states
        action_hidden_states = last_hidden_state[action_chunk_mask]
        
        if action_hidden_states.shape[0] > 0:
            # Decode actions
            predicted_actions = self.action_decoder(action_hidden_states)
            
            # Reshape to (action_chunk_length, action_dim)
            if self.config.gen_action_type == "action_token":
                predicted_actions = predicted_actions.reshape(action_chunk_length, -1)
            else:
                predicted_actions = predicted_actions.reshape(-1, self.config.action_dim)
            
            if verbose:
                logger.info(f"Predicted actions shape: {predicted_actions.shape}")
            
            return predicted_actions
        else:
            logger.warning("No action tokens found in the model output")
            return None
    
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Standard generation without motion constraints."""
        return self.vlm_adapter.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            stopping_criteria=stopping_criteria,
            **generate_kwargs,
        )
    
    
    # Motion-related methods for backward compatibility
    def motion_block_id_to_mano(self, motion_block_ids, offset=False, denormalize=True, return_list=False):
        """Convert motion block IDs to MANO parameters."""
        if self.motion_adapter is None:
            raise RuntimeError("Motion adapter not available")
        return self.motion_adapter.decode_motion(motion_block_ids, offset, denormalize, return_list)
    
    def mano_to_motion_block_id(self, mano_list, normalize=True, return_list=False):
        """Convert MANO parameters to motion block IDs."""
        if self.motion_adapter is None:
            raise RuntimeError("Motion adapter not available")
        result = self.motion_adapter.encode_motion(mano_list, normalize)
        if return_list:
            # Would need to implement return_list logic if needed
            return [result]
        return result
    
    # Property access for backward compatibility
    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()
    
    @property
    def motion_config(self):
        """Access motion config for backward compatibility with original code."""
        if self.motion_adapter is not None:
            return self.motion_adapter.motion_config
        return None
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    
    # Remove custom tie_weights to match original behavior exactly
    
    def _set_gradient_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing for memory efficiency."""
        # Apply to language model directly for DeepSpeed compatibility
        if hasattr(self.language_model, 'gradient_checkpointing_enable'):
            if enable:
                self.language_model.gradient_checkpointing_enable()
            else:
                self.language_model.gradient_checkpointing_disable()
        
        # Also apply to vision model if available
        if hasattr(self.vision_model, 'gradient_checkpointing_enable'):
            if enable:
                self.vision_model.gradient_checkpointing_enable()
            else:
                self.vision_model.gradient_checkpointing_disable()
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation - delegate to VLM adapter."""
        return self.vlm_adapter.prepare_inputs_for_generation(*args, **kwargs)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the model."""
        self._set_gradient_checkpointing(True)
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self._set_gradient_checkpointing(False)
    
    def _detect_checkpoint_format(self, state_dict):
        """Detect checkpoint format using VLM adapter's detection with validation.
        
        Args:
            state_dict: The state dictionary to analyze
            
        Returns:
            str: Format identifier (e.g., 'internvl', 'beingvla', 'deepspeed')
            
        Raises:
            ValueError: If state_dict is invalid
        """
        # Validate input
        if not isinstance(state_dict, dict):
            raise ValueError(f"state_dict must be a dictionary, got {type(state_dict)}")
        
        if not state_dict:
            raise ValueError("state_dict cannot be empty")
        
        # Sample keys for analysis
        keys = list(state_dict.keys())
        logger.info(f"Analyzing checkpoint with {len(keys)} keys")
        logger.debug(f"Sample keys: {keys[:5] if len(keys) > 5 else keys}")
        
        # Check for DeepSpeed format first
        deepspeed_keys = [k for k in keys if k.startswith('module.')]
        if deepspeed_keys:
            logger.info(f"Detected DeepSpeed format (found {len(deepspeed_keys)} module.* keys)")
            return 'deepspeed'
        
        # Check if it's already in BeingVLA format
        beingvla_keys = [k for k in keys if k.startswith(('vlm_adapter.', 'motion_adapter.'))]
        if beingvla_keys:
            logger.info(f"Detected BeingVLA format (found {len(beingvla_keys)} adapter keys)")
            return 'beingvla'
        
        # Let the VLM adapter detect its own format
        if hasattr(self.vlm_adapter, 'detect_checkpoint_format'):
            try:
                format_type = self.vlm_adapter.detect_checkpoint_format(state_dict)
                if format_type != 'unknown':
                    logger.info(f"VLM adapter detected format: {format_type}")
                    return format_type
            except Exception as e:
                logger.warning(f"VLM adapter format detection failed: {e}")
        
        # Log some key patterns for debugging
        key_patterns = {}
        for key in keys[:20]:  # Check first 20 keys
            parts = key.split('.')
            pattern = parts[0] if parts else 'root'
            key_patterns[pattern] = key_patterns.get(pattern, 0) + 1
        
        logger.warning(f"Unknown checkpoint format. Key patterns: {key_patterns}")
        return 'unknown'
    
    def _convert_checkpoint_format(self, state_dict, source_format):
        """Convert checkpoint from source format to BeingVLA format with validation.
        
        Args:
            state_dict: The state dict to convert
            source_format: The detected source format
            
        Returns:
            Converted state dict in BeingVLA format
            
        Raises:
            ValueError: If conversion fails or produces invalid result
        """
        logger.info(f"Converting checkpoint from {source_format} to BeingVLA format")
        
        # Validate inputs
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Invalid state_dict for conversion")
        
        if source_format == 'beingvla':
            logger.info("No conversion needed for BeingVLA format")
            return state_dict
        
        # Get key mappings with error handling
        try:
            key_mappings = self.vlm_adapter.get_state_dict_key_mapping()
        except Exception as e:
            logger.error(f"Failed to get VLM adapter key mappings: {e}")
            key_mappings = {}
        
        # Add motion adapter mappings if available
        if self.motion_adapter and hasattr(self.motion_adapter, 'get_state_dict_key_mapping'):
            try:
                motion_mappings = self.motion_adapter.get_state_dict_key_mapping()
                key_mappings.update(motion_mappings)
            except Exception as e:
                logger.warning(f"Failed to get motion adapter key mappings: {e}")
        else:
            # Default motion mappings for backward compatibility
            if self.motion_adapter:
                key_mappings.update({
                    'vq_model.': 'motion_adapter.vq_model.',
                    'mano_model.': 'motion_adapter.mano_model.',
                })
        
        # Perform conversion with validation
        new_state_dict = {}
        skip_keys = {'upsample_ratio', 'num_joints'}  # Scalar attributes to skip
        converted_count = 0
        
        for key, value in state_dict.items():
            # Skip scalar attributes
            if key in skip_keys:
                logger.debug(f"Skipping scalar attribute: {key}")
                continue
            
            # Validate tensor
            if not isinstance(value, torch.Tensor):
                logger.warning(f"Non-tensor value for key {key}: {type(value)}")
                
            new_key = key
            for old_prefix, new_prefix in key_mappings.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    converted_count += 1
                    logger.debug(f"Converted: {key} -> {new_key}")
                    break
                    
            new_state_dict[new_key] = value
        
        # Validation of conversion results
        if not new_state_dict:
            raise ValueError("Conversion resulted in empty state dict")
        
        logger.info(f"Conversion completed: {converted_count} keys converted, "
                   f"{len(new_state_dict)} total keys in result")
        
        # Check for critical components
        critical_components = ['vlm_adapter._vision_model', 'vlm_adapter._language_model']
        missing_components = []
        for component in critical_components:
            if not any(k.startswith(component) for k in new_state_dict.keys()):
                missing_components.append(component)
        
        if missing_components:
            logger.warning(f"Missing critical components after conversion: {missing_components}")
            
        return new_state_dict
    
    def _strip_deepspeed_prefix(self, state_dict):
        """Remove module. prefix from DeepSpeed checkpoints."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with automatic format detection and conversion.
        
        Supports:
        1. Original InternVL checkpoints (direct component keys)
        2. BeingVLA checkpoints (nested adapter keys)
        3. DeepSpeed checkpoints (with module. prefix)
        
        Args:
            state_dict: The state dictionary to load
            strict: Whether to strictly enforce that the keys match
            
        Returns:
            Result from parent's load_state_dict method
            
        Raises:
            ValueError: If checkpoint format detection or conversion fails
            RuntimeError: If loading fails
        """
        try:
            logger.info(f"Loading state dict with {len(state_dict)} keys")
            
            # Detect checkpoint format with validation
            format_type = self._detect_checkpoint_format(state_dict)
            logger.info(f"Detected checkpoint format: {format_type}")
            
            # Store original for recovery if needed
            original_state_dict = state_dict.copy() if len(state_dict) < 10000 else None
            
            # Convert format if needed
            if format_type == 'deepspeed':
                logger.info("Processing DeepSpeed checkpoint")
                state_dict = self._strip_deepspeed_prefix(state_dict)
                # Re-detect format after stripping
                format_type = self._detect_checkpoint_format(state_dict)
                logger.info(f"After stripping DeepSpeed prefix, format is: {format_type}")
            
            if format_type not in ['beingvla', 'unknown']:
                logger.info(f"Converting {format_type} checkpoint to BeingVLA format")
                state_dict = self._convert_checkpoint_format(state_dict, format_type)
                logger.info(f"Successfully converted {format_type} checkpoint to BeingVLA format")
            elif format_type == 'unknown':
                logger.warning(
                    "Unknown checkpoint format detected. Attempting to load as-is. "
                    "This may fail if key names don't match the model structure."
                )
            
            # Validate final state dict before loading
            if not state_dict:
                raise ValueError("State dict is empty after processing")
            
            # Attempt to load with enhanced error handling
            try:
                return super().load_state_dict(state_dict, strict=strict)
            except Exception as load_error:
                if not strict and original_state_dict is not None:
                    logger.warning(f"Strict loading failed ({load_error}), trying non-strict mode with original dict")
                    try:
                        return super().load_state_dict(original_state_dict, strict=False)
                    except Exception as fallback_error:
                        logger.error(f"Fallback loading also failed: {fallback_error}")
                        raise load_error
                else:
                    raise load_error
                    
        except Exception as e:
            error_msg = f"Failed to load state dict: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained BeingVLA model with comprehensive checkpoint support.
        
        This method handles:
        1. Sharded safetensors checkpoints (model-00001-of-00004.safetensors)
        2. Single safetensors files (model.safetensors)  
        3. PyTorch checkpoints (pytorch_model.bin, pytorch_model.pt)
        4. Original InternVL models with automatic format conversion
        
        Parameters:
            build_motion_model (bool): Whether to build motion processing models
            use_flash_attn (bool): Whether to use Flash Attention
            motion_code_path (str): Optional path to override motion codes (supports + separated paths)
        """
        # Extract BeingVLA-specific kwargs before passing to parent
        build_motion_model = kwargs.pop('build_motion_model', False)
        use_flash_attn = kwargs.pop('use_flash_attn', True)
        motion_code_path = kwargs.pop('motion_code_path', None)
        
        # Handle config loading manually to avoid the unpacking issue
        config = kwargs.pop('config', None)
        if config is None:
            # Load config manually to ensure compatibility
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )
        
        # Track motion_args_list for later use
        motion_args_list = None
        
        # Handle motion_code_path override if provided
        if motion_code_path and build_motion_model:
            logger.info(f"Overriding motion config with motion_code_path: {motion_code_path}")
            
            # Import here to avoid circular imports
            from ..motion.m2m.tokenizer.config import MotionArguments
            from ..motion.m2m.utils.misc import load_resume_args
            from dataclasses import asdict
            
            # Create temporary MotionArguments to load the config
            temp_motion_args = MotionArguments()
            temp_motion_args.motion_code_path = motion_code_path
            
            # Load motion arguments from the specified path(s)
            motion_args_list = load_resume_args(temp_motion_args)
            motion_config_list = [asdict(m_args) for m_args in motion_args_list]
            
            # Update the config with new motion configuration
            config.motion_config = motion_config_list
            if not hasattr(config, 'motion_type') or not config.motion_type:
                config.motion_type = 'mano'
            
            logger.info(f"Updated config with {len(motion_config_list)} motion configurations")
        
        # Create the model instance with our custom init parameters
        model = cls(
            config,
            use_flash_attn=use_flash_attn,
            build_motion_model=build_motion_model
        )
        
        # Load the state dict
        state_dict = None
        if os.path.isdir(pretrained_model_name_or_path):
            # Check for sharded safetensors first
            index_file = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                # Load sharded safetensors model
                from safetensors.torch import load_file
                import json
                
                logger.info(f"Loading sharded model from {pretrained_model_name_or_path}")
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                # Load all unique shard files
                shard_files = set(index['weight_map'].values())
                state_dict = {}
                
                for shard_file in sorted(shard_files):
                    shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                    logger.info(f"Loading shard: {shard_file}")
                    shard_dict = load_file(shard_path)
                    state_dict.update(shard_dict)
            else:
                # Check for single checkpoint files
                for filename in ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.pt']:
                    candidate = os.path.join(pretrained_model_name_or_path, filename)
                    if os.path.exists(candidate):
                        # Load state dict based on file format
                        if candidate.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state_dict = load_file(candidate)
                        else:
                            state_dict = torch.load(candidate, map_location='cpu', weights_only=False)
                        break
        
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded BeingVLA model from {pretrained_model_name_or_path}")
        else:
            logger.warning(f"No checkpoint found at {pretrained_model_name_or_path}, model initialized from config only")
        
        # Handle dtype conversion if specified
        torch_dtype = kwargs.get('torch_dtype', None)
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        
        # Update motion adapter config if motion_code_path was provided
        if motion_args_list is not None and hasattr(model, 'motion_adapter') and model.motion_adapter is not None:
            model.motion_adapter._motion_config = motion_args_list
            logger.info("Updated motion adapter with new motion configuration")
        
        return model
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model to a directory.
        
        This saves in BeingVLA format with nested adapter keys.
        When loading, the format will be auto-detected.
        """
        # Save using parent's save_pretrained
        super().save_pretrained(save_directory, **kwargs)
        
        # Log the format for clarity
        logger.info(f"Saved BeingVLA model to {save_directory} in BeingVLA format")