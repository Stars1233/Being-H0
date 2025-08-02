# InternVL Adapter Implementation
# 
# This module extracts the VLM-specific logic from the original InternVLMotionModel
# to create a modular InternVL adapter that can be used with the VLA framework.

import warnings
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    GenerationConfig, Qwen2ForCausalLM, LogitsProcessor, StoppingCriteriaList
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from ...utils.conversation import get_conv_template
from ...utils.constants import (
    FLASH_ATTN_SUPPORTED_DTYPES, FLASH_ATTN_CREATION_SAFE_DTYPE
)
from .internvl.modeling_intern_vit import InternVisionModel, has_flash_attn
from .internvl.configuration_intern_vit import InternVisionConfig
from peft import LoraConfig, get_peft_model

from .base import BaseVLMAdapter
from .registry import register_vlm_adapter

logger = logging.get_logger(__name__)


# InternVLAdapterConfig removed - using BeingVLAConfig directly for better consolidation


def version_cmp(v1, v2, op='eq'):
    """Compare two version strings."""
    import operator
    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLAdapter(BaseVLMAdapter):
    """
    InternVL adapter for the VLA framework.
    
    This adapter extracts the vision and language processing logic from the
    original InternVLMotionModel, making InternVL a replaceable component.
    """
    
    def __init__(self, config, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)
        
        # Import transformers version check
        import transformers
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        
        # Determine target dtype from config or use bfloat16 as default
        self.target_dtype = getattr(config, 'torch_dtype', torch.bfloat16)
        if self.target_dtype is None:
            self.target_dtype = torch.bfloat16
        
        # Extract VLM config - prioritize vlm_config if available
        vlm_config = getattr(config, 'vlm_config', {})
        
        # Vision configuration - check multiple locations for backward compatibility
        vision_config = vlm_config.get('vision_config')
        if vision_config is None and hasattr(config, 'vision_config'):
            # Backward compatibility: check top-level for old checkpoints
            vision_config = config.vision_config
        
        if vision_config is None:
            raise ValueError("No vision_config found in the configuration")
            
        if isinstance(vision_config, dict):
            # Extract image size from multiple possible locations
            image_size = (vlm_config.get('force_image_size') or 
                         getattr(config, 'force_image_size', None) or 
                         vision_config.get('image_size', 448))
            patch_size = vision_config.get('patch_size', 14)
        else:
            image_size = (vlm_config.get('force_image_size') or 
                         getattr(config, 'force_image_size', None) or 
                         vision_config.image_size)
            patch_size = vision_config.patch_size
        
        # Extract InternVL-specific parameters from vlm_config
        self.patch_size = patch_size
        self.select_layer = vlm_config.get('select_layer', getattr(config, 'select_layer', -1))
        self.template = vlm_config.get('template', getattr(config, 'template', 'internvl2_5'))
        downsample_ratio = vlm_config.get('downsample_ratio', getattr(config, 'downsample_ratio', 0.5))
        self._num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.downsample_ratio = downsample_ratio
        self.ps_version = vlm_config.get('ps_version', getattr(config, 'ps_version', 'v2'))
        
        # Language configuration - check multiple locations for backward compatibility
        llm_config = vlm_config.get('llm_config')
        if llm_config is None and hasattr(config, 'llm_config'):
            # Backward compatibility: check top-level for old checkpoints
            llm_config = config.llm_config
            
        if llm_config is None:
            raise ValueError("No llm_config found in the configuration")
            
        if isinstance(llm_config, dict):
            self.llm_arch_name = llm_config.get('architectures', [''])[0]
        else:
            self.llm_arch_name = llm_config.architectures[0]
        
        # Flash attention configuration - handle dict or object configs
        use_flash_attn = use_flash_attn if has_flash_attn else False
        if isinstance(vision_config, dict):
            vision_config['use_flash_attn'] = True if use_flash_attn else False
        else:
            vision_config.use_flash_attn = True if use_flash_attn else False
        
        if isinstance(llm_config, dict):
            llm_config['attn_implementation'] = 'flash_attention_2' if use_flash_attn else 'eager'
        else:
            llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        
        logger.info(f'num_image_token: {self._num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'target_dtype: {self.target_dtype}')
        
        # Initialize vision model with proper dtype
        if vision_model is not None:
            self._vision_model = vision_model
        else:
            # Convert dict config to InternVisionConfig if needed
            if isinstance(vision_config, dict):
                vision_config_obj = InternVisionConfig(**vision_config)
            else:
                vision_config_obj = vision_config
            
            # Create vision model with proper dtype
            self._vision_model = InternVisionModel(vision_config_obj)
            self._vision_model = self._vision_model.to(self.target_dtype)
        
        # Initialize language model with proper dtype
        if language_model is not None:
            self._language_model = language_model
        else:
            self._language_model = self._create_language_model(llm_config, use_flash_attn)
        
        # Vision-language projection layer with proper dtype (MUST be created before weight loading)
        if isinstance(vision_config, dict):
            vit_hidden_size = vision_config.get('hidden_size', 1024)
        else:
            vit_hidden_size = vision_config.hidden_size
        
        if isinstance(llm_config, dict):
            llm_hidden_size = llm_config.get('hidden_size', 4096)
        else:
            llm_hidden_size = llm_config.hidden_size
        
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        ).to(self.target_dtype)
            
        # Load pretrained weights (AFTER all components are created)
        base_model_for_extraction = getattr(config, 'base_model_for_extraction', None)
        if base_model_for_extraction:
            # Use the already-loaded base model for weight extraction
            self._extract_weights_from_base_model(base_model_for_extraction)
        else:
            pretrained_model_path = getattr(config, 'pretrained_model_path', None)
            if pretrained_model_path:
                # Fallback to manual weight loading
                self._load_pretrained_weights(pretrained_model_path)
        
        # Context management
        self._img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0
        
        # LoRA configuration
        use_backbone_lora = vlm_config.get('use_backbone_lora', getattr(config, 'use_backbone_lora', 0))
        if use_backbone_lora:
            self.wrap_backbone_lora(r=use_backbone_lora, lora_alpha=2 * use_backbone_lora)
        
        use_llm_lora = vlm_config.get('use_llm_lora', getattr(config, 'use_llm_lora', 0))
        if use_llm_lora:
            self.wrap_llm_lora(r=use_llm_lora, lora_alpha=2 * use_llm_lora)
    
    def _create_language_model(self, llm_config, use_flash_attn=True):
        """Create the appropriate language model based on architecture with proper dtype."""
        # Handle dict config
        if isinstance(llm_config, dict):
            arch = llm_config.get('architectures', [''])[0]
            # Set proper attn_implementation for flash attention
            if use_flash_attn:
                llm_config['attn_implementation'] = 'flash_attention_2'
        else:
            arch = llm_config.architectures[0]
            # Set proper attn_implementation for flash attention
            if use_flash_attn:
                llm_config.attn_implementation = 'flash_attention_2'
        
        # Import config classes
        from transformers import LlamaConfig, Qwen2Config
        
        # When creating models from scratch with Flash Attention enabled, we need to be careful
        # about dtype handling. The safest approach is to create without Flash Attention
        # when the target dtype is not float32.
        safe_use_flash_attn = use_flash_attn and self.target_dtype == torch.float32
        
        if arch == 'LlamaForCausalLM':
            if isinstance(llm_config, dict):
                # Use safe flash attention setting for creation
                llm_config['attn_implementation'] = 'flash_attention_2' if safe_use_flash_attn else 'eager'
                llm_config_obj = LlamaConfig(**llm_config)
            else:
                llm_config.attn_implementation = 'flash_attention_2' if safe_use_flash_attn else 'eager'
                llm_config_obj = llm_config
            
            model = LlamaForCausalLM(llm_config_obj)
            
        elif arch == 'Qwen2ForCausalLM':
            if isinstance(llm_config, dict):
                # Use safe flash attention setting for creation
                llm_config['attn_implementation'] = 'flash_attention_2' if safe_use_flash_attn else 'eager'
                llm_config_obj = Qwen2Config(**llm_config)
            else:
                llm_config.attn_implementation = 'flash_attention_2' if safe_use_flash_attn else 'eager'
                llm_config_obj = llm_config
                
            model = Qwen2ForCausalLM(llm_config_obj)
        
        # Convert to target dtype if needed
        if self.target_dtype != torch.float32:
            model = model.to(self.target_dtype)
        else:
            raise NotImplementedError(f'{arch} is not implemented. BeingVLA currently supports LlamaForCausalLM and Qwen2ForCausalLM.')
            
        return model
    
    def _extract_weights_from_base_model(self, base_model):
        """Extract weights from an already-loaded base model."""
        logger.info('Extracting weights from loaded base model')
        
        try:
            # Extract vision model weights
            if hasattr(base_model, 'vision_model'):
                self._vision_model.load_state_dict(base_model.vision_model.state_dict(), strict=False)
                logger.info('Loaded vision model weights from base model')
            
            # Extract language model weights
            if hasattr(base_model, 'language_model'):
                self._language_model.load_state_dict(base_model.language_model.state_dict(), strict=False)
                logger.info('Loaded language model weights from base model')
                
            # Extract MLP weights
            if hasattr(base_model, 'mlp1'):
                self.mlp1.load_state_dict(base_model.mlp1.state_dict(), strict=False)
                logger.info('Loaded MLP weights from base model')
                
            logger.info('Successfully extracted all weights from base model')
            
        except Exception as e:
            logger.error(f'Failed to extract weights from base model: {e}')
            import traceback
            logger.error(f'Traceback: {traceback.format_exc()}')
            
    def _load_pretrained_weights(self, model_path):
        """Load pretrained weights from InternVL model."""
        logger.info(f'Loading pretrained weights from {model_path}')
        
        try:
            # Try component-wise loading first (following original pattern)
            self._load_component_weights(model_path)
        except Exception as e:
            logger.warning(f'Component-wise loading failed: {e}. Trying unified loading...')
            import traceback
            logger.warning(f'Component loading traceback: {traceback.format_exc()}')
            try:
                # Fallback to unified safetensors loading
                self._load_unified_weights(model_path)
            except Exception as e2:
                logger.error(f'Unified loading also failed: {e2}. Model will have random weights!')
                logger.error(f'Unified loading traceback: {traceback.format_exc()}')
    
    def _load_component_weights(self, model_path):
        """Load weights using component-wise approach (following original pattern)."""
        # Try to load from_pretrained for each component
        from transformers import AutoModel
        
        # Load the base model to check its structure with proper dtype
        try:
            base_model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=self.target_dtype
            )
            
            # Extract components if they exist
            if hasattr(base_model, 'vision_model'):
                vision_weights = base_model.vision_model.state_dict()
                self._vision_model.load_state_dict(vision_weights, strict=False)
                logger.info('Loaded vision model weights')
            
            if hasattr(base_model, 'language_model'):
                language_weights = base_model.language_model.state_dict()
                self._language_model.load_state_dict(language_weights, strict=False)
                logger.info('Loaded language model weights')
                
            if hasattr(base_model, 'mlp1'):
                mlp_weights = base_model.mlp1.state_dict()
                self.mlp1.load_state_dict(mlp_weights, strict=False)
                logger.info('Loaded MLP weights')
                
            del base_model  # Free memory
            logger.info('Successfully loaded pretrained weights using component approach')
            
        except Exception as e:
            raise RuntimeError(f'Component loading failed: {e}')
    
    def _load_unified_weights(self, model_path):
        """Load weights from unified safetensors files with robust prefix handling."""
        model_file = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(model_file):
            # Try single file format
            single_file = os.path.join(model_path, "model.safetensors")
            if os.path.exists(single_file):
                from safetensors.torch import load_file
                state_dict = load_file(single_file, device='cpu')
                self._load_from_unified_state_dict(state_dict)
                return
            else:
                raise FileNotFoundError(f'No safetensors files found in {model_path}')
        
        # Multi-file safetensors format
        import json
        with open(model_file, 'r') as f:
            index = json.load(f)
        
        from safetensors.torch import load_file
        all_state_dict = {}
        
        for file_name in set(index['weight_map'].values()):
            file_path = os.path.join(model_path, file_name)
            state_dict = load_file(file_path, device='cpu')
            all_state_dict.update(state_dict)
        
        self._load_from_unified_state_dict(all_state_dict)
        logger.info('Successfully loaded pretrained weights using unified approach')
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, 
                             missing_keys, unexpected_keys, error_msgs):
        """Load weights from state dict with PyTorch-compatible signature.
        
        This method is called by PyTorch's module loading system with a specific signature.
        We override it to handle our custom loading logic while maintaining compatibility.
        """
        # Handle our internal component mapping (_vision_model, _language_model)
        # The prefix passed here already includes "vlm_adapter." when loading BeingVLA checkpoints
        
        # Create temporary mappings for our private attributes
        key_mappings = []
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                # Extract the key without prefix
                local_key = key[len(prefix):]
                
                # Map private attributes to public ones for loading
                if local_key.startswith('_vision_model.'):
                    new_key = prefix + 'vision_model.' + local_key[14:]
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                        key_mappings.append(new_key)
                elif local_key.startswith('_language_model.'):
                    new_key = prefix + 'language_model.' + local_key[16:] 
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                        key_mappings.append(new_key)
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        
        # Clean up temporary mappings
        for key in key_mappings:
            if key in state_dict:
                del state_dict[key]
    
    def _load_from_unified_state_dict(self, state_dict):
        """Load weights from state dict with robust prefix handling.
        
        This is our custom loading logic that was previously in _load_from_state_dict.
        """
        # Define component prefixes and their target models
        component_mapping = {
            'vision_model.': self._vision_model,
            'language_model.': self._language_model,
            'mlp1.': self.mlp1,
        }
        
        for prefix, target_model in component_mapping.items():
            # Extract weights for this component using robust prefix stripping
            component_state = {}
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    # Remove prefix properly
                    new_key = key[len(prefix):]
                    if new_key in target_model.state_dict():
                        component_state[new_key] = value
            
            if component_state:
                missing_keys, unexpected_keys = target_model.load_state_dict(component_state, strict=False)
                logger.info(f'Loaded {len(component_state)} weights for {prefix[:-1]}')
                if missing_keys:
                    logger.warning(f'Missing keys for {prefix[:-1]}: {len(missing_keys)} keys')
                if unexpected_keys:
                    logger.warning(f'Unexpected keys for {prefix[:-1]}: {len(unexpected_keys)} keys')
    
    @property
    def vision_model(self):
        """Return the vision model component - required by BaseVLMAdapter."""
        return self._vision_model
    
    @property
    def language_model(self):
        """Return the language model component - required by BaseVLMAdapter."""
        return self._language_model
    
    @property
    def num_image_token(self) -> int:
        return self._num_image_token
    
    @property
    def img_context_token_id(self) -> Optional[int]:
        return self._img_context_token_id
    
    @img_context_token_id.setter
    def img_context_token_id(self, value: int):
        self._img_context_token_id = value
    
    def pixel_shuffle(self, x, scale_factor=0.5):
        """Pixel shuffle operation for vision feature processing."""
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def extract_feature(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Extract visual features from pixel values."""
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        
        vit_embeds = vit_embeds[:, 1:, :]
        
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Core VLM forward pass."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        image_flags = image_flags.squeeze(-1) if image_flags.dim() > 1 else image_flags
        
        # Get input embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        
        # Handle 5D pixel_values tensor [batch, num_images, channels, height, width]
        if pixel_values.dim() == 5:
            # Squeeze the second dimension if it's 1
            if pixel_values.shape[1] == 1:
                pixel_values = pixel_values.squeeze(1)
            else:
                # Flatten batch and num_images dimensions
                batch_size, num_images = pixel_values.shape[:2]
                pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        
        # Extract and process visual features
        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        
        # Debug logging
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.debug(f'Dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                logger.debug(f'Total samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')
        
        # Replace image context tokens with visual features
        input_ids = input_ids.reshape(B * N)
        
        # Ensure img_context_token_id is set
        if self.img_context_token_id is None:
            raise ValueError("img_context_token_id must be set on the model before forward pass")
            
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            logger.warning(f'{e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                           f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True
        
        # Handle proprioception embeddings if provided (for robot alignment)
        if 'proprio_embeds' in kwargs and 'prop_context_token_id' in kwargs:
            proprio_embeds = kwargs['proprio_embeds']
            prop_context_token_id = kwargs['prop_context_token_id']
            
            prop_selected = (input_ids == prop_context_token_id)
            if prop_selected.sum() > 0:
                assert prop_selected.sum() == proprio_embeds.shape[0], \
                    f"Mismatch between proprio placeholders ({prop_selected.sum()}) and proprio embeddings ({proprio_embeds.shape[0]})"
                input_embeds[prop_selected] = proprio_embeds.to(input_embeds.device)
        
        input_embeds = input_embeds.reshape(B, N, C)
        
        # Language model forward pass
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Apply loss weighting if provided
            if loss_weight is not None:
                loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
                shift_weights = loss_weight[..., 1:].contiguous()
                shift_weights = shift_weights.view(-1)
                shift_weights = shift_weights.to(shift_logits.device)
                shift_weights_sum = shift_weights.sum()
                if loss_reduction_all_gather:
                    import torch.distributed as dist
                    dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)
            
            # Compute cross-entropy loss
            loss_fct = CrossEntropyLoss(reduction='none')
            
            # Use actual tensor dimension as vocab_size to avoid shape mismatches
            actual_vocab_size = shift_logits.shape[-1]
            shift_logits_flat = shift_logits.view(-1, actual_vocab_size)
            shift_labels_flat = shift_labels.view(-1)
            shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)
            
            loss = loss_fct(shift_logits_flat, shift_labels_flat)
            
            if loss_weight is not None:
                loss = loss * shift_weights
                loss = loss.sum() / shift_weights_sum
            else:
                loss = loss.mean()
            
            if ignore_flag:
                loss = loss * 0.0
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
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
        """Generate text responses given visual and textual inputs."""
        assert self.img_context_token_id is not None
        
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **generate_kwargs,
        )
        
        return outputs
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        # This is typically handled by the language model
        # For InternVL, we delegate to the language model's prepare_inputs_for_generation
        if hasattr(self._language_model, 'prepare_inputs_for_generation'):
            return self._language_model.prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        else:
            # Default implementation
            if past_key_values:
                input_ids = input_ids[:, -1:]
            
            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}
            
            model_inputs.update({
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            })
            return model_inputs
    
    def chat(
        self,
        tokenizer,
        pixel_values,
        question: str,
        generation_config: Dict,
        answer: Optional[str] = None,
        history: Optional[List] = None,
        return_history: bool = False,
        num_patches_list: Optional[List] = None,
        IMG_START_TOKEN: str = '<img>',
        IMG_END_TOKEN: str = '</img>',
        IMG_CONTEXT_TOKEN: str = '<IMG_CONTEXT>',
        verbose: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, List]]:
        """High-level chat interface for interactive use."""
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        
        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.debug(f'Dynamic ViT batch size: {image_bs}')
        
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
        response = response.split(template.sep.strip())[0]
        bos = tokenizer.bos_token
        if bos and response.startswith(bos):
            response = response[len(bos):]
        if response.startswith(template.roles[1]):
            response = response[len(template.roles[1]):]
        response = response.strip()
        history.append((question, response))
        
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                logger.info(f"{query_to_print} {response}")
            return response
    
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        """Apply LoRA to the vision backbone."""
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self._vision_model = get_peft_model(self._vision_model, lora_config)
        self._vision_model.print_trainable_parameters()
    
    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        """Apply LoRA to the language model."""
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplementedError
        
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self._language_model = get_peft_model(self._language_model, lora_config)
        self._language_model.enable_input_require_grads()
        self._language_model.print_trainable_parameters()
    
    # Checkpoint conversion methods for true modularity
    def get_state_dict_key_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from InternVL checkpoint keys to BeingVLA format keys.
        
        Returns:
            Dictionary mapping old InternVL keys to new BeingVLA keys
        """
        return {
            'vision_model.': 'vlm_adapter._vision_model.',
            'language_model.': 'vlm_adapter._language_model.',
            'mlp1.': 'vlm_adapter.mlp1.',
            # Note: motion model mappings are handled by the motion adapter
        }
    
    def detect_checkpoint_format(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """
        Detect if this is an InternVL checkpoint format.
        
        Args:
            state_dict: The state dict to analyze
            
        Returns:
            'internvl' if this is an InternVL checkpoint, 'beingvla' if already converted
        """
        # Check for BeingVLA format first
        if any(k.startswith('vlm_adapter.') for k in state_dict.keys()):
            return 'beingvla'
        
        # Check for direct InternVL keys
        has_vision_model = any(k.startswith('vision_model.') for k in state_dict.keys())
        has_language_model = any(k.startswith('language_model.') for k in state_dict.keys())
        has_mlp1 = any(k.startswith('mlp1.') for k in state_dict.keys())
        
        if has_vision_model and (has_language_model or has_mlp1):
            return 'internvl'
        
        return 'unknown'
    
    def convert_legacy_config(self, legacy_config) -> Dict[str, any]:
        """
        Convert InternVLMotionConfig to BeingVLA config format.
        
        Args:
            legacy_config: InternVLMotionConfig instance
            
        Returns:
            Dictionary with BeingVLA config parameters
        """
        # Extract robot alignment parameters if present
        proprio_dim = getattr(legacy_config, 'proprio_dim', 13)
        action_dim = getattr(legacy_config, 'action_dim', 13)
        action_chunk_length = getattr(legacy_config, 'action_chunk_length', 16)
        gen_action_type = getattr(legacy_config, 'gen_action_type', 'prop_hidden')
        loss_func = getattr(legacy_config, 'loss_func', 'l1')
        
        return {
            'vlm_type': 'internvl',
            'motion_type': 'mano',
            'vlm_config': {
                'vision_config': legacy_config.vision_config,
                'llm_config': legacy_config.llm_config,
                'downsample_ratio': legacy_config.downsample_ratio,
                'ps_version': legacy_config.ps_version,
                'template': legacy_config.template,
                'select_layer': legacy_config.select_layer,
                'force_image_size': legacy_config.force_image_size,
                'use_backbone_lora': legacy_config.use_backbone_lora,
                'use_llm_lora': legacy_config.use_llm_lora,
                'system_message': getattr(legacy_config, 'system_message', None),
                'use_thumbnail': getattr(legacy_config, 'use_thumbnail', False),
                'dynamic_image_size': getattr(legacy_config, 'dynamic_image_size', True),
                'min_dynamic_patch': getattr(legacy_config, 'min_dynamic_patch', 1),
                'max_dynamic_patch': getattr(legacy_config, 'max_dynamic_patch', 12),
            },
            'motion_config': getattr(legacy_config, 'motion_config', None),
            # Robot alignment parameters at top level
            'proprio_dim': proprio_dim,
            'action_dim': action_dim,
            'action_chunk_length': action_chunk_length,
            'gen_action_type': gen_action_type,
            'loss_func': loss_func,
            'enable_robot_alignment': proprio_dim > 0,  # Enable if proprio_dim is set
        }
    
    def preprocess_conversation(
        self,
        sources: List[Dict],
        tokenizer,
        **kwargs
    ) -> Dict[str, any]:
        """
        Preprocess conversations using InternVL-specific templates.
        
        This handles the InternVL conversation format and templates.
        """
        # Import the InternVL-specific preprocessing function
        from ...dataset.dataset_internvl import preprocess_internvl2_5
        
        # Extract relevant kwargs for InternVL preprocessing
        preprocess_kwargs = {
            'template_name': kwargs.get('template_name', self.template),
            'system_message': kwargs.get('system_message', getattr(self.config, 'system_message', '')),
            'num_image_token_list': kwargs.get('num_image_token_list', [self.num_image_token]),
            'max_length': kwargs.get('max_length', 2048),
            'ds_name': kwargs.get('ds_name', None),
            'num_image_token': self.num_image_token,
        }
        
        return preprocess_internvl2_5(
            sources=sources,
            tokenizer=tokenizer,
            **preprocess_kwargs
        )
    
    def get_conv_template(self, template_name=None):
        """Get conversation template for chat interface."""
        if template_name is None:
            template_name = self.template
        return get_conv_template(template_name)


# Register the InternVL adapter with the global registry
# Protected registration to avoid circular import issues
def _register_internvl_adapter():
    """Register InternVL adapter with import protection."""
    try:
        register_vlm_adapter('internvl', InternVLAdapter, InternVLAdapter.convert_legacy_config)
    except Exception as e:
        logger.warning(f"Failed to register InternVL adapter: {e}")

# Safe registration with error handling
if __name__ != '__main__':
    _register_internvl_adapter()