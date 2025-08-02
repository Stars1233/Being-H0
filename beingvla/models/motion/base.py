# Base adapter interfaces for VLA modular architecture

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import GenerationConfig, LogitsProcessor, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast


class BaseVLMAdapter(ABC, nn.Module):
    """
    Abstract base class for Vision-Language Model adapters.
    
    This interface allows different VLMs (InternVL, LLaVA, QwenVL, etc.) to be
    plugged into the VLA framework while maintaining consistent APIs.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @property
    @abstractmethod
    def vision_model(self):
        """Return the vision model component."""
        pass
    
    @property
    @abstractmethod
    def language_model(self):
        """Return the language model component."""
        pass
    
    @property
    @abstractmethod
    def num_image_token(self) -> int:
        """Return the number of image tokens."""
        pass
    
    @property
    @abstractmethod
    def img_context_token_id(self) -> Optional[int]:
        """Return the image context token ID."""
        pass
    
    @img_context_token_id.setter
    @abstractmethod
    def img_context_token_id(self, value: int):
        """Set the image context token ID."""
        pass
    
    @abstractmethod
    def extract_feature(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Extract visual features from pixel values.
        
        Args:
            pixel_values: Input images tensor
            
        Returns:
            Visual feature embeddings
        """
        pass
    
    @abstractmethod
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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for VLM training.
        
        This should handle the core VLM forward pass including:
        - Vision feature extraction
        - Input embedding preparation
        - Language model forward pass
        """
        pass
    
    @abstractmethod
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
        """
        Generate text responses given visual and textual inputs.
        """
        pass
    
    @abstractmethod
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
        **kwargs
    ) -> Union[str, Tuple[str, List]]:
        """
        High-level chat interface for interactive use.
        """
        pass
    
    def get_input_embeddings(self):
        """Get input embeddings from the language model."""
        return self.language_model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """Get output embeddings from the language model."""
        return self.language_model.get_output_embeddings()


class BaseMotionAdapter(ABC, nn.Module):
    """
    Abstract base class for Motion Model adapters.
    
    This interface allows different motion models (MANO, SMPL, etc.) to be
    plugged into the VLA framework while maintaining consistent APIs.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @property
    @abstractmethod
    def motion_config(self) -> List:
        """Return motion configuration parameters."""
        pass
    
    @property
    @abstractmethod
    def motion_block_shape(self) -> List[List[int]]:
        """Return the shape of motion blocks for each motion part."""
        pass
    
    @property
    @abstractmethod
    def motion_block_length(self) -> int:
        """Return the total length of a motion block."""
        pass
    
    @property
    @abstractmethod
    def codebook_size_list(self) -> List[int]:
        """Return the codebook sizes for each motion part."""
        pass
    
    @abstractmethod
    def encode_motion(
        self, 
        mano_list: List[torch.FloatTensor], 
        normalize: bool = True
    ) -> torch.LongTensor:
        """
        Encode MANO parameters to motion block IDs.
        
        Args:
            mano_list: List of MANO parameter tensors for each motion part
            normalize: Whether to normalize the MANO parameters
            
        Returns:
            Encoded motion block IDs
        """
        pass
    
    @abstractmethod
    def decode_motion(
        self, 
        motion_block_ids: torch.LongTensor, 
        offset: bool = False,
        denormalize: bool = True,
        return_list: bool = False
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Decode motion block IDs to MANO parameters.
        
        Args:
            motion_block_ids: Encoded motion block IDs
            offset: Whether to apply offset correction
            denormalize: Whether to denormalize the output
            return_list: Whether to return list of parts or concatenated tensor
            
        Returns:
            Decoded MANO parameters
        """
        pass
    
    @abstractmethod
    def get_motion_tokens_mask(
        self,
        input_ids: torch.LongTensor,
        mot_start_id: int,
        mot_end_id: int,
        mot_noop_id: int
    ) -> torch.BoolTensor:
        """
        Get mask for motion tokens in the input sequence.
        
        Args:
            input_ids: Input token IDs
            mot_start_id: Motion start token ID
            mot_end_id: Motion end token ID  
            mot_noop_id: Motion no-op token ID
            
        Returns:
            Boolean mask for motion tokens
        """
        pass


class BaseMotionProcessor(LogitsProcessor):
    """
    Abstract base class for motion generation processors.
    
    These processors control motion generation by modifying logits during
    the generation process to enforce motion-specific constraints.
    """
    
    def __init__(
        self,
        mot_start_id: int,
        mot_end_id: int,
        eos_id: int,
        **kwargs
    ):
        self.mot_start_id = mot_start_id
        self.mot_end_id = mot_end_id
        self.eos_id = eos_id
    
    @abstractmethod
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to enforce motion generation constraints.
        
        Args:
            input_ids: Current sequence of token IDs
            scores: Current token logits
            
        Returns:
            Modified logits with motion constraints applied
        """
        pass