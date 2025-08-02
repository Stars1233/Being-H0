# MANO Motion Adapter Implementation
#
# This module extracts the motion-specific logic from the original InternVLMotionModel
# to create a modular MANO motion adapter that can be used with the VLA framework.

import os
import json
import math
import re
from typing import List, Optional, Union
import numpy as np
import torch
from torch import nn
from transformers.utils import logging

from .m2m.tokenizer.model import ManoVQModel
from .m2m.tokenizer.config import MotionArguments, calculate_num_tokens, calculate_codebook_size
from ...utils.constants import MOT_START_TOKEN, MOT_END_TOKEN, MOT_NOOP

from .base import BaseMotionAdapter, BaseMotionProcessor

logger = logging.get_logger(__name__)


class MotionChunkHelper:
    """
    Helper class for parsing motion chunks from input sequences.
    Hypothesis: the motion token ids are contiguous in the vocabulary.
    """
    
    def __init__(self, mot_start_id: int, mot_end_id: int, mot_noop_id: int = -1):
        if mot_noop_id == -1:
            self.mot_noop_id = mot_end_id - 1
        else:
            self.mot_noop_id = mot_noop_id
        self.mot_start_id = mot_start_id
        self.mot_end_id = mot_end_id

        self.pattern_between = re.compile(re.escape(MOT_START_TOKEN) + r'(.*?)' + re.escape(MOT_END_TOKEN))
        self.token_re = re.compile(r'<motion_id_(\d+)>')
        
    def get_chunks(self, inputs: Union[torch.LongTensor, List[str], str]) -> List[torch.LongTensor]:
        """
        Extract motion chunks from the input tensor or list of strings.
        
        Args:
            inputs: The input tensor or list of strings.
            
        Returns:
            A list of motion chunks, each chunk is a tensor.
        """
        if isinstance(inputs, str): 
            inputs = [inputs]
        
        if isinstance(inputs, List):
            mot_chunks = []
            for input_str in inputs:
                try:
                    assert isinstance(input_str, str), f'Input should be a string or a list of strings, but got {type(input_str)}'
                    mot_chunk = []
                    for match in self.pattern_between.findall(input_str):
                        mot_block = []
                        for token in self.token_re.findall(match):
                            mot_block.append(int(token))
                        if mot_chunk:
                            assert len(mot_block) == len(mot_chunk[-1])
                        mot_chunk.append(torch.as_tensor(mot_block, dtype=torch.int64))
                    mot_chunk = torch.stack(mot_chunk, dim=0)
                    mot_chunks.append(mot_chunk)
                except Exception as e:
                    logger.error(f'Error in parsing motion chunks from input: {input_str}. Error: {e}')
                    mot_chunks.append(torch.as_tensor([], dtype=torch.int64))
                    
        elif isinstance(inputs, torch.Tensor):
            mot_start = (inputs == self.mot_start_id)  # (batch_size, seq_len)
            mot_end = (inputs == self.mot_end_id)  # (batch_size, seq_len)
            mot_inside = (mot_start.cumsum(dim=-1) > mot_end.cumsum(dim=-1)) & (~ mot_start)  # (batch_size, seq_len)
            mot_token_mask = (inputs > self.mot_start_id) & (inputs < self.mot_end_id) & inputs.ne(self.mot_noop_id)  
            mot_token_mask = mot_token_mask & mot_inside  # (batch_size, seq_len)
            
            mot_chunks = []
            for batch_idx in range(inputs.shape[0]):
                try:
                    mot_tokens = inputs[batch_idx][mot_token_mask[batch_idx]]  # (num_mot_tokens,)
                    mot_start_num = mot_start[batch_idx].sum().item()
                    mot_end_num = mot_end[batch_idx].sum().item()
                    assert mot_start_num == mot_end_num, f'Motion start and end tokens mismatch in batch {batch_idx}: {mot_start_num} vs {mot_end_num}'
                    assert mot_start_num > 0, f'No motion start tokens found in batch {batch_idx}'
                    mot_chunk = mot_tokens.reshape(mot_start_num, -1)  
                    mot_chunks.append(mot_chunk)
                except Exception as e:
                    logger.error(f'Error in parsing motion chunks from batch {batch_idx}. Error: {e}')
                    mot_chunks.append(torch.tensor([]))
        else:
            raise TypeError(f'Unsupported input type: {type(inputs)}. Expected torch.Tensor or List[str].')
        return mot_chunks


class MotionProcessor(BaseMotionProcessor):
    """Processor for motion generation with various constraint modes."""
    
    def __init__(
        self, 
        mot_start_id: int, 
        mot_end_id: int, 
        eos_id: int,
        allow_non_motion: bool = False,
        fixed_block_length: Optional[int] = None
    ):
        super().__init__(mot_start_id, mot_end_id, eos_id)
        self.allow_non_motion = allow_non_motion
        self.fixed_block_length = fixed_block_length
        self.noop = self.allow_non_motion and (self.fixed_block_length is None)
        
        self.max_block_num = None
        self.min_block_num = None  # Add minimum block number control
        self.prefix_base = 0
        
    def set_block_num_control(
        self, 
        max_block_num: Union[int, List[int], torch.Tensor] = None,
        min_block_num: Union[int, List[int], torch.Tensor] = None,
        prefix_base: Union[int, List[int], torch.Tensor] = 0
    ):
        """Set block number control parameters."""
        self.max_block_num = max_block_num
        self.min_block_num = min_block_num
        self.prefix_base = prefix_base
        self.noop = False
    
    def reset_block_num_control(self):
        """Reset block number control to default state."""
        self.max_block_num = None
        self.min_block_num = None
        self.prefix_base = 0
        self.noop = self.allow_non_motion and (self.fixed_block_length is None)
            
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to enforce motion generation constraints."""
        if not self.noop:
            input_start = input_ids == self.mot_start_id # (batch_size, seq_len)
            input_start_sum = input_start.sum(dim=-1)
            input_end = input_ids == self.mot_end_id 
            input_end_sum = input_end.sum(dim=-1)
        
            inside_motion = (input_start_sum > input_end_sum) # (batch_size,)
        
        if not self.allow_non_motion:
            scores[inside_motion, :self.mot_start_id] = -float('inf')  
            scores[inside_motion, self.mot_end_id + 1:] = -float('inf')
        
        if self.fixed_block_length is not None:
            # find the last start token in each sequence
            if input_start.shape[-1] == 0:
                last_mot_length = torch.zeros(input_ids.shape[0], dtype=torch.int64, device=input_ids.device)
            else:
                last_start_idx = input_start.cumsum(dim=-1).argmax(dim=-1)  # (batch_size,)
                last_mot_length = input_ids.shape[-1] - 1 - last_start_idx
                last_mot_length[~inside_motion] = 0  # if not inside motion, set length to 0
            reach_max_length = last_mot_length >= self.fixed_block_length
            scores[reach_max_length, :self.mot_end_id] = -float('inf') 
            scores[reach_max_length, self.mot_end_id + 1:] = -float('inf')
        
        if self.max_block_num is not None:
            # count the number of segments
            chunk_num = input_end_sum - self.prefix_base
            max_reached = chunk_num >= self.max_block_num  # (batch_size,)
            if max_reached.any():
               scores[max_reached, :self.eos_id] = -float('inf')  
               scores[max_reached, self.eos_id + 1:] = -float('inf') 
        
        if self.min_block_num is not None:
            # Prevent EOS token until minimum chunks are generated
            chunk_num = input_end_sum - self.prefix_base
            min_not_reached = chunk_num < self.min_block_num  # (batch_size,)
            if min_not_reached.any():
                # Force the model to continue generating motion tokens
                scores[min_not_reached, self.eos_id] = -float('inf')
                # If not inside motion, force starting a new motion block
                not_inside_and_min_not_reached = min_not_reached & (~inside_motion)
                if not_inside_and_min_not_reached.any():
                    # Only allow START token
                    scores[not_inside_and_min_not_reached, :self.mot_start_id] = -float('inf')
                    scores[not_inside_and_min_not_reached, self.mot_start_id + 1:] = -float('inf')

        return scores


class ManoMotionAdapter(BaseMotionAdapter):
    """
    MANO motion adapter for the VLA framework.
    
    This adapter handles MANO motion model loading, encoding/decoding,
    and motion processing logic extracted from the original InternVLMotionModel.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        if hasattr(config, 'motion_config') and len(config.motion_config) > 0:
            if not isinstance(config.motion_config, list):
                config.motion_config = [config.motion_config]
            self._motion_config = [MotionArguments(**motion_config) for motion_config in config.motion_config]
            self._build_motion_model()
        else:
            logger.warning('No motion_config provided. Motion model will not be initialized.')
            self._motion_config = None
            self.motion_model = None
        
        # Motion processing helpers
        self.motion_chunk_helper = None
        self.motion_processor = None
        
        # Block configuration
        self._motion_block_shape = None
        self._motion_block_length = None
        self._motion_block_length_sep = None
        self._codebook_size_list = None
        self._codebook_base_list = None
        self._res_len_list = None
        
        if self._motion_config:
            self._setup_block_configuration()
    
    def _build_motion_model(self):
        """Build motion model from configuration."""
        motion_model_list = []
        motion_mean_list = []
        motion_std_list = []
        
        for motion_args in self._motion_config:
            motion_model = ManoVQModel(motion_args)
            checkpoint_dir_candidates = [motion_args.motion_code_path, motion_args.motion_resume_pth]
            checkpoint_name_candidates = ['overfit.pth', 'net_best_mpjpe.pth', 'net_best_mpjpe_labeling.pth']
            checkpoint_path = None
            
            for checkpoint_dir in checkpoint_dir_candidates:
                if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
                    continue
                for checkpoint_name in checkpoint_name_candidates:
                    candidate_path = os.path.join(checkpoint_dir, checkpoint_name)
                    if os.path.exists(candidate_path):
                        checkpoint_path = candidate_path
                        break
                if checkpoint_path:
                    break
                    
            if checkpoint_path is None:
                logger.warning(f"Pretrained motion model not found in {checkpoint_dir_candidates} with names {checkpoint_name_candidates}.")
                self.motion_model = None
                return
            else:
                logger.info('Loading Motion model checkpoint from {}'.format(checkpoint_path))
                motion_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'], strict=True)
                motion_model.eval()
                motion_model_list.append(motion_model)
            
            # Load metadata
            meta_mask_path = os.path.join(os.path.dirname(checkpoint_path), 'meta_mask.json')
            assert os.path.exists(meta_mask_path), f'Model meta mask file {meta_mask_path} does not exist.'
            with open(meta_mask_path, 'r') as f:
                model_info = json.load(f)
                motion_meta_mask = model_info.get('meta_mask', None)

            motion_meta_path = os.path.join(os.path.dirname(checkpoint_path), 'motion_meta.json')
            assert os.path.exists(motion_meta_path), f'Motion metadata file {motion_meta_path} does not exist.'
            with open(motion_meta_path, 'r') as f:
                logger.info(f'Loading motion metadata from {motion_meta_path}')
                metadata = json.load(f)
                motion_mean = torch.as_tensor(metadata['mean'], dtype=torch.float32)
                motion_std = torch.as_tensor(metadata['std'], dtype=torch.float32)
                if motion_meta_mask is not None:
                    motion_meta_mask = torch.as_tensor(motion_meta_mask, dtype=torch.bool)
                    motion_mean = motion_mean[motion_meta_mask]
                    motion_std = motion_std[motion_meta_mask]
                motion_mean_list.append(nn.Parameter(motion_mean, requires_grad=False))
                motion_std_list.append(nn.Parameter(motion_std, requires_grad=False))
        
        if motion_model_list:
            assert len(motion_model_list) == len(motion_mean_list) == len(motion_std_list), \
                f'Motion Model and metadata Length mismatch: {len(motion_model_list)}, {len(motion_mean_list)}, {len(motion_std_list)}'
            
            self.motion_model = nn.ModuleList(motion_model_list)
            self.motion_mean = nn.ParameterList(motion_mean_list)
            self.motion_std = nn.ParameterList(motion_std_list)
    
    def _setup_block_configuration(self):
        """Setup motion block configuration."""
        self._motion_block_shape, self._res_len_list = calculate_num_tokens(self._motion_config)
        motion_block_length = [math.prod(shape) for shape in self._motion_block_shape]
        self._motion_block_length_sep = [0] + list(np.cumsum(motion_block_length, dtype=int))
        self._motion_block_length = self._motion_block_length_sep[-1]
        
        self._codebook_size_list = calculate_codebook_size(self._motion_config)
        self._codebook_base_list = [0] + list(np.cumsum(self._codebook_size_list[:-1]))
    
    @property
    def motion_config(self) -> List:
        return self._motion_config
    
    @property
    def motion_block_shape(self) -> List[List[int]]:
        return self._motion_block_shape
    
    @property
    def motion_block_length(self) -> int:
        return self._motion_block_length
    
    @property
    def codebook_size_list(self) -> List[int]:
        return self._codebook_size_list
    
    def setup_motion_processor(
        self, 
        tokenizer, 
        allow_non_motion: bool = False, 
        fixed_block_length_control: bool = False,
        padded_block: bool = True, 
        max_block_num_control: bool = False,
        min_block_num_control: bool = False,
        gt_constraint: str = 'none', 
        alpha: float = 0.5
    ):
        """
        Setup motion processor for controlling generation of motion blocks.
        
        Args:
            tokenizer: The tokenizer for encoding/decoding
            allow_non_motion: If True, allows non-motion tokens
            fixed_block_length_control: If True, enables fixed length control
            padded_block: Whether to use padded block length
            max_block_num_control: If True, enables max block number control
            min_block_num_control: If True, enables min block number control
            gt_constraint: Ground truth constraint ('none', 'soft', 'hard')
            alpha: Alpha parameter for soft constraint
        """
        if fixed_block_length_control:
            block_length = self.motion_block_length if not padded_block else 2 * self.motion_block_length
        else:
            block_length = None
        
        mot_start_id = tokenizer.convert_tokens_to_ids(MOT_START_TOKEN)
        mot_end_id = tokenizer.convert_tokens_to_ids(MOT_END_TOKEN)
        mot_noop_id = tokenizer.convert_tokens_to_ids(MOT_NOOP)
        eos_id = tokenizer.eos_token_id
        
        self.motion_processor = MotionProcessor(
            mot_start_id=mot_start_id,
            mot_end_id=mot_end_id,
            eos_id=eos_id,
            allow_non_motion=allow_non_motion,
            fixed_block_length=block_length
        )
        
        self.motion_chunk_helper = MotionChunkHelper(
            mot_start_id=mot_start_id,
            mot_end_id=mot_end_id,
            mot_noop_id=mot_noop_id,
        )
        
        logger.info(f'Motion processor setup: start={mot_start_id}, end={mot_end_id}, noop={mot_noop_id}, eos={eos_id}')
    
    @torch.no_grad()
    def decode_motion(
        self, 
        motion_block_ids: torch.LongTensor, 
        offset: bool = False,
        denormalize: bool = True,
        return_list: bool = False
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        """
        Convert motion block IDs to MANO parameters.
        
        Args:
            motion_block_ids: Encoded motion block IDs
            offset: Whether to apply offset correction
            denormalize: Whether to denormalize the output
            return_list: Whether to return list of parts or concatenated tensor
            
        Returns:
            Decoded MANO parameters
        """
        if self.motion_model is None:
            raise RuntimeError("Motion model not initialized")
            
        # Apply offset if needed
        if offset:
            motion_block_ids = motion_block_ids - self.motion_processor.mot_start_id - 1
        mano_list = []
        prefix_shape = motion_block_ids.shape[:-1]
        
        for i, sub_block_shape in enumerate(self.motion_block_shape):
            res_len = self._res_len_list[i]
            codebook_offset = self._codebook_base_list[i]
            start_idx = self._motion_block_length_sep[i]
            end_idx = self._motion_block_length_sep[i + 1]
            
            motion_args = self.motion_config[i]
            
            # Extract tokens for this motion part - EXACTLY like original
            motion_sub_block_ids = motion_block_ids[..., start_idx:end_idx] - codebook_offset
            motion_sub_block_ids = motion_sub_block_ids.reshape(-1, *sub_block_shape)
            motion_model = self.motion_model[i]
            
            if 'residualvq' in motion_args.quantizer_name and not motion_args.shared_codebook:
                motion_sub_block_ids %= motion_args.nb_code
            
            if 'group' in motion_args.quantizer_name:
                motion_sub_block_ids = motion_sub_block_ids.transpose(0,1)
                
            decoder_output = motion_model.forward_decoder(motion_sub_block_ids)
            mano = decoder_output[..., :-res_len, :] if res_len > 0 else decoder_output
            
            if denormalize:
                mano_mean = self.motion_mean[i].to(mano.dtype)
                mano_std = self.motion_std[i].to(mano.dtype)
                mano = mano * mano_std + mano_mean
            
            logger.debug(f'Motion Part {i}: mano.shape={mano.shape}')
            mano_list.append(mano.reshape(*prefix_shape, *mano.shape[1:]))
        
        if not return_list:
            return torch.cat(mano_list, dim=-1)
        else:
            return mano_list
    
    @torch.no_grad()
    def encode_motion(
        self, 
        mano_list: List[torch.FloatTensor], 
        normalize: bool = True
    ) -> torch.LongTensor:
        """
        Convert MANO parameters to motion block IDs.
        
        Args:
            mano_list: List of MANO parameter tensors for each motion part
            normalize: Whether to normalize the MANO parameters
            
        Returns:
            Encoded motion block IDs
        """
        if self.motion_model is None:
            raise RuntimeError("Motion model not initialized")
            
        motion_block_ids_list = []
        
        for i, part_mano in enumerate(mano_list):
            if normalize:
                mano_mean = self.motion_mean[i].to(part_mano.dtype)
                mano_std = self.motion_std[i].to(part_mano.dtype)
                part_mano = (part_mano - mano_mean) / (mano_std + 1e-6)
                
            res_len = self._res_len_list[i]
            codebook_base = self._codebook_base_list[i]
            motion_model = self.motion_model[i]
            motion_args = self.motion_config[i]
            
            part_mano = torch.cat([
                part_mano, 
                torch.zeros(*part_mano.shape[:-2], res_len, part_mano.shape[-1], device=part_mano.device)
            ], dim=-2)
            motion_block_ids = motion_model.encode(part_mano)
            
            if 'residualvq' in motion_args.quantizer_name and not motion_args.shared_codebook:
                for j in range(motion_block_ids.shape[-1]):
                    motion_block_ids[..., j] = motion_block_ids[..., j] + motion_args.nb_code * j
            
            if 'group' in motion_args.quantizer_name:
                motion_block_ids = motion_block_ids.transpose(0, 1)
                
            motion_block_ids = motion_block_ids.flatten(start_dim=1) + codebook_base
            motion_block_ids_list.append(motion_block_ids)
            
        return torch.cat(motion_block_ids_list, dim=-1) + self.motion_processor.mot_start_id + 1
    
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
        mot_start = (input_ids == mot_start_id)
        mot_end = (input_ids == mot_end_id)
        mot_inside = (mot_start.cumsum(dim=-1) > mot_end.cumsum(dim=-1)) & (~mot_start)
        mot_token_mask = (input_ids > mot_start_id) & (input_ids < mot_end_id) & input_ids.ne(mot_noop_id)
        return mot_token_mask & mot_inside