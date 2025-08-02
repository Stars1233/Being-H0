# BeingVLA Generation Mixins
#
# This module contains generation-specific logic extracted from BeingVLAModel to achieve
# better separation of concerns and modular design.

from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import GenerationConfig, StoppingCriteriaList
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BeingVLAGenerationMixin:
    """
    Generation-specific functionality for BeingVLA models.
    
    This mixin contains all generation-related logic including motion-aware generation,
    GT constraints, soft motion blending, and high-level chat interfaces.
    """
    
    def __init_generation_attributes__(self):
        """Initialize generation-related attributes."""
        # Motion generation attributes
        self.max_block_num_control = False
        self.motion_gt_constraint = 'none'
        self.alpha = 0.5
    
    def build_motion_processor(self, tokenizer, allow_non_motion: bool = False, 
                              fixed_block_length_control: bool = False, padded_block: bool = True, 
                              max_block_num_control: bool = False, min_block_num_control: bool = False,
                              gt_constraint: str = 'none', alpha: float = 0.5):
        """
        Build motion processor for motion-aware generation.
        
        Args:
            tokenizer: Tokenizer for motion tokens
            allow_non_motion: Whether to allow non-motion sequences
            fixed_block_length_control: Whether to use fixed block length control
            padded_block: Whether to use padded blocks
            max_block_num_control: Whether to control maximum block number
            min_block_num_control: Whether to control minimum block number
            gt_constraint: Ground truth constraint type ('none', 'soft', 'hard')
            alpha: Blending factor for soft constraints
        """
        if self.motion_adapter is None:
            logger.warning("Motion adapter not available. Cannot build motion processor.")
            return
        
        self.motion_adapter.setup_motion_processor(
            tokenizer=tokenizer,
            allow_non_motion=allow_non_motion,
            fixed_block_length_control=fixed_block_length_control,
            padded_block=padded_block,
            max_block_num_control=max_block_num_control,
            min_block_num_control=min_block_num_control,
            gt_constraint=gt_constraint,
            alpha=alpha
        )
        
        # Store generation parameters
        self.max_block_num_control = max_block_num_control
        self.motion_gt_constraint = gt_constraint
        self.alpha = alpha
        
        logger.info(f'Motion processor built: max_block_control={max_block_num_control}, '
                   f'gt_constraint={gt_constraint}, alpha={alpha}')
    
    @torch.no_grad()
    def generate_with_motion(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        answer_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Motion-aware generation with constraints.
        
        This method performs generation with motion awareness, supporting various
        constraint modes and motion guidance strategies.
        
        Args:
            pixel_values: Input images
            input_ids: Input token IDs
            answer_ids: Ground truth answer IDs for constraints
            attention_mask: Attention mask
            visual_features: Pre-extracted visual features
            generation_config: Generation configuration
            output_hidden_states: Whether to output hidden states
            stopping_criteria: Stopping criteria for generation
            **generate_kwargs: Additional generation arguments
            
        Returns:
            Generated token sequences
        """
        # Convert generation_config dict to GenerationConfig object if needed
        if isinstance(generation_config, dict):
            from transformers import GenerationConfig
            generation_config = GenerationConfig(**generation_config)
            
        if self.motion_adapter is None or self.motion_adapter.motion_processor is None:
            logger.warning("Motion adapter or motion processor not available. Falling back to standard generation.")
            return self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                stopping_criteria=stopping_criteria,
                **generate_kwargs,
            )
        
        # Extract visual features if needed
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
        
        # Setup motion constraints if needed
        if self.max_block_num_control or self.motion_gt_constraint != 'none':
            # Check if motion processor already has block number control set
            if (hasattr(self.motion_adapter.motion_processor, 'max_block_num') and 
                self.motion_adapter.motion_processor.max_block_num is not None):
                # Block control is already set manually, skip automatic setup
                pass
            else:
                # Use answer_ids to set block control if available
                assert answer_ids is not None, "answer_ids must be provided when using motion constraints without manual block control."
                answer_chunks = [chunk.to(input_ids.device) for chunk in 
                               self.motion_adapter.motion_chunk_helper.get_chunks(answer_ids)]
                answer_chunk_num = torch.as_tensor([max(1, len(chunk)) for chunk in answer_chunks], 
                                                 device=input_ids.device)
                prefix_base = (input_ids == self.motion_adapter.motion_processor.mot_end_id).reshape(B, N).sum(dim=-1)
        
        # Generate with motion constraints
        if self.motion_gt_constraint == 'none':
            if self.max_block_num_control:
                # Only set block control if not already manually set
                if not (hasattr(self.motion_adapter.motion_processor, 'max_block_num') and 
                        self.motion_adapter.motion_processor.max_block_num is not None):
                    self.motion_adapter.motion_processor.set_block_num_control(answer_chunk_num, prefix_base=prefix_base)
            
            outputs = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                logits_processor=[self.motion_adapter.motion_processor],
                **generate_kwargs,
            )
        else:
            # Complex motion-guided generation with GT constraints
            assert answer_ids is not None, "answer_ids must be provided when using GT constraints."
            outputs = self._generate_with_gt_constraints(
                input_embeds, attention_mask, answer_chunks, answer_chunk_num, 
                prefix_base, generation_config, output_hidden_states, 
                stopping_criteria, **generate_kwargs
            )
        
        return outputs
    
    def _generate_with_gt_constraints(self, input_embeds, attention_mask, answer_chunks, 
                                    answer_chunk_num, prefix_base, generation_config, 
                                    output_hidden_states, stopping_criteria, **generate_kwargs):
        """
        Generate with ground truth motion constraints (soft/hard).
        
        This method implements multi-turn generation with motion constraints,
        supporting both hard replacement and soft blending of motion tokens.
        """
        batch_size = input_embeds.shape[0]
        generate_turn = answer_chunk_num.max().item()
        generate_ids = [[] for _ in range(batch_size)]
        
        valid_beam_mask = torch.ones(batch_size, dtype=torch.bool, device=input_embeds.device)
        
        for turn_idx in range(generate_turn):
            logger.info(f'Generating turn {turn_idx + 1}/{generate_turn}...')
            self.motion_adapter.motion_processor.set_block_num_control(1, 0)
            
            block_output = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                logits_processor=[self.motion_adapter.motion_processor],
                **generate_kwargs,
            )
            
            if isinstance(block_output, torch.Tensor):
                new_gen_ids_list = block_output
            else:
                new_gen_ids_list = block_output.sequences
                
            valid_idx_list = torch.where(valid_beam_mask)[0]
            pred_id_list = []
            pred_mot_mask_list = []
            pred_mot_id_list = []
            gt_mot_id_list = []
            valid_entry_mask = torch.ones(input_embeds.shape[0], dtype=torch.bool, device=input_embeds.device)
            
            for i, valid_idx in enumerate(valid_idx_list):
                new_gen_ids = new_gen_ids_list[i]
                pad_id = self.motion_adapter.motion_processor.pad_id if hasattr(self.motion_adapter.motion_processor, 'pad_id') else 0
                eos_id = self.motion_adapter.motion_processor.eos_id
                
                non_special_mask = new_gen_ids.ne(pad_id) & new_gen_ids.ne(eos_id)
                new_gen_ids = new_gen_ids[non_special_mask]
                generate_ids[valid_idx].append(new_gen_ids)
                
                if len(generate_ids[valid_idx]) >= answer_chunk_num[valid_idx]:
                    valid_entry_mask[i] = False
                    valid_beam_mask[valid_idx] = False
                    continue
                    
                try:
                    gen_mot_mask = self.motion_adapter.get_motion_tokens_mask(
                        new_gen_ids.unsqueeze(0),
                        self.motion_adapter.motion_processor.mot_start_id,
                        self.motion_adapter.motion_processor.mot_end_id,
                        self.motion_adapter.motion_processor.mot_noop_id if hasattr(self.motion_adapter.motion_processor, 'mot_noop_id') else -1
                    ).squeeze(0)
                    gen_mot_ids = new_gen_ids[gen_mot_mask]
                    gt_ids = answer_chunks[valid_idx][turn_idx]
                    
                    assert len(gen_mot_ids) == len(gt_ids), f'Motion token mismatch: {len(gen_mot_ids)} vs {len(gt_ids)}'
                    
                    if self.motion_gt_constraint == 'hard':
                        new_append_ids = new_gen_ids.clone()
                        new_append_ids[gen_mot_mask] = gt_ids
                        pred_id_list.append(new_append_ids)
                    else:  # soft constraint
                        pred_mot_mask_list.append(gen_mot_mask)
                        pred_mot_id_list.append(gen_mot_ids)
                        gt_mot_id_list.append(gt_ids)
                        pred_id_list.append(new_gen_ids)
                        
                except Exception as e:
                    logger.warning(f"Motion constraint failed: {e}")
                    valid_entry_mask[i] = False
                    valid_beam_mask[valid_idx] = False
            
            if not valid_entry_mask.any():
                break
            
            # Apply soft constraints if needed
            if self.motion_gt_constraint == 'soft' and pred_mot_id_list:
                pred_id_list = self._apply_soft_motion_constraints(
                    pred_id_list, pred_mot_mask_list, pred_mot_id_list, gt_mot_id_list, valid_entry_mask
                )
            
            # Prepare embeddings for next turn
            if valid_entry_mask.sum() > 0:
                input_embeds, attention_mask = self._prepare_next_turn_embeddings(
                    pred_id_list, input_embeds, attention_mask, valid_entry_mask
                )
        
        # Concatenate all generated sequences
        final_outputs = []
        for gen_ids in generate_ids:
            if gen_ids:
                final_outputs.append(torch.cat(gen_ids, dim=-1))
            else:
                final_outputs.append(torch.tensor([], dtype=torch.long, device=input_embeds.device))
        
        return final_outputs
    
    def _apply_soft_motion_constraints(self, pred_id_list, pred_mot_mask_list, 
                                     pred_mot_id_list, gt_mot_id_list, valid_entry_mask):
        """
        Apply soft motion constraints using motion model.
        
        This method blends predicted and ground truth motion parameters
        using the alpha blending factor.
        """
        if not pred_mot_id_list or self.motion_adapter is None:
            return pred_id_list
            
        try:
            valid_mot_ids = torch.stack(pred_mot_id_list, dim=0)
            valid_gt_ids = torch.stack(gt_mot_id_list, dim=0)
            valid_length = valid_mot_ids.shape[0]
            mano_inputs = torch.cat([valid_mot_ids, valid_gt_ids], dim=0)
            
            # Decode to MANO parameters
            mano_list = self.motion_adapter.decode_motion(
                mano_inputs, denormalize=False, return_list=True, offset=True
            )
            
            # Apply soft blending
            soft_mano_list = []
            for mano in mano_list:
                pred_mano = mano[:valid_length]
                gt_mano = mano[valid_length:]
                soft_mano = self.alpha * pred_mano + (1 - self.alpha) * gt_mano
                soft_mano_list.append(soft_mano)
            
            # Encode back to motion tokens
            replace_ids = self.motion_adapter.encode_motion(soft_mano_list, normalize=False).flatten()
            replace_ptr = 0
            
            # Replace motion tokens in predictions
            for idx in range(valid_entry_mask.sum()):
                new_append_ids = pred_id_list[idx]
                gen_mot_mask = pred_mot_mask_list[idx]
                new_append_ids[gen_mot_mask] = replace_ids[replace_ptr:replace_ptr + gen_mot_mask.sum()]
                pred_id_list[idx] = new_append_ids
                replace_ptr += gen_mot_mask.sum()
                
        except Exception as e:
            logger.warning(f"Soft constraint application failed: {e}")
            
        return pred_id_list
    
    def _prepare_next_turn_embeddings(self, pred_id_list, input_embeds, attention_mask, valid_entry_mask):
        """
        Prepare embeddings for the next generation turn.
        
        This method concatenates current and predicted embeddings for multi-turn generation.
        """
        max_length = max(pred_ids.shape[-1] for pred_ids in pred_id_list)
        pad_id = 0  # Default padding token
        
        append_pred_ids = torch.ones((valid_entry_mask.sum(), max_length), 
                                   dtype=torch.long, device=input_embeds.device) * pad_id
        append_attention_mask = torch.ones_like(append_pred_ids, dtype=torch.long)
        
        for i, pred_ids in enumerate(pred_id_list):
            append_pred_ids[i, -pred_ids.shape[-1]:] = pred_ids
            append_attention_mask[i, :-pred_ids.shape[-1]] = 0
        
        append_embeds = self.language_model.get_input_embeddings()(append_pred_ids)
        input_embeds = torch.cat([input_embeds[valid_entry_mask], append_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask[valid_entry_mask], append_attention_mask], dim=-1)
        
        return input_embeds, attention_mask
    
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
        with_motion: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, List]]:
        """
        High-level chat interface.
        
        This method provides a convenient chat interface that can optionally
        use motion-aware generation.
        """
        if with_motion and self.motion_adapter is not None:
            # Motion-aware chat - would need to implement motion-specific logic
            # For now, fall back to standard chat
            logger.warning("Motion-aware chat not fully implemented. Using standard chat.")
        
        return self.vlm_adapter.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            answer=answer,
            history=history,
            return_history=return_history,
            num_patches_list=num_patches_list,
            **kwargs
        )
    
    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, 
                   num_patches_list=None, answers=None, history=None, return_history=False, 
                   IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', with_motion=False,
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None,
                   return_mano=False, **kwargs):
        """
        Batch chat interface for VLA models with motion support.
        
        Args:
            tokenizer: Tokenizer for text processing
            pixel_values: Batch of images (B, C, H, W)
            questions: List of questions
            generation_config: Generation configuration
            num_patches_list: Number of patches for each image
            answers: Ground truth answers (for motion constraints)
            history: Conversation history (not supported in batch mode)
            return_history: Whether to return history (not supported)
            IMG_START_TOKEN: Image start token
            IMG_END_TOKEN: Image end token
            with_motion: Whether to use motion-aware generation
            IMG_CONTEXT_TOKEN: Image context token
            verbose: Whether to print debug info
            image_counts: Deprecated, use num_patches_list
            return_mano: Whether to return MANO parameters
            **kwargs: Additional arguments
        
        Returns:
            List of generated responses
        """
        if history is not None or return_history:
            logger.error('Multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError
        
        if image_counts is not None:
            num_patches_list = image_counts
            logger.warning('`image_counts` is deprecated. Please use `num_patches_list` instead.')
        
        # Get image context token ID
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.debug(f'Dynamic ViT batch size: {image_bs}')
        
        # Build queries with proper templates
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            # Get conversation template from VLM adapter
            template = self.vlm_adapter.get_conv_template()
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            
            # Replace image tokens
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        
        # Tokenize all queries
        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = self.device
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        
        if verbose:
            logger.debug(f'Input shape: {input_ids.shape}')
        
        # Generate with or without motion
        if with_motion:
            if answers is None:
                answer_ids = None
            else:
                answer_ids = tokenizer(answers, return_tensors='pt', padding=True)['input_ids'].to(device)
            
            generation_output = self.generate_with_motion(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                answer_ids=answer_ids,
                generation_config=generation_config,
                **kwargs
            )
        else:
            generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                **kwargs
            )
        
        # Decode responses
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=False)
        responses = [self._process_response(response, template) for response in responses]
        
        return responses
    
    def _process_response(self, response, template):
        """Process a single response to extract the assistant's reply."""
        # Find the assistant's response after the separator
        sep_idx = response.rfind(template.sep)
        if sep_idx != -1:
            response = response[sep_idx + len(template.sep):].strip()
        
        # Remove any trailing tokens
        for stop_token in [template.sep2, '</s>', '<|endoftext|>']:
            if stop_token is not None and response.endswith(stop_token):
                response = response[:-len(stop_token)].strip()
        
        return response
    
    def get_generation_metrics(self) -> dict:
        """
        Get current generation configuration metrics.
        
        Returns:
            Dictionary of generation parameters
        """
        return {
            'max_block_num_control': self.max_block_num_control,
            'motion_gt_constraint': self.motion_gt_constraint,
            'alpha': self.alpha,
            'motion_processor_available': (
                self.motion_adapter is not None and 
                hasattr(self.motion_adapter, 'motion_processor') and 
                self.motion_adapter.motion_processor is not None
            )
        }