# BeingVLA Training Mixins
#
# This module contains training-specific logic extracted from BeingVLAModel to achieve
# better separation of concerns and modular design.

from typing import Optional
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BeingVLATrainingMixin:
    """
    Training-specific functionality for BeingVLA models.
    
    This mixin contains all training-related logic including motion space optimization,
    curriculum learning, and motion-aware loss computation.
    """
    
    def __init_training_attributes__(self):
        """Initialize training-related attributes."""
        # Motion optimization attributes
        self.optimize_rate = 0.0
        self.curr_top_rate = 1.0
        self.curr_bottom_rate = 0.0
        self.vocab_mask = None
        self.mot_start_id = None
        self.mot_end_id = None
    
    def set_optimize_motion(self, optimize_rate: float, mot_start_id: int, mot_end_id: int, 
                           curr_top_rate: float = 1.0, curr_bottom_rate: float = 0.0):
        """
        Set motion optimization parameters for training.
        
        Args:
            optimize_rate: Rate of motion space optimization (0.0 to 1.0)
            mot_start_id: Token ID for motion start token
            mot_end_id: Token ID for motion end token  
            curr_top_rate: Top percentile for curriculum learning
            curr_bottom_rate: Bottom percentile for curriculum learning
        """
        self.optimize_rate = optimize_rate
        self.mot_start_id = mot_start_id
        self.mot_end_id = mot_end_id
        
        # Create vocabulary mask for non-motion tokens
        # Use actual vocab size from embeddings in case tokens were added
        if hasattr(self.language_model, 'get_input_embeddings'):
            actual_vocab_size = self.language_model.get_input_embeddings().weight.shape[0]
        else:
            actual_vocab_size = self.language_model.config.vocab_size
            
        self.vocab_mask = torch.zeros(actual_vocab_size, dtype=torch.bool)
        self.vocab_mask[:mot_start_id] = True
        if mot_end_id + 1 < actual_vocab_size:
            self.vocab_mask[mot_end_id + 1:] = True
        
        # Set curriculum learning parameters
        self.curr_top_rate = curr_top_rate
        self.curr_bottom_rate = curr_bottom_rate
        
        logger.info(f'Motion optimization configured: rate={optimize_rate}, '
                   f'curriculum=({curr_bottom_rate}, {curr_top_rate})')
    
    def _apply_motion_training_logic(self, outputs: CausalLMOutputWithPast, 
                                   labels: torch.LongTensor) -> CausalLMOutputWithPast:
        """
        Apply motion-specific training logic to model outputs.
        
        This method implements:
        1. Motion space optimization - constrains predictions on motion tokens
        2. Curriculum learning - filters losses based on difficulty percentiles
        
        Args:
            outputs: Raw model outputs
            labels: Training labels
            
        Returns:
            Modified outputs with motion-aware loss
        """
        logits = outputs.logits
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Apply motion space optimization if enabled
        if self.optimize_rate > 0.0:
            # Identify motion tokens in labels
            mot_mask = (shift_labels > self.mot_start_id) & (shift_labels < self.mot_end_id)
            
            # Randomly mask non-motion predictions on motion tokens
            random_mask = torch.rand(shift_labels.shape, device=shift_logits.device) < self.optimize_rate
            non_mot_space_mask = (mot_mask & random_mask).unsqueeze(-1) & self.vocab_mask.to(shift_logits.device)
            
            # Set non-motion predictions to negative infinity on motion tokens
            shift_logits.masked_fill_(non_mot_space_mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Use actual vocab size from tensor shape to handle added tokens
        actual_vocab_size = shift_logits.shape[-1]
        shift_logits_flat = shift_logits.view(-1, actual_vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        shift_labels_flat = shift_labels_flat.to(shift_logits_flat.device)
        loss = loss_fct(shift_logits_flat, shift_labels_flat)
        
        # Apply curriculum learning if enabled
        if self.curr_top_rate < 1.0 or self.curr_bottom_rate > 0.0:
            loss = self._apply_curriculum_learning(loss)
        else:
            loss = loss.mean()
        
        # Return updated outputs with motion-aware loss
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _apply_curriculum_learning(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply curriculum learning by filtering losses based on difficulty percentiles.
        
        Args:
            loss: Per-token losses
            
        Returns:
            Filtered average loss
        """
        with torch.no_grad():
            # Only consider valid (non-zero) losses
            valid_loss = loss[loss > 0.0]
            
            if len(valid_loss) > 0:
                # Calculate percentile thresholds
                top_threshold = torch.quantile(valid_loss, self.curr_top_rate)
                bottom_threshold = torch.quantile(valid_loss, self.curr_bottom_rate)
                
                # Create mask for losses within curriculum range
                loss_mask = (loss >= bottom_threshold) & (loss <= top_threshold) & (loss > 0.0)
            else:
                # If no valid losses, use all losses
                loss_mask = torch.ones_like(loss, dtype=torch.bool)
        
        # Apply curriculum mask and compute average
        filtered_loss = loss * loss_mask.float()
        num_valid_tokens = loss_mask.sum()
        
        if num_valid_tokens > 0:
            return filtered_loss.sum() / num_valid_tokens
        else:
            return loss.mean()
    
    def get_training_metrics(self) -> dict:
        """
        Get current training configuration metrics.
        
        Returns:
            Dictionary of training parameters
        """
        return {
            'optimize_rate': self.optimize_rate,
            'curr_top_rate': self.curr_top_rate,
            'curr_bottom_rate': self.curr_bottom_rate,
            'motion_optimization_enabled': self.optimize_rate > 0.0,
            'curriculum_learning_enabled': (self.curr_top_rate < 1.0 or self.curr_bottom_rate > 0.0),
        }