# BeingVLA Training Patches
# Self-contained training utility patches extracted from InternVL
# Originally from InternVL Copyright (c) 2024 OpenGVLab

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.data.data_collator import DataCollatorMixin

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# ============== Safe Token Resizing ==============

def safe_resize_token_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Safely resize token embeddings to match the tokenizer vocabulary size.
    Handles both monolithic models and modular BeingVLA architecture.
    
    Args:
        model: The model whose embeddings to resize
        tokenizer: The tokenizer to match
    """
    target_vocab_size = len(tokenizer)
    
    # Check if this is a BeingVLA model with modular architecture
    if hasattr(model, 'vlm_adapter') and hasattr(model.vlm_adapter, '_language_model'):
        # Resize embeddings in the actual language model
        language_model = model.vlm_adapter._language_model
        logger.info(f'Resizing embeddings in BeingVLA language model from {language_model.config.vocab_size} to {target_vocab_size}')
        language_model.resize_token_embeddings(target_vocab_size)
        
        # Update config  
        language_model.config.vocab_size = target_vocab_size
        
        # Also update the main model config if it exists
        if hasattr(model.config, 'llm_config'):
            if hasattr(model.config.llm_config, 'vocab_size'):
                model.config.llm_config.vocab_size = target_vocab_size
            else:
                # If it's a dict, use dict assignment
                if isinstance(model.config.llm_config, dict):
                    model.config.llm_config['vocab_size'] = target_vocab_size
        
        logger.info(f'Successfully resized BeingVLA embeddings to {target_vocab_size}')
        
    else:
        # Fallback to original behavior for monolithic models
        logger.info(f'Resizing embeddings in monolithic model to {target_vocab_size}')
        model.resize_token_embeddings(target_vocab_size)
        
        if hasattr(model.config, 'text_config'):
            model.config.text_config.vocab_size = target_vocab_size
        else:
            model.config.vocab_size = target_vocab_size


# ============== Data Collators ==============

def concat_pad_data_collator(features: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    """
    Data collator that concatenates and pads features for batch processing.
    
    Args:
        features: List of feature dictionaries
        pad_id: Padding token ID
        
    Returns:
        Batched and padded features
    """
    first = features[0]
    batch = {}
    
    for key in first.keys():
        if key == 'labels':
            # Special handling for labels - use IGNORE_TOKEN_ID for padding
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f[key]) for f in features],
                batch_first=True,
                padding_value=IGNORE_TOKEN_ID
            )
        else:
            # Regular padding for other keys
            if isinstance(first[key], torch.Tensor):
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=pad_id
                )
            elif isinstance(first[key], (int, float)):
                batch[key] = torch.tensor([f[key] for f in features])
            else:
                batch[key] = [f[key] for f in features]
    
    return batch


def pad_data_collator(features: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Simple padding data collator using the tokenizer's pad token.
    
    Args:
        features: List of feature dictionaries
        tokenizer: Tokenizer for getting pad token ID
        
    Returns:
        Batched and padded features
    """
    return concat_pad_data_collator(features, pad_id=tokenizer.pad_token_id)


# ============== Trainer Patches ==============

# ============== Train Sampler Replacement ==============

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """Get indices grouped by length for efficient batching."""
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length.
    """
    def __init__(self, batch_size: int, world_size: int, dataset=None, lengths=None, model_input_name=None, generator=None):
        if dataset is None and lengths is None:
            raise ValueError('One of dataset and lengths must be provided.')

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else 'input_ids'
            from transformers.tokenization_utils_base import BatchEncoding
            if (not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                    or model_input_name not in dataset[0]):
                raise ValueError(
                    'Can only automatically infer lengths for datasets whose items are dictionaries with an '
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info('Converting lengths tensor to list...')
            lengths = lengths.tolist()
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def _get_train_sampler(self):
    """Get train sampler for the Trainer class."""
    from transformers.trainer import has_length, RandomSampler
    
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    
    # Build the sampler.
    if self.args.group_by_length:
        lengths = []
        if hasattr(self.train_dataset, 'datasets'):
            # ConcatDataset case
            for dataset in self.train_dataset.datasets:
                if hasattr(dataset, 'length'):
                    lengths = lengths + dataset.length
                else:
                    # Fallback: compute lengths
                    lengths = lengths + [len(self.tokenizer(str(item), return_tensors='pt').input_ids[0]) for item in dataset]
        else:
            # Single dataset case
            if hasattr(self.train_dataset, 'length'):
                lengths = self.train_dataset.length
            else:
                lengths = [len(item['input_ids']) if 'input_ids' in item else 100 for item in self.train_dataset]
                
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )
    else:
        return RandomSampler(self.train_dataset)


def replace_train_sampler():
    """
    Patch the Trainer class to use our custom _get_train_sampler method.
    """
    import transformers
    transformers.Trainer._get_train_sampler = _get_train_sampler
    logger.info("Patched Trainer._get_train_sampler for grouped length sampling")


def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training DataLoader with custom settings for packed datasets.
    
    This method is patched onto the Trainer class to handle packed dataset logic.
    """
    if self.train_dataset is None:
        raise ValueError('Trainer: training requires a train_dataset.')

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    
    # Handle datasets library if available
    import datasets
    from transformers.trainer import is_datasets_available, seed_worker
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description='training')
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

    dataloader_params = {
        'batch_size': self._train_batch_size,
        'collate_fn': data_collator,
        'num_workers': self.args.dataloader_num_workers,
        'pin_memory': self.args.dataloader_pin_memory,
        'persistent_workers': self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = self._get_train_sampler()
        dataloader_params['drop_last'] = self.args.dataloader_drop_last
        dataloader_params['worker_init_fn'] = seed_worker

    # Special handling for packed datasets
    if getattr(self.args, 'use_packed_ds', False):
        return DataLoader(train_dataset, **dataloader_params)
    
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def replace_train_dataloader():
    """
    Patch the Trainer class to use our custom get_train_dataloader method.
    """
    import transformers
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    logger.info("Patched Trainer.get_train_dataloader for packed dataset support")


# ============== Additional Collators (for completeness) ==============

def concat_robot_pad_data_collator(features: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    """
    Data collator for robot/motion data that needs special handling.
    
    Handles proprioception values and action labels in addition to standard data.
    """
    # First get the standard collated batch
    batch = concat_pad_data_collator(features, pad_id)
    
    # Handle proprioception values if present
    if 'proprio' in features[0]:
        proprio_list = [f['proprio'] for f in features]
        batch['proprioception_values'] = torch.stack(proprio_list)
    
    # Handle action labels if present
    if 'action' in features[0]:
        action_list = [f['action'] for f in features]
        batch['action_labels'] = torch.stack(action_list)
    
    return batch


def dpo_concat_pad_data_collator(features: List[Dict[str, Any]], pad_id: int = 0) -> Dict[str, Any]:
    """
    Data collator for Direct Preference Optimization (DPO) training.
    
    This is a placeholder for DPO-specific data collation logic.
    """
    # For now, just use the regular concat_pad_data_collator
    # In the full implementation, this would handle chosen/rejected pairs
    return concat_pad_data_collator(features, pad_id)


# Export all patch functions
__all__ = [
    'safe_resize_token_embeddings',
    'concat_pad_data_collator',
    'pad_data_collator',
    'replace_train_sampler', 
    'replace_train_dataloader',
    'concat_robot_pad_data_collator',
    'dpo_concat_pad_data_collator',
]