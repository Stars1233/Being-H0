# BeingVLA Constants
# Self-contained constants for BeingVLA framework
# Originally extracted from InternVL for independence

"""
This module contains all constants used throughout the BeingVLA framework.
These constants define special tokens, normalization values, and configuration
parameters that ensure consistent behavior across all components.
"""

# Image tokens - Used to mark image content in text sequences
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

# Box and reference tokens - Used for spatial grounding and object references
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'

# Motion tokens - Used to mark motion sequences in generation
MOT_START_TOKEN = '<mot>'
MOT_END_TOKEN = '</mot>'
MOT_NOOP = '<MOT_NOOP>'

# Image normalization constants - Mean and std values for different vision models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# DeepSpeed compatibility module names - Modules that should not be split during parallelization  
DEEPSPEED_NO_SPLIT_MODULES = [
    'InternVisionModel',
    'Qwen2DecoderLayer'
]

# Flash Attention compatible dtypes - Data types supported by Flash Attention optimization
FLASH_ATTN_SUPPORTED_DTYPES = ['torch.float16', 'torch.bfloat16']
FLASH_ATTN_CREATION_SAFE_DTYPE = 'torch.float32'