# BeingVLA Inference Module
#
# This module provides inference capabilities for BeingVLA models,
# particularly for models trained with the InternVL adapter.

from .vla_internvl_inference import main as internvl_inference_main

__all__ = ['internvl_inference_main']