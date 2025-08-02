# BeingVLA (Vision-Language-Action) Modular Framework
# 
# This package provides a modular architecture for BeingVLA models that separates
# concerns between vision-language models and motion processing.

from .models.vla.being_vla_model import BeingVLAModel
from .models.vla.config import BeingVLAConfig

__version__ = "0.1.0"
__all__ = ["BeingVLAModel", "BeingVLAConfig"]