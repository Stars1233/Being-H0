# BeingVLA InternVL Configuration Classes
# Self-contained configuration classes extracted from InternVL for independence
# Originally from InternVL Copyright (c) 2024 OpenGVLab, Licensed under The MIT License

import copy
from transformers import AutoConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
# No additional constants needed - using direct values

logger = logging.get_logger(__name__)


class InternVisionConfig(PretrainedConfig):
    """
    Configuration class for InternVision model.
    Self-contained version extracted from InternVL.
    """
    model_type = 'intern_vit'

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        patch_size=14,
        num_heads=16,
        num_layers=40,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        drop_rate=0.0,
        init_values=0.1,
        use_flash_attn=True,
        qk_normalization=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.init_values = init_values
        self.use_flash_attn = use_flash_attn
        self.qk_normalization = qk_normalization


class InternVLChatConfig(PretrainedConfig):
    """
    Configuration class for InternVL Chat model.
    Self-contained version extracted from InternVL.
    """
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version='v1',
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        **kwargs
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            default_architecture = ''
            llm_config = {'architectures': [default_architecture]}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)
        
        # Handle only supported LLM architectures
        if llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError(
                f'Unsupported architecture: {llm_config["architectures"][0]}. '
                'Only Qwen2ForCausalLM is supported.'
            )
            
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        self.hidden_size = self.llm_config.hidden_size
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output


class InternVLMotionConfig(InternVLChatConfig):
    """
    Configuration class for InternVL Motion model.
    Self-contained version extracted from InternVL.
    """
    model_type = 'internvl_motion'
    is_composition = True

    def __init__(
        self,
        motion_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if isinstance(motion_config, dict):
            self.motion_config = motion_config
        elif isinstance(motion_config, (list, tuple)):
            flag = False
            for item in motion_config:
                if not isinstance(item, dict):
                    flag = True
                    break
            if flag:
                logger.warning("motion_config should be a list of dicts. Initializing as an empty list.")
                self.motion_config = {}
            else:
                self.motion_config = motion_config
        else:
            logger.warning("motion_config should be a dict. Initializing as an empty dictionary.")
            self.motion_config = {}

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = super().to_dict()
        output['motion_config'] = self.motion_config
        return output