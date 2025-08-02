
import re
from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from argparse import Namespace
import transformers


@dataclass
class MotionArguments:
    # Data loading
    motion_feat: str = field(default="smpl263", metadata={"help": "pose feature type"})
    fps: int = field(default=20, metadata={"help": "frames per second"})
    unit_length: int = field(default=4, metadata={"help": "frames per state"})
    seq_len: int = field(default=64, metadata={"help": "training motion length"})
    val_split_file: str = field(default="val/test.txt", metadata={"help": "validation split file"})
    min_motion_len: int = field(default=40, metadata={"help": "minimum motion length"})
    max_motion_len: int = field(default=200, metadata={"help": "maximum motion length"})
    max_eval_motion_len: int = field(default=80, metadata={"help": "maximum motion length"})
    tmr_model_path: str = field(default="", metadata={"help": "path for text-motion-retrieval model"})

    # VQ-VAE Architecture
    code_dim: int = field(default=512, metadata={"help": "embedding dimension"})
    motion_unit_length: int = field(default=4, metadata={"help": "motion unit length"})
    nb_code: int = field(default=512, metadata={"help": "number of embeddings"})
    mu: float = field(default=0.99, metadata={"help": "EMA decay for codebook updates"})
    down_t: int = field(default=2, metadata={"help": "downsampling rate"})
    stride_t: int = field(default=2, metadata={"help": "stride size"})
    width: int = field(default=512, metadata={"help": "network width"})
    depth: int = field(default=3, metadata={"help": "network depth"})
    dilation_growth_rate: int = field(default=3, metadata={"help": "dilation growth rate"})
    output_emb_width: int = field(default=512, metadata={"help": "output embedding width"})
    activate: str = field(default="relu", metadata={
        "help": "Activation function",
        "choices": ["relu", "silu", "gelu"]
    })
    norm: str = field(default=None, metadata={"help": "normalization layer type"})
    nb_joints: int = field(default=22, metadata={"help": "number of joints"})
    window_size: int = field(default=64, metadata={"help": "training motion window size"})
    eval_window_size: int = field(default=15, metadata={"help": "evaluation motion window size"})
    frame_interval: int = field(default=2, metadata={"help": "frame interval"})
    chunk_size: int = field(default=10, metadata={"help": "chunk size for training"})
    relative: bool = field(default=True, metadata={"help": "relative mano"})
    multi_coords: bool = field(default=False, metadata={"help": "use multi-coordinates for motion"})
    normalize: bool = field(default=True, metadata={"help": "normalize motion features"})
    num_conv_layers: int = field(default=1, metadata={"help": "number of conv layer for Encoder/Decoder"})
    use_part: str = field(default=None, metadata={"help": "use part for motion feature", "choices": [None, "wrist", "finger", "both"]})

    # Quantization
    quantizer_name: str = field(
        default=None,
        metadata={
            "help": "Quantizer type",
            "choices": [
                None, 'ema', 'orig', 'ema_reset', 'reset', 'residualvq',
                "part_residualvq", "group_residualvq", "part_group_residualvq", 
                "lfq", "fsq"
            ]
        }
    )
    quantbeta: float = field(default=1.0, metadata={"help": "quantization beta"})
    num_quantizers: int = field(default=8, metadata={"help": "number of quantizers for Res_VQ"})
    num_quant_groups: int = field(default=2, metadata={"help": "number of groups for Group_Res_VQ"})
    shared_codebook: bool = field(default=False, metadata={"help": "share codebook for Res_VQ"})

    # Loss functions
    recons_loss: str = field(default="l2", metadata={"help": "reconstruction loss type"})
    commit_weight: float = field(default=0.02, metadata={"help": "commitment loss weight"})
    vel_weight: float = field(default=0.5, metadata={"help": "velocity loss weight"})
    
    # Training state
    new_token_type: Optional[str] = field(default="insert")
    max_text_len: Optional[int] = field(default=20)
    depth_first: bool = field(
        default=True,
        metadata={"help": "flatten multi-layer motion codes depth-first"}
    )

    is_hm3d_old: bool = field(
        default=False,
        metadata={"help": "use HM3D old feature for train/val"}
    )
    instruction: Optional[str] = field(
        default=None,
        metadata={"help": "text instruction for inference"}
    )

    motion_resume_pth: Optional[str] = field(default=None, metadata={"help": "path to resume checkpoint"})
    cache_path: str = field(default=None, metadata={"help": "Path to store the motion token cache"})
    motion_feat_dir: str = field(default=None, metadata={"help": "Path to store the motion feature"})
    motion_code_path: str = field(default=None, metadata={"help": "Path to store the quantized motion code"})

@dataclass
class DataArguments:
    dataset: str = field(default="motionx", metadata={"help": "Dataset name"})
    val_dataset: str = field(default="motionx", metadata={"help": "Dataset name"})
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    cache_file_dir: str = field(default=None, metadata={"help": "mano cache directory"})
    cache_name: str = field(default=None, metadata={"help": "mano cache name"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    lr_scheduler_gamma: float = field(default=0.05, metadata={"help": "learning rate decay"})
    lr_scheduler_milestones: List[int] = field(
        default_factory=lambda: [200000, 400000],  # Use default_factory to set default list
        metadata={"help": "learning rate schedule (iterations)"}
    )

    is_eval: bool = field(default=False)
    resume_pth: Optional[str] = field(default=None, metadata={"help": "path to resume checkpoint"})
    debugging: bool = field(default=False, metadata={"help": "whether to use debug mode"})

def calculate_num_tokens(args_list: List[MotionArguments]) -> List[int]:
    chunk_shape_list = []
    res_len_list = []
    for args in args_list:
        window_size = args.window_size
        downsample_unit_len = 2 ** args.down_t
        res_len = downsample_unit_len - window_size % downsample_unit_len
        window_length = (window_size + res_len) // downsample_unit_len
    
        chunk_shape = [window_length, ]
        if 'residualvq' in args.quantizer_name:
            chunk_shape = chunk_shape + [args.num_quantizers, ]
        if 'group' in args.quantizer_name:
            chunk_shape = [args.num_quant_groups, ] + chunk_shape
        chunk_shape_list.append(chunk_shape)
        res_len_list.append(res_len)
    
    return chunk_shape_list, res_len_list

def calculate_codebook_size(args_list: List[MotionArguments]) -> List[int]:
    codebook_size_list = []
    for args in args_list:
        if 'residualvq' in args.quantizer_name and not args.shared_codebook:
            codebook_size = args.nb_code * args.num_quantizers
        else:
            codebook_size = args.nb_code
        codebook_size_list.append(codebook_size)
    
    return codebook_size_list
# [0] + list(np.cumsum(codebook_size_list))

def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_eval_config(cfg_path, device):
    cfg = Namespace()
    cfg_dict = vars(cfg)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')

    print('Reading', cfg_path)
    with open(cfg_path) as f:
        for line in f:
            
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    cfg_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    cfg_dict[key] = float(value)
                elif is_number(value):
                    cfg_dict[key] = int(value)
                else:
                    cfg_dict[key] = str(value)

    cfg_dict['which_epoch'] = 'finest'
    cfg.save_root = pjoin(cfg.checkpoints_dir, cfg.dataset_name, cfg.name)
    cfg.model_dir = pjoin(cfg.save_root, 'model')
    cfg.meta_dir = pjoin(cfg.save_root, 'meta')

    cfg.dim_word = 300
    cfg.num_classes = 200 // cfg.unit_length
    cfg.is_train = False
    cfg.is_continue = False
    cfg.device = device

    return cfg