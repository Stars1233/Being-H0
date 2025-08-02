#!/usr/bin/env python3

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, List
from tqdm import tqdm
import re 
import h5py
import numpy as np
from torch.utils.data import DataLoader
import json
import cv2
import torch

# BeingVLA imports (replacing internvl imports)
from beingvla.models.vla import BeingVLAModel
from beingvla.utils.constants import MOT_END_TOKEN, MOT_START_TOKEN

from PIL import Image, ImageFile, PngImagePlugin
from decord import VideoReader
from transformers import (AutoTokenizer, HfArgumentParser)

from beingvla.models.motion.m2m.tokenizer.config import get_eval_config, MotionArguments, calculate_num_tokens, calculate_codebook_size
from beingvla.models.motion.m2m.utils.word_vectorizer import WordVectorizer
from beingvla.dataset.dataset_internvl import LazySupervisedDataset
from beingvla.models.motion.m2m.utils.mano_utils import vector_to_mano
from beingvla.models.motion.m2m.utils.mano_vis import mano_forward, world_hand_to_camera, hand_2d_render, hand_render
from beingvla.inference.utils import dynamic_preprocess, build_transform

import time

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

@dataclass
class ModelArguments:
    """
    Arguments for Evaluation model.
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to the pretrained model or model identifier from HuggingFace Hub.'
        },
    )


@dataclass
class InferenceArguments:
    """Arguments for inference."""
    input_intrinsic: Optional[List[float]] = field(
        default=None,
        metadata={"help": "A list of 4 floats representing the intrinsic parameters (e.g., fx, fy, cx, cy)."}
    )
    num_seconds: Optional[int] = field(
        default=4,
        metadata={"help": "Number of seconds of motion to generate (default: 4, each second is 15 frames)"}
    )
    num_samples: Optional[int] = field(
        default=0,
        metadata={"help": "Number of response samples to generate (default: 0)"}
    )
    enable_render: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable video rendering (default: False)"}
    )
    output_dir: Optional[str] = field(
        default="./work_dirs",
        metadata={"help": "Output directory for results (default: ./work_dirs)"}
    )
    input_image: Optional[str] = field(
        default=None,
        metadata={"help": "Path to input image file (required)"}
    )
    gpu_device: Optional[int] = field(
        default=0,
        metadata={"help": "GPU device number to use (default: 0)"}
    )
    task_description: Optional[str] = field(
        default="unplug the charging cable from the AirPods",
        metadata={"help": "Task description for motion generation (required)"}
    )
    hand_mode: Optional[str] = field(
        default="both",
        metadata={"help": "Hand mode: 'both', 'left', or 'right' (default: both)"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, MotionArguments, InferenceArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, motion_args, inference_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, motion_args, inference_args = parser.parse_args_into_dataclasses()
    
    # Load model using standard from_pretrained with motion_code_path parameter
    logger.info(f"Loading model from: {model_args.model_path}")
    if motion_args.motion_code_path:
        logger.info(f"Using motion_code_path override: {motion_args.motion_code_path}")
    else:
        logger.info("Using model's saved motion configuration")
    
    model = BeingVLAModel.from_pretrained(
        model_args.model_path,
        motion_code_path=motion_args.motion_code_path,  # Will be None if not provided
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        use_flash_attn=True,
        build_motion_model=True
    ).eval()
    
    # Get motion_args_list from model for compatibility with rest of the script
    if hasattr(model, 'motion_adapter') and model.motion_adapter is not None and hasattr(model.motion_adapter, '_motion_config'):
        motion_args_list = model.motion_adapter._motion_config
    elif hasattr(model, 'config') and hasattr(model.config, 'motion_config') and model.config.motion_config:
        # Convert saved motion config back to MotionArguments for compatibility
        motion_args_list = []
        motion_configs = model.config.motion_config if isinstance(model.config.motion_config, list) else [model.config.motion_config]
        
        for motion_cfg in motion_configs:
            if isinstance(motion_cfg, dict):
                # Create MotionArguments from dict
                temp_args = MotionArguments()
                for key, value in motion_cfg.items():
                    if hasattr(temp_args, key):
                        setattr(temp_args, key, value)
                motion_args_list.append(temp_args)
        
        if not motion_args_list:
            # Fallback to default
            motion_args_list = [motion_args]
    else:
        motion_args_list = []
    
    # Check motion adapter initialization
    if model.motion_adapter is None:
        logger.error("Motion adapter is not initialized! Check that the model config has proper motion_config.")
        logger.error(f"Model config motion_config: {model.config.motion_config if hasattr(model.config, 'motion_config') else 'Not found'}")
        logger.error(f"Model config motion_type: {model.config.motion_type if hasattr(model.config, 'motion_type') else 'Not found'}")
        return
    
    # Log the motion code paths being used
    if motion_args_list:
        logger.info("Motion code paths in use:")
        for i, m_args in enumerate(motion_args_list):
            if hasattr(m_args, 'motion_code_path') and m_args.motion_code_path:
                logger.info(f"  Motion {i}: {m_args.motion_code_path}")
            else:
                logger.info(f"  Motion {i}: (from model config)")
    
    # Create output directory
    os.makedirs(inference_args.output_dir, exist_ok=True)
    
    # Set device based on arguments
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{inference_args.gpu_device}')
        logger.info(f"Using GPU device: cuda:{inference_args.gpu_device}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, use_fast=False)        
    tokenizer.padding_side = "left"  # it's better for motion generation
    logger.info(f"padding side: {tokenizer.padding_side}")
    logger.info(f"truncation side: {tokenizer.truncation_side}")


    """building motion processor"""
    # Note: codebook_size_list is calculated but not used in inference
    # It's kept for potential future use or debugging
    _ = calculate_codebook_size(motion_args_list)
    
    model.build_motion_processor(tokenizer, allow_non_motion=True, fixed_block_length_control=True, max_block_num_control=True, min_block_num_control=True, gt_constraint = 'none')
    
    transform = build_transform(is_train=False, input_size=448, pad2square=False,normalize_type='imagenet')
    
    # Validate hand mode
    if inference_args.hand_mode not in ['both', 'left', 'right']:
        logger.error(f"Invalid hand_mode: {inference_args.hand_mode}. Must be 'both', 'left', or 'right'")
        return
    
    # Set side_list based on hand_mode
    if inference_args.hand_mode == 'both':
        side_list = ['left', 'right']
    elif inference_args.hand_mode == 'left':
        side_list = ['left']
    else:  # right
        side_list = ['right']
    
    # Load input image
    if inference_args.input_image and not os.path.exists(inference_args.input_image):
        logger.error(f"Input image not found: {inference_args.input_image}")
        return
    
    # Load input image/video
    if inference_args.input_image.endswith(('.mp4', '.avi', '.mov')):
        raw_frame = VideoReader(inference_args.input_image).get_batch([60]).asnumpy()[0]
    else:
        pil_image = Image.open(inference_args.input_image).convert('RGB')
        raw_frame = np.array(pil_image)
    
    logger.debug(f'raw_frame shape: {raw_frame.shape}')
    
    # Create reference frame
    frame = cv2.resize(raw_frame, (448, 448), interpolation=cv2.INTER_LINEAR)
    _frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ref_frame_path = os.path.join(inference_args.output_dir, 'test_frame.jpg')
    cv2.imwrite(ref_frame_path, _frame)
    target_img_size = (1920,1080) 
    
    # Camera intrinsics
    if inference_args.input_intrinsic is not None:
        fx, fy, cx, cy = inference_args.input_intrinsic
        raw_camera_intrinsic = torch.tensor([[fx, 0., cx],
                                            [0, fy, cy],
                                            [0, 0, 1]], device=device, dtype=torch.float32)
    else:
        raw_camera_intrinsic = torch.tensor([[2304, 0. , 1729],
                                             [0, 1942, 971],
                                            [0, 0, 1]], device=device, dtype=torch.float32)
    target_camera_intrinsic = torch.tensor([[736.6339, 0., 960.], 
                                     [0., 736.6339, 540.], 
                                     [0., 0., 1.]], dtype=torch.float32, device=device)
    H = target_camera_intrinsic @ torch.linalg.inv(raw_camera_intrinsic)
    # Apply camera perspective transformation
    raw_frame = cv2.resize(raw_frame, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    new_raw_frame = cv2.warpPerspective(raw_frame, H.cpu().numpy(), target_img_size,
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
    new_frame = cv2.resize(new_raw_frame, (448, 448), interpolation=cv2.INTER_LINEAR)
    
    # Prepare inference parameters
    sample_num = inference_args.num_samples
    generate_seconds = inference_args.num_seconds
    step_num = generate_seconds * 15
    
    # Task description
    task = inference_args.task_description
    
    # Construct question based on hand_mode
    if inference_args.hand_mode == 'both':
        hand_phrase = "the both hands"
    elif inference_args.hand_mode == 'left':
        hand_phrase = "the left hand"
    else:  # right
        hand_phrase = "the right hand"
    
    question = [f'In <image>, {task}, what is the {step_num}-step motion sequence for {hand_phrase}?']*sample_num 
    # Create PIL Image from RGB array
    images = dynamic_preprocess(Image.fromarray(new_frame), min_num=1, max_num=12, image_size=448, use_thumbnail=False)
    logger.debug(f'images length: {len(images)}')
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).repeat(sample_num, 1, 1, 1)
    logger.debug(f'pixel_values shape: {pixel_values.shape}')
    pattern_between = re.escape(MOT_START_TOKEN) + r'(.*?)' + re.escape(MOT_END_TOKEN)
    token_re = re.compile(r'<motion_id_(\d+)>')
    
    # Generate motion sequences
    logger.info(f"Generating {sample_num} motion sequences with {generate_seconds} seconds each...")
    
    with torch.no_grad():
        start_time = time.time()
        # Set motion processor constraints for fixed length generation
        if hasattr(model, 'motion_adapter') and model.motion_adapter is not None:
            if hasattr(model.motion_adapter, 'motion_processor') and model.motion_adapter.motion_processor is not None:
                # Set fixed block number control
                model.motion_adapter.motion_processor.set_block_num_control(
                    min_block_num=generate_seconds * len(side_list),
                    max_block_num=generate_seconds * len(side_list)
                )
                logger.debug(f"Set motion processor to generate exactly {generate_seconds * len(side_list)} segments")
            
        response = model.batch_chat(tokenizer, pixel_values.to(dtype=torch.bfloat16,device=device), question, 
                            generation_config=dict(max_new_tokens=3096, do_sample=True),with_motion=True,
                            num_patches_list=[1]*sample_num)
        generation_time = time.time() - start_time
    
    logger.info(f"Generation completed in {generation_time:.2f} seconds")
    
    # Process responses
    logger.info(f"Processing {len(response)} responses...")
    
    # Prepare results storage
    results = []
    
    for t, single_response in enumerate(response):
        print(f"\n{'='*60}")
        print(f"Sample {t+1}/{len(response)}")
        print(f"{'='*60}")
        
        response_segments = re.findall(pattern_between, single_response)
        logger.debug(f"Found {len(response_segments)} motion segments in response")
        beta = {'left': torch.tensor([ 1.1270238 , -0.25079092, -2.2111652 ,  4.350904  , -2.3538678 ,
                                    3.7198792 ,  0.80536413,  0.9086072 , -0.41488478,  1.8571062 ], device=device),
                'right': torch.tensor([ 0.50233305, -1.332463  , -5.994158  , -1.2552336 ,  2.1817644 ,
                                    -5.384019  , -2.189624  ,  2.3371675 , -4.9490705 , -2.3550193 ], device=device)}

        mano_code_response = []
        motion_tokens_list = []
    
        for seg_idx, seg in enumerate(response_segments):
            mot_ids = []
            for match in token_re.finditer(seg):
                if match.group(1):
                    mot_ids.append(int(match.group(1)))
            mot_ids = np.array(mot_ids, dtype=np.int64)
            if len(mot_ids) > 0:
                print(f'  Segment {seg_idx}: {mot_ids.shape[0]} tokens')
                print(f'  Token IDs: {mot_ids.tolist()}')
            motion_tokens_list.append(mot_ids.tolist())
            mano_code_response.append(torch.from_numpy(mot_ids).to(device=device))
        # Process motion tokens
        if mano_code_response:
            all_tokens = torch.cat(mano_code_response, dim=0)
            total_tokens = all_tokens.shape[0]
            
            tokens_per_segment = 128
            expected_segments = generate_seconds * len(side_list)
            expected_total = expected_segments * tokens_per_segment
            
            logger.debug(f'Total tokens received: {total_tokens}, expected: {expected_total}')
            
            # Pad or truncate to expected size
            if total_tokens < expected_total:
                padding = torch.zeros(expected_total - total_tokens, device=device, dtype=all_tokens.dtype)
                all_tokens = torch.cat([all_tokens, padding], dim=0)
                logger.debug(f'Padded {expected_total - total_tokens} tokens')
            elif total_tokens > expected_total:
                all_tokens = all_tokens[:expected_total]
                logger.debug(f'Truncated to {expected_total} tokens')
                
            mano_code_response = all_tokens.reshape(1, generate_seconds, len(side_list), tokens_per_segment)
            logger.debug(f'Final mano_code_response shape: {mano_code_response.shape}')
        else:
            logger.error(f'No motion tokens found in response for sample {t+1}')
            continue
        
        # Decode motion tokens to MANO parameters
        try:
            mano_vec = model.motion_block_id_to_mano(mano_code_response, offset=False, denormalize=True, return_list=False)
            logger.debug(f'mano_vec shape: {mano_vec.shape}')
        except Exception as e:
            logger.error(f'Error decoding motion for sample {t+1}: {e}')
            continue
        
        response_hand_pose_rot, response_hand_pose_trans, response_hand_pose_theta, response_hand_pose_beta = vector_to_mano(mano_vec)
        response_hand_pose = dict(beta={}, theta={}, trans={}, rot={}) 
        
        logger.debug(f'response_hand_pose_rot shape: {response_hand_pose_rot.shape}')
        logger.debug(f'response_hand_pose_trans shape: {response_hand_pose_trans.shape}')
        logger.debug(f'response_hand_pose_theta shape: {response_hand_pose_theta.shape}')
        logger.debug(f'response_hand_pose_beta shape: {response_hand_pose_beta.shape if response_hand_pose_beta is not None else "None"}')
        
        # Reshape MANO parameters
        actual_seconds = mano_code_response.shape[1]
        for i,side in enumerate(side_list):
            response_hand_pose['theta'][side] = response_hand_pose_theta[:, :, i].reshape(-1, 45)
            response_hand_pose['rot'][side] = response_hand_pose_rot[:, :, i].reshape(-1, 3, 3)
            response_hand_pose['trans'][side] = response_hand_pose_trans[:, :, i].reshape(-1, 3)                                
            response_hand_pose['beta'][side] = response_hand_pose_beta[:, :, i].reshape(-1,10) \
                if response_hand_pose_beta is not None else beta[side]
        
        num_frames = response_hand_pose['theta'][side_list[0]].shape[0]
        logger.info(f"Sample {t+1}: Generated {num_frames} frames ({actual_seconds} seconds)")
        response_hand_data = mano_forward(**response_hand_pose, sides=side_list, relative=True)
        cpu_hand_data = {side: {
            'joints': response_hand_data[side]['joints'].cpu(),
            'verts': response_hand_data[side]['verts'].cpu(),
            'faces': response_hand_data[side]['faces'].cpu(),
        } for side in side_list}
        
        # Store result data (without motion_tokens)
        result = {
            'sample_id': t + 1,
            'text_response': single_response,
            'num_segments': len(response_segments),
            'num_seconds': actual_seconds,
            'num_frames': num_frames,
            'sides': side_list,
            'generation_info': {
                'task_description': task,
                'hand_mode': inference_args.hand_mode,
                'requested_seconds': generate_seconds,
                'actual_seconds': actual_seconds
            }
        }
        
        # Get fps from motion config
        fps = motion_args_list[0].window_size if motion_args_list else 15
        
        # Render video if enabled
        if inference_args.enable_render:
            output_video_path = os.path.join(inference_args.output_dir, f'response_{t}.mp4')
            logger.info(f"  Rendering video...")
            hand_2d_render(cpu_hand_data, target_camera_intrinsic, output_video_path, new_raw_frame, fps=fps, blend_alpha=0.8)
            logger.info(f"  Video saved: {output_video_path}")
            result['video_path'] = output_video_path
        
        results.append(result)
    
    # Save detailed results to JSON
    output_json_path = os.path.join(inference_args.output_dir, 'inference_results.json')
    output_data = {
        'model_path': model_args.model_path,
        'input_image': inference_args.input_image,
        'task_description': task,
        'hand_mode': inference_args.hand_mode,
        'inference_config': {
            'num_samples': sample_num,
            'requested_seconds': generate_seconds,
            'generation_time': generation_time,
            'device': str(device)
        },
        'results': results
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    logger.info("=" * 50)
    logger.info(f"Inference Summary:")
    logger.info(f"  Model: {model_args.model_path}")
    logger.info(f"  Task: {task}")
    logger.info(f"  Hand mode: {inference_args.hand_mode}")
    logger.info(f"  Samples generated: {len(results)}")
    logger.info(f"  Total time: {generation_time:.2f}s")
    
    for result in results:
        logger.info(f"  Sample {result['sample_id']}: {result['num_frames']} frames ({result['num_seconds']} seconds)")
    
    logger.info(f"  Results saved to: {output_json_path}")
    if inference_args.enable_render:
        logger.info(f"  Videos saved to: {inference_args.output_dir}")
    logger.info("=" * 50)
    
if __name__ == '__main__':
    main()