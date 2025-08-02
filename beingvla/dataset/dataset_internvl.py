# InternVL-specific Dataset Classes
# Combined LazySupervisedDataset and PackedDataset for InternVL training
# Originally from InternVL Copyright (c) 2024 OpenGVLab

import bisect
import copy
import logging
import os
import torch
import random
import json
import traceback
import numpy as np
import pickle as pkl
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
from copy import deepcopy
from decord import VideoReader
from typing import Dict, List, Union
from tqdm import tqdm
import yaml

import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers.trainer_pt_utils import LabelSmoother

from .datasets import (
    build_transform, dynamic_preprocess,
    preprocess, preprocess_mpt
)
from ..utils.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, MOT_START_TOKEN, MOT_END_TOKEN, MOT_NOOP
from ..utils.conversation import get_conv_template

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    """Preprocess conversations for InternVL2.5 format."""
    import torch.nn.functional as F
    
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


# ============== Packed Dataset Utilities ==============

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# ============== LazySupervisedDataset ==============

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    DATASET_PATHS = {
        'all_data': {
            'base': '/share/dataset/UniHand/',
            'frame': '448_chunked_videos',
            'raw_video': 'aligned_raw_videos',
            'raw_motion': 'mano_feature',
            'video_chunk_size': 300,
            'camera_intrinsic': torch.tensor([[[736.6339, 0., 960.],
                                              [0., 736.6339, 540.], 
                                              [0., 0., 1.]]], dtype=torch.float32) # shape (1, 3, 3)
        },
    }
    
    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        codebook_base_list=None,
        motion_unit_length=32,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        motion_code_format='pad',
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        motion_args=None,
        debugging=False,
        split="train",
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer  
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')
        
        self.image_size = image_size
        self.motion_unit_length = motion_unit_length
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.debugging = debugging
        self.split = split

        if isinstance(motion_args, list):
            self.motion_dir = []
            for sub_motion_args in motion_args:
                self.motion_dir.append(sub_motion_args.motion_code_path)
        else:
            self.motion_dir = motion_args.motion_code_path if motion_args is not None else None
        
        if codebook_base_list is None:
            self.codebook_base_list = [0] * len(self.motion_dir) if isinstance(self.motion_dir, list) else [0]
        else:
            self.codebook_base_list = codebook_base_list
            
        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        self._state_dict = {}
        
        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        # if split == 'train':
        datafile_path = meta['annotation']
        # else:
        #     datafile_path = meta['annotation'].replace('.jsonl', f'_test.jsonl')
        
        self.rng = np.random.default_rng(seed=random_seed)
        
        with open(datafile_path, 'r') as f:
            self.raw_data = f.readlines()
            
            # Apply debugging length limitation BEFORE repeat_time processing
            if 'debugging_length' in meta and meta['debugging_length'] > 0:
                original_length = len(self.raw_data)
                target_length = min(meta['debugging_length'], original_length)
                if target_length < original_length:
                    logger.info(f'[Dataset {ds_name}] Limiting from {original_length} to {target_length} samples for debugging')
                    self.raw_data = self.raw_data[:target_length]
            
            if repeat_time < 1:
                entry_num = int(len(self.raw_data) * repeat_time)
                entry_idx = self.rng.choice(
                    len(self.raw_data), size=entry_num, replace=True
                )
                self.raw_data = [self.raw_data[idx] for idx in entry_idx]
            elif repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time
        
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.motion_args = motion_args
        self.motion_code_format = motion_code_format
        self.motion_cache = {}
        
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in tqdm(self.raw_data, desc=f"Loading {self.ds_name}"):
                if self.debugging and len(self.length)>1000:
                    break
                data_item = json.loads(data_item)
           
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

      
    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values) # n,3,448,448
        
        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()
        
        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
                
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret
    
    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def get_motion_text(self, traj_path, index_list, hand_type='both'):
        motion_text_list = []
        if isinstance(index_list[0][0], int):
            index_list = [index_list]
        
        traj_path_list = traj_path if isinstance(traj_path, list) else [traj_path]
        
        motion_file_list = [None, None]
        if hand_type == 'left' or hand_type == 'both':
            left_traj_paths = [traj_path + '_left.json' for traj_path in traj_path_list]
            if any(not os.path.exists(path) for path in left_traj_paths):
                return None
            motion_file_list[0] = [json.load(open(path, 'r')) for path in left_traj_paths]
        if hand_type == 'right' or hand_type == 'both':
            right_traj_paths = [traj_path + '_right.json' for traj_path in traj_path_list]
            if any(not os.path.exists(path) for path in right_traj_paths):
                return None
            motion_file_list[1] = [json.load(open(path, 'r')) for path in right_traj_paths]
        
        for rel_mot_list in index_list:
            if isinstance(rel_mot_list[0], int):
                rel_mot_list = [rel_mot_list]
            
            motion_text = ''
            for rel_mot in rel_mot_list:
                mot_sec = str(rel_mot[0])
                cor_sec = str(rel_mot[1])
                local_motion_text_list = []
                motion_text += MOT_START_TOKEN
                for motion_file in motion_file_list:
                    local_motion_text = ''
                    if motion_file is not None: # and mot_sec in motion_file and cor_sec in motion_file[mot_sec]:
                        for i,sub_motion_file in enumerate(motion_file):
                            assert mot_sec in sub_motion_file and cor_sec in sub_motion_file[mot_sec], \
                                f"Motion file does not contain section {mot_sec} or {cor_sec}"
                            motion_id_list = sub_motion_file[mot_sec][cor_sec]
                            local_motion_text += ''.join([f'<motion_id_{int(motion_id)+self.codebook_base_list[i]}>' for motion_id in motion_id_list])
                            # else:
                            #     local_motion_text = ''
                            #     break

                    if self.motion_code_format == 'pad' and not local_motion_text:
                        local_motion_text = MOT_NOOP * self.motion_unit_length  
                    if local_motion_text:
                        local_motion_text_list.append(local_motion_text)
                
                conj = MOT_END_TOKEN+MOT_START_TOKEN if self.motion_code_format == 'sep' else ''
                motion_text += conj.join(local_motion_text_list)
                motion_text += MOT_END_TOKEN
            motion_text_list.append(motion_text)    
        return motion_text_list
    
    def get_trajectory_frames(self, traj_path, index_list, video_chunk_size=-1):
        frames = []
        if isinstance(index_list, int):
            index_list = [index_list]
        
        video_reader = {} if video_chunk_size > 0 else VideoReader(traj_path+ '.mp4')
        for index in index_list:
            if video_chunk_size > 0:
                chunk_id = index // video_chunk_size
                frame_index = index % video_chunk_size
                if chunk_id not in video_reader:
                    # Load the video chunk
                    if 'egodex' in traj_path:
                        chunk_path = traj_path + f'_{chunk_id:04d}.mp4'
                    else:
                        chunk_path = traj_path + f'_{chunk_id:03d}.mp4'
                    vr = VideoReader(chunk_path)
                    video_reader[chunk_id] = vr
                else:
                    vr = video_reader[chunk_id]
                frames.append(Image.fromarray(vr[frame_index].asnumpy()))
            else:
                # Load the frame directly from the video reader
                frames.append(Image.fromarray(video_reader[index].asnumpy()))
            
        return frames
        
        
    def trajectory_get_item(self, data_item):
        dataset_name = data_item.get('dataset_name')
        traj_key = data_item.get('trajectory')
        hand_type = data_item.get('handtype', None)
        if dataset_name not in self.DATASET_PATHS:
            back_dataset_name =  dataset_name
            dataset_name = 'all'
            
        dataset_dir = self.DATASET_PATHS[dataset_name]['base']
        
        if 'motion' in data_item:
            # If motion data is present, get the motion text
            assert self.motion_dir is not None, "Motion directory is not set"
            if isinstance(self.motion_dir, list):
                traj_motion_path = [os.path.join(motion_dir, traj_key) for motion_dir in self.motion_dir]
            else:
                traj_motion_path = os.path.join(self.motion_dir, traj_key)
            #TD
            # data_motion = data_item['motion']
            # data_frame = data_item['frame'][0]
            # data_motion = [(x[0], data_frame//30) for x in data_motion]
            # motion_text_list = self.get_motion_text(traj_motion_path, data_motion, hand_type)
            motion_text_list = self.get_motion_text(traj_motion_path, data_item['motion'], hand_type)
            if motion_text_list is None:
                return None
            motion_idx = 0
            for conv in data_item['conversations']:
                while '<motion>' in conv['value']:
                    if motion_idx < len(motion_text_list):
                        conv['value'] = conv['value'].replace('<motion>', motion_text_list[motion_idx])
                        motion_idx += 1
                    else:
                        raise ValueError("Not enough motion text for the conversation")
        
        if 'frame' in data_item:
            # If frames data is present, get the trajectory frames
            frame_dir = self.DATASET_PATHS[dataset_name]['frame']
            video_chunk_size = self.DATASET_PATHS[dataset_name].get('video_chunk_size', -1)
            traj_frames_path = os.path.join(dataset_dir, frame_dir, traj_key)
            frames = self.get_trajectory_frames(traj_frames_path, data_item['frame'], video_chunk_size)
            num_image = len(data_item['frame'])
            
        else:
            frames = [Image.new('RGB', (224, 224), (255, 255, 255))]
            num_image = 1
        
        images, num_tiles = [], []
        
        for frame in frames:
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(frame, min_num=self.min_dynamic_patch,
                                        max_num=max(1, self.max_dynamic_patch // num_image),
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(frame)
                num_tiles.append(1)
        
        transform = self.get_transform()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        
        if self.split == 'train':
        # Select the appropriate preprocessing function based on the template name
            preprocess_function = self.get_preprocess_function()
            
            # Preprocess the conversations and generate the return dictionary
            ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                    self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                    use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

            # Calculate position_ids for packed dataset
            position_ids = ret['attention_mask'].long().cumsum(-1) - 1
            position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
            image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
            assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

            # Create the final return dictionary
            ret = dict(
                input_ids=ret['input_ids'][0],
                labels=ret['labels'][0],
                attention_mask=ret['attention_mask'][0],
                position_ids=position_ids[0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
            )
        else:
            assert len(data_item['conversations']) == 2, \
                f'Only support two conversations in test mode, but got {len(data_item["conversations"])}'
            type_id = data_item['id'].split('-')[0]
            ret = dict(
                pixel_values=pixel_values,
                question=data_item['conversations'][0]['value'],
                answer=data_item['conversations'][1]['value'],
                frame_idx=data_item['frame'][0],
                video_path=os.path.join(dataset_dir, self.DATASET_PATHS[dataset_name]['raw_video'], traj_key + '.mp4'),
                camera_intrinsic=self.DATASET_PATHS[dataset_name]['camera_intrinsic'],
                motion_idx_list=data_item['motion'][1] if type_id == '08' or type_id == '07' or type_id =='09' else data_item['motion'],
                hand_type=hand_type,
                motion_path=os.path.join(dataset_dir, self.DATASET_PATHS[dataset_name]['raw_motion'], traj_key),
                id=data_item['id'],
                num_patches=num_patches
            )
                   
        return ret
            
        
    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)
        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                if "motion" in data_item or "frame" in data_item:
                    ret = self.trajectory_get_item(data_item)      
                    if ret is None:
                        i = random.randint(0, len(self.raw_data) - 1)
                        continue
                elif 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                logger.debug(f"Error loading data: {e}, dataset: {self.ds_name}")
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        logger.warning(f'Failed to load image: {images}, dataset: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        logger.warning(f'Failed to load image: {data_path}, dataset: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    logger.warning(f'Failed to load video: {data_path}, dataset: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


# ============== PackedDataset ==============

class PackedDataset(IterableDataset):
    """InternVL PackedDataset for efficient batch training with variable-length sequences."""
    
    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 32768,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        self.dataset_type = [d.dataset_type for d in self.datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]

        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            'sample_info': {d.ds_name:0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f'deduplicated ds_name: {ds_name_list}'

        for ds in self.datasets:
            if ds.max_num_images > self.num_images_expected:
                logger.warning(f'{ds.max_num_images=} of {ds.ds_name} is larger than {self.num_images_expected=}')
                ds.max_num_images = num_images_expected

            if ds.max_tokens > self.max_packed_tokens:
                logger.warning(f'{ds.max_tokens=} of {ds.ds_name} is larger than {self.max_packed_tokens=}')
                ds.max_tokens = self.max_packed_tokens

            self._state_dict[ds.ds_name] = {}

        if get_rank() == 0:
            logger.info(
                f'Loaded dataset to pack: {ds_name_list}, '
                f'{self.num_images_expected=}, {self.max_packed_tokens=}, '
                f'{self.replacement=}, {self.allow_overflow=}',
            )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f'{ds.ds_name:<25}: {ds_w*100:.2f}%')
            temp = '\n'.join(temp)
            logger.info(
                f'Sampling prob for each dataset:\n{temp}'
            )

        if self.allow_empty_data:
            logger.warning('allow_empty_data is enabled, note that empty data may be generated!')

    def load_state_dict(self, state_dict, custom_infos=None):

        self.worker_custom_infos = custom_infos

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f'{ds.ds_name=} is resumed.')
            else:
                logger.warning(f'{ds.ds_name=} is not resumed.')

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                break  # Exit loop if successful
            except StopIteration:
                if self.replacement:
                    # logger.info(f'[Worker id {self.worker_id}] Dataset {self.datasets[current_dataset_idx].ds_name} is exhausted, restart it.')
                    try:
                        self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                        break
                    except:
                        # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)

                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)

                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except:
                logger.error('Unexpected error!')
                if len(self.datasets) == 0:
                    raise StopIteration
                current_dataset_idx = np.random.choice(len(self.datasets))

        current_ds_name = self.datasets[current_dataset_idx].ds_name
        current_sample['type_ids'] = torch.zeros_like(current_sample['input_ids']) + current_dataset_idx

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop('meta_info', {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        # NOTE: use `bisect` to search might be faster

        find = False
        find_idx = -1
        num_images_current = new_sample['pixel_values'].size(0)
        for buffer_idx, buffer in enumerate(buffer_list):
            num_images_buffer = buffer['pixel_values'].size(0)
            if num_images_buffer + num_images_current <= self.num_images_expected:
                num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)

                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        return None

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()

        assert buffer.keys() == new_sample.keys()
        for k in buffer:
            buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer

    @staticmethod
    def check_valid(sample_to_check, min_active_tokens_ratio=1/256):
        num_ignore_tokens = (sample_to_check['labels'] == IGNORE_TOKEN_ID).sum()
        num_tokens = sample_to_check['labels'].numel()
        return (1 - num_ignore_tokens / num_tokens) > min_active_tokens_ratio

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id):
        if buffer['input_ids'].size(0) <= max_tokens:
            return [buffer]

        def _image_is_splitted(input_ids, cut_idx):
            is_image_start = input_ids[cut_idx].item() == img_start_token_id
            is_image_token = input_ids[cut_idx].item() == img_token_id
            is_image_end = input_ids[cut_idx].item() == img_end_token_id
            return is_image_start or is_image_token or is_image_end

        def _split(sample_to_split, left_idx, right_idx, left_img_idx, right_img_idx):
            assert (right_idx is None) == (right_img_idx is None)

            left_sample = {}
            right_sample = {} if right_idx is not None else None
            for k in sample_to_split:
                if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index', 'type_ids']:
                    left_sample[k] = sample_to_split[k][:left_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_idx:]
                elif k in ['pixel_values', 'image_flags']:
                    left_sample[k] = sample_to_split[k][:left_img_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_img_idx:]
                else:
                    raise NotImplementedError(f'find unsupported keys: {k} from {sample_to_split.keys()}')
            return left_sample, right_sample

        splitted_buffer = []
        while buffer['input_ids'].size(0) > max_tokens:
            img_start_idx_list = (buffer['input_ids'] == img_start_token_id).nonzero().squeeze(1).tolist()
            img_end_idx_list = (buffer['input_ids'] == img_end_token_id).nonzero().squeeze(1).tolist()
            assert len(img_start_idx_list) == len(img_end_idx_list)

            if _image_is_splitted(buffer['input_ids'], max_tokens):
                cut_idx = bisect.bisect_left(img_start_idx_list, max_tokens)
                if buffer['input_ids'][max_tokens] == img_start_token_id:
                    assert max_tokens == img_start_idx_list[cut_idx]
                    cut_left_idx = img_start_idx_list[cut_idx]
                    cut_left_img_idx = cut_idx
                else:
                    cut_left_idx = img_start_idx_list[cut_idx - 1]
                    cut_left_img_idx = cut_idx - 1
                cut_right_idx = cut_left_idx
                cut_right_img_idx = cut_left_img_idx
            else:
                cut_img_idx = bisect.bisect(img_start_idx_list, max_tokens)
                if cut_img_idx < len(img_start_idx_list):
                    cut_right_idx = img_start_idx_list[cut_img_idx]
                    cut_right_img_idx = cut_img_idx
                else:
                    cut_right_idx = None
                    cut_right_img_idx = None

                cut_left_idx = max_tokens
                cut_left_img_idx = cut_right_img_idx if cut_right_img_idx is not None else buffer['pixel_values'].size(0)

            left, right = _split(
                sample_to_split=buffer,
                left_idx=cut_left_idx,
                left_img_idx=cut_left_img_idx,
                right_idx=cut_right_idx,
                right_img_idx=cut_right_img_idx,
            )

            assert (left['input_ids'] == img_end_token_id).sum() == (left['input_ids'] == img_start_token_id).sum() == left['pixel_values'].size(0)
            if right is not None:
                assert (right['input_ids'] == img_end_token_id).sum() == (right['input_ids'] == img_start_token_id).sum() == right['pixel_values'].size(0)

            if left['pixel_values'].size(0) >= 1 and PackedDataset.check_valid(left):
                splitted_buffer.append(left)

            if right is None or right['pixel_values'].size(0) == 0:
                break

            buffer = right
            if buffer['input_ids'].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                splitted_buffer.append(buffer)
                break

        logger.debug(
            f'split a sample into {len(splitted_buffer)} samples, '
            f'current max_tokens={max_tokens}'
        )
        return splitted_buffer

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        # NOTE: in-place operation

        splitted_buffer = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
        )

        for each_buffer in splitted_buffer:
            if each_buffer['pixel_values'].size(0) > self.num_images_expected:
                logger.error(
                    f"Find a sample with {each_buffer['pixel_values'].size(0)} images, "
                    f'which exceeds {self.num_images_expected}'
                )
                continue

            if each_buffer['input_ids'].size(0) >= self.max_packed_tokens:
                assert each_buffer['input_ids'].size(0) == self.max_packed_tokens
                buffer_max_len_list.append(each_buffer)
                continue

            find_idx = len(buffer_list)
            num_images_new_sample = each_buffer['pixel_values'].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]['pixel_values'].size(0) < num_images_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        for i in range(1, len(buffer_list)):
            assert buffer_list[i-1]['pixel_values'].size(0) >= buffer_list[i]['pixel_values'].size(0)

        return buffer_list, buffer_max_len_list

    def pad_buffer(self, buffer):
        if buffer['pixel_values'].size(0) == self.num_images_expected:
            return buffer

        num_pad_images = self.num_images_expected - buffer['pixel_values'].size(0)
        pad_images = torch.stack([
            torch.zeros_like(buffer['pixel_values'][0])
            for _ in range(num_pad_images)
        ])
        pad_image_flags = torch.tensor([0] * num_pad_images, dtype=torch.long)

        buffer['pixel_values'] = torch.cat([buffer['pixel_values'], pad_images])
        buffer['image_flags'] = torch.cat([buffer['image_flags'], pad_image_flags])

        return buffer

    def postprocess_buffer(self, buffer, custom_infos=None):
        buffer['worker_state_key'] = self.worker_state_key
        buffer['worker_state_dict'] = self._state_dict
        if custom_infos is not None:
            buffer['custom_infos'] = {self.worker_state_key: copy.deepcopy(custom_infos)}
        return buffer

    def print_log(self, iter_idx, buffer_list):
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(
                f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}"
            )

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f'Begin to iter, {len(buffer_list)=}')

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f'work_state_{self.worker_id}'
        self.datasets = [d for d in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            # if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
            ds.worker_id = worker_id
            ds.worker_state_key = f'work_state_{self.worker_id}'
            ds.num_workers = num_workers
            if self._should_log() and worker_id == 0:
                logger.info(f'set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}')

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if 'buffer_list' in custom_infos and isinstance(custom_infos['buffer_list'], list):
                buffer_list = custom_infos['buffer_list']
                if self._should_log() and worker_id == 0:
                    logger.info(f'[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}')
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(
            f'{self.__class__.__name__} Rank {self.data_rank} '
            f'Worker {worker_id} begin to load data'
        )

        while True:
            self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]
            current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)

            try:
                current_sample = self.next_data(current_dataset_idx)
            except:
                logger.info(f'All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})')
                while len(buffer_list) > 0:
                    if self.strict_mode:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(0)))
                    else:
                        yield self.postprocess_buffer(buffer_list.pop(0))
                logger.info(f'buffer_list is empty! ({len(buffer_list)=})')
                return

            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer)

            while len(buffer_max_len_list) > 0:
                if buffer_max_len_list[0]['pixel_values'].size(0) != self.max_packed_tokens:
                    logger.debug(
                        f'num tokens of a buffer exceed {self.max_packed_tokens=}, '
                        f"yield a sample with {buffer_max_len_list[0]['pixel_values'].size(0)} images"
                    )
                if self.strict_mode and buffer_max_len_list[0]['pixel_values'].size(0) != self.num_images_expected:
                    # buffer_max_len_list.pop(0)
                    yield self.postprocess_buffer(self.pad_buffer(buffer_max_len_list.pop(0)), {'buffer_list': buffer_list})
                else:
                    yield self.postprocess_buffer(buffer_max_len_list.pop(0), {'buffer_list': buffer_list})

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'].size(0) > self.num_images_expected:
                logger.error(
                    f"num images of a buffer ({buffer_list[0]['pixel_values'].size(0)}) "
                    f'is larger than num_images_expected({self.num_images_expected})'
                )
                buffer_list.pop(0)

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'].size(0) == self.num_images_expected:
                if self.debug_mode:
                    debug_data = self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list})
                    while True:
                        yield debug_data.copy()

                yield self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list})

            while len(buffer_list) > self.max_buffer_size:
                logger.debug(
                    f'Failed to pack data to exactly {self.num_images_expected} images, '
                    f"yield a data sample with {buffer_list[0]['pixel_values'].size(0)} images."
                )
                if self.strict_mode:
                    yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(0)), {'buffer_list': buffer_list})
                else:
                    yield self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list})

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1

    @staticmethod
    def get_cu_seqlens_and_indexes(
        data_index: torch.LongTensor,  # (seq_len,)
        input_ids: torch.LongTensor,   # (seq_len,)
        labels: torch.LongTensor,   # (seq_len,)
        len2weight: callable,
    ):
        indexes = []
        cu_seqlens = [0]
        loss_weight = []

        start = data_index.min()
        end = data_index.max() + 1
        for i in range(start, end):
            num_tokens = (data_index == i).sum().item()
            indexes.extend(list(range(num_tokens)))
            cu_seqlens.append(cu_seqlens[-1] + num_tokens)
            assert num_tokens > 0

            curr_data_index = data_index[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
            assert (curr_data_index == i).all(), data_index

            curr_labels = labels[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
            num_effective_tokens = (curr_labels != IGNORE_TOKEN_ID).sum().item()
            loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)

        assert len(indexes) == data_index.size(0), f'{len(indexes)=}, {data_index.size(0)=}'

        loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
        return cu_seqlens, indexes, loss_weight


# ============== Packed Dataset Collator ==============

WARNING_CNT = defaultdict(int)


def packed_collate_fn(
    features,
    data_collator,
    len2weight: callable,
    max_item_length: int,
    micro_num: int = 1,
    loss_reduction_all_gather: bool = False,
    pad_id: int = 0,
):
    """Collate function for PackedDataset."""
    if not isinstance(features, list):
        features = [features]

    if len(features) > micro_num:
        raise NotImplementedError(f'{len(features)=} > {micro_num=}')

    if len(features) < micro_num and WARNING_CNT['micro_num_warning'] < 5:
        logger.warning(
            f'{len(features)=} > {micro_num=}, '
            f'the features will be padded to satisfy micro_num requirement'
        )
        WARNING_CNT['micro_num_warning'] += 1

    # ensure that the len(features) is equal to the required micro_num
    num_features = len(features)
    while len(features) < micro_num:
        features.append(copy.deepcopy(features[0]))
        features[-1]['labels'] = torch.full_like(features[-1]['labels'], IGNORE_TOKEN_ID)

    indexes = []
    cu_seqlens = []
    cu_num_images_list = [0]

    worker_state_key_list = []
    worker_state_dict_list = []
    worker_state_custom_infos_list = []

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]

    num_samples = 0
    num_padding_tokens = 0
    for feat_idx, feat in enumerate(features):
        data_index = feat.pop('data_index')
        curr_cu_seqlens, curr_indexes, curr_loss_weight = PackedDataset.get_cu_seqlens_and_indexes(
            data_index=data_index,
            input_ids=feat['input_ids'],
            labels=feat['labels'],
            len2weight=len2weight,
        )

        feat['loss_weight'] = curr_loss_weight

        if feat_idx < num_features:
            num_samples += len(curr_cu_seqlens) - 1

        if curr_cu_seqlens[-1] < max_item_length:
            curr_cu_seqlens.append(max_item_length)
            curr_indexes.extend(list(range(max_item_length - curr_cu_seqlens[-2])))

        indexes.append(torch.tensor(curr_indexes, dtype=torch.long))
        cu_seqlens.append(torch.tensor(curr_cu_seqlens, dtype=torch.int32))

        worker_state_key_list.append(feat.pop('worker_state_key'))
        worker_state_dict_list.append(feat.pop('worker_state_dict'))
        worker_state_custom_infos_list.append(feat.pop('custom_infos', None))

        num_padding_tokens += (max_item_length - feat['input_ids'].size(0))
        cu_num_images_list.append(cu_num_images_list[-1] + feat['pixel_values'].size(0))

    batch = data_collator(features=features, max_item_length=max_item_length, pad_id=pad_id)
    # convert it to list in case it is converted into bf16
    batch['loss_weight'] = torch.where(batch['labels'] == IGNORE_TOKEN_ID, 0, batch['loss_weight']).tolist()
    batch['attention_mask'] = torch.stack(cu_seqlens)
    batch['loss_reduction_all_gather'] = loss_reduction_all_gather
    batch['statistics'] = torch.tensor(
        [
            num_samples,
            num_padding_tokens,
            batch['image_flags'].numel() - batch['image_flags'].sum().item(),
        ],
        dtype=torch.long,
    )
    batch.pop('type_ids')
    return batch


# Export all classes and functions
__all__ = [
    'LazySupervisedDataset',
    'PackedDataset', 
    'packed_collate_fn',
]