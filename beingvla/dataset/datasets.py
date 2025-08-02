# BeingVLA Dataset Classes
# Self-contained dataset implementations extracted from InternVL for training
# Originally from InternVL Copyright (c) 2024 OpenGVLab

import io
import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from decord import VideoReader
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from transformers.trainer_pt_utils import LabelSmoother

from ..utils.conversation import get_conv_template
from ..utils.constants import (
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
    MOT_START_TOKEN, MOT_END_TOKEN, MOT_NOOP
)

# Petrel client removed - using local storage only

# Constants for image normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class WeightedConcatDataset(ConcatDataset):
    """Dataset that concatenates multiple datasets with weighted sampling."""
    
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    """Load PIL image from bytes string."""
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


# TCSLoader removed - using PIL directly for local image loading


def expand2square(pil_img, background_color):
    """Expand image to square by padding."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    """Create a function that simulates JPEG compression degradation."""
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)
            img_jpeg = Image.open(output).copy()
        return img_jpeg
    return jpeg_degrade


# Pre-create all JPEG compression functions for efficiency
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


# Define pickleable transform functions and classes
def convert_to_rgb(img):
    """Convert image to RGB if needed."""
    return img.convert('RGB') if img.mode != 'RGB' else img


def expand2square_imagenet(img):
    """Expand to square with ImageNet mean background."""
    return expand2square(img, tuple(int(x * 255) for x in IMAGENET_MEAN))


def expand2square_clip(img):
    """Expand to square with CLIP mean background."""
    return expand2square(img, tuple(int(x * 255) for x in CLIP_MEAN))


def expand2square_siglip(img):
    """Expand to square with SigLIP mean background."""  
    return expand2square(img, tuple(int(x * 255) for x in SIGLIP_MEAN))


class RandomJPEGCompression:
    """Pickleable random JPEG compression transform."""
    def __init__(self, qualities):
        self.qualities = qualities
    
    def __call__(self, img):
        quality = random.choice(self.qualities)
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)
            img_jpeg = Image.open(output).copy()
        return img_jpeg


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    """Build image transform pipeline."""
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        expand_func = expand2square_imagenet
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
        expand_func = expand2square_clip
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
        expand_func = expand2square_siglip
    else:
        raise NotImplementedError(f"Unknown normalize_type: {normalize_type}")
    
    if is_train:  # use data augmentation
        transform = T.Compose([
            T.Lambda(convert_to_rgb),
            RandomJPEGCompression(qualities),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:
            transform = T.Compose([
                T.Lambda(convert_to_rgb),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(convert_to_rgb),
                T.Lambda(expand_func),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
    
    return transform


# Note: read_frames_* functions would need to be added if video support is needed
# They are omitted here for brevity but can be extracted from the original dataset.py




# Import necessary components for preprocessing
from ..utils.conversation import get_conv_template
from ..utils.constants import (
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
    MOT_END_TOKEN, MOT_START_TOKEN
)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamically preprocess images into multiple patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def preprocess(
        template_name,
        sources,
        tokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
):
    """Generic preprocessing function."""
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


# Placeholder preprocessing functions for different models
# These can be expanded as needed for specific model requirements
def preprocess_mpt(template_name, sources, tokenizer, num_image_token_list, **kwargs):
    """Preprocessing for MPT models."""
    return preprocess(template_name, sources, tokenizer, num_image_token_list, **kwargs)
