import os
import sys
import cv2
import json
import torch
# import transforms
import logging
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from pprint import pprint
from dataclasses import asdict, fields
from decord import VideoReader
from decord import cpu
import copy


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            pprint(f"Rank {dist.get_rank()}: ", *args)
    else:
        pprint(*args)


def rank0_logger(metrics, logger_dir):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            with open(logger_dir+"/result.log", "a") as f:
                f.writelines(str(metrics)+"\n")
    else:
        with open(logger_dir+"/result.log", "a") as f:
            f.writelines(str(metrics)+"\n")
            

class rank0tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        if dist.is_initialized() and dist.get_rank() != 0:
            return  
        super().display(msg=msg, pos=pos)


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def load_pretrained_args(train_args, motion_args):
    with open(f"{train_args.output_dir}/motion_args.json", "r") as f:
        loaded_args_dict = json.load(f)
    for field in fields(motion_args):
        if field.name in ['tmr_model_path', 'val_split_file']:
            continue
        if field.name in loaded_args_dict:
            setattr(motion_args, field.name, loaded_args_dict[field.name])
    return motion_args

def load_resume_args(motion_args):
    load_path = motion_args.motion_code_path if motion_args.motion_resume_pth is None else motion_args.motion_resume_pth
    if '+' in load_path:
        load_path_list = load_path.split('+')
    else:
        load_path_list = [load_path]
    
    motion_args_list = []
    for sub_path in load_path_list:
        motion_args_path = os.path.join(sub_path, "motion_args.json")
        sub_motion_args = copy.deepcopy(motion_args)
        if os.path.exists(motion_args_path):
            with open(motion_args_path, "r") as f:
                loaded_args_dict = json.load(f)
            for field in fields(sub_motion_args):
                if field.name in ['tmr_model_path', 'val_split_file', 'motion_resume_pth', 'motion_code_path']:
                    continue
                if field.name in loaded_args_dict:
                    setattr(sub_motion_args, field.name, loaded_args_dict[field.name])
        if motion_args.motion_resume_pth is None:
            setattr(sub_motion_args, 'motion_code_path', sub_path)
        motion_args_list.append(sub_motion_args)
    
    return motion_args_list

def write_images_to_video(traj_name, frames, save_dir, in_mode="rgb"):
    if frames.shape[1]!=3:
        frames = frames.permute(0,3,1,2)
    short_side = min(frames.shape[2], frames.shape[3])
    if short_side>448:
        # frames = F.resize(frames, size=448)
        # resize_transform = transforms.Resize(size=(448, 448))  # For grayscale, use size=448
        # frames = resize_transform(frames)
        frames = torch.nn.functional.interpolate(frames, size=(448, 448), mode='bilinear', align_corners=False)
    frames = frames.permute(0,2,3,1).numpy()
    if in_mode == "rgb":
        frames = frames[:,:,:,::-1]
    
    vw = cv2.VideoWriter(f"{save_dir}/{traj_name}", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames.shape[2], frames.shape[1]))  # W, H
    for i in range(frames.shape[0]):
        vw.write(np.uint8(frames[i]))
    
    vw.release()

import imageio

def write_imagelist_to_video(video_path, images, min_size=448, fps=30):
    """
    Args:
        video_path (str): Output video path (e.g. 'output.mp4').
        images (list): List containing image file paths.
        fps (int): Frame rate of the video.
    """
    if not images:
        print("Error: Image list is empty.")
        return

    print(f"Starting to create video using imageio: {video_path}, fps: {fps}")
    
    # Using 'with' statement ensures writer is properly closed
    with imageio.get_writer(video_path, fps=fps) as writer:
        # Iterate through all image paths
        for i, img_path in enumerate(images):
            # Print progress
            print(f"Adding frame {i+1}/{len(images)}: {img_path}", end='\r')
            try:
                # Read image
                img = imageio.imread(img_path)
                # Add image data as a frame
                writer.append_data(img)
            except FileNotFoundError:
                print(f"\nWarning: File not found, skipping: {img_path}")
            except Exception as e:
                print(f"\nWarning: Error occurred while reading or writing image '{img_path}': {e}")

    print(f"\nVideo successfully saved to: {video_path}")

# def write_imagelist_to_video(video_path, images, min_size=448, fps=30):
#     if not os.path.exists(video_path):
#         h, w = cv2.imread(images[0]).shape[:2]
#         if w > h:
#             new_h = min_size
#             new_w = int(w * (min_size / h))
#         else:
#             new_w = min_size
#             new_h = int(h * (min_size / w))
        
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video_writer = cv2.VideoWriter(
#             video_path,
#             fourcc,
#             fps,
#             (new_w, new_h)
#         )
#         for img_path in images:
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"跳过无法读取的图像: {img_path}")
#                 continue
#             resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#             video_writer.write(resized)

#         video_writer.release()


def frame_sampler(frame_len, target_fps, raw_fps):
    raw_idxs = len(range(frame_len))
    idxs = [int(idx*raw_fps/target_fps) for idx in range(raw_idxs)]
    max_idx = int(frame_len*target_fps/raw_fps)
    sampled_idxs = idxs[:max_idx]
    return sampled_idxs, len(sampled_idxs)


def traj_len_update(traj_len, min_traj_len, max_traj_len, avg_traj_len):
    if traj_len > max_traj_len:
        max_traj_len = traj_len
    elif traj_len < min_traj_len:
        min_traj_len = traj_len
    avg_traj_len.append(traj_len)
    return min_traj_len, max_traj_len, avg_traj_len


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_count, video_fps


def video_writer(traj_name, raw_video_path, frame_idxs, out_data_dir):
    vr = VideoReader(raw_video_path, ctx=cpu(0))
    
    cap = cv2.VideoCapture(raw_video_path)
    out_of_bound = np.where(np.asarray(frame_idxs)>=cap.get(cv2.CAP_PROP_FRAME_COUNT))[0]

    if len(out_of_bound)>0:
        frames = vr.get_batch(frame_idxs[:out_of_bound[0]])
        app_frame_num = frames[-1].unsqueeze(0).repeat(len(frame_idxs)-frames.shape[0],1,1,1)
        frames = torch.cat([frames, app_frame_num])
    else:
        frames = vr.get_batch(frame_idxs)

    frames = frames.permute(0,3,1,2)
    
    write_images_to_video(traj_name, frames, f"{out_data_dir}/images")


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox


def mp42imagefiles(video_path, save_rgb_dir, frame_format="%06d.jpg", way='ffmpeg'):
    """
    Save each frame in a sequence to an image file.

    Args:
        video_path (str): Path to the input MP4 video file.
        save_rgb_dir (str): Directory to save the extracted frames.
        frame_format (str): Format for saving frames (e.g., "%06d.jpg").
    """
    if os.path.exists(save_rgb_dir):
        return
    os.makedirs(save_rgb_dir, exist_ok=True)

    if way == 'cv2':
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        # total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_save = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            frame_filename = os.path.join(save_rgb_dir, frame_format % frame_save)
            cv2.imwrite(frame_filename, frame)
            frame_save += 1
        cap.release()
    elif way == 'ffmpeg':
        command = f'ffmpeg -i "{video_path}" -start_number 0 {os.path.join(save_rgb_dir,"%06d.jpg")}'# -vf "rotate=PI:bilinear=0" -q:v 2
        # os.system(command) 
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # quietly
    # print(f"Saved {frame_save} frames to {save_rgb_dir}")

def ari2imagefiles(video_path, save_depth_dir, frame_format="%06d.png", way='ffmpeg'):
    if os.path.exists(save_depth_dir):
        return
    os.makedirs(save_depth_dir, exist_ok=True)
    if way == 'ffmpeg':
        command = f'ffmpeg -i "{video_path}" -start_number 0 {os.path.join(save_depth_dir, frame_format)}'
        # os.system(command) 
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # quietly
    
    # depth_png = cv2.imread("depth_frame.png", cv2.IMREAD_ANYDEPTH)  # Critical for 16-bit