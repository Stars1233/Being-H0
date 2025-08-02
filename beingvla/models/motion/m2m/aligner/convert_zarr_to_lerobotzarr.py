import zarr
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
# from decord import VideoReader, cpu

def build_attr_template(fps, state_dim, action_dim):
    ATTRS_TEMPLATE = {
        'info': {
            'total_episodes': 0,
            'total_frames': 0,
            'fps': fps,
            'robot_type': 'MyDexHand',
            'codebase_version': 'v2.0',
            'splits': {
                'train': '0:100'  # Will be updated later
            },
            'features': {
                'action': {
                    'shape': [action_dim], 
                    'dtype': 'float64',
                    'names': [f'motor_{i}' for i in range(action_dim)]
                },
                'observation.state': {
                    'shape': [state_dim], 
                    'dtype': 'float64',
                    'names': [f'motor_{i}' for i in range(state_dim)]
                },
                'observation.images.ego_view': {
                    'shape': [256, 256, 3], 
                    'dtype': 'video',  # uint8 in Zarr, can be marked as 'video' in info
                    'names': ['height', 'width', 'channel'],
                    "video_info": {
                        "video.fps": fps,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                'episode_index': {'shape': [1], 'dtype': 'int64'},
                'index': {'shape': [1], 'dtype': 'int64'},
                'next.done': {'shape': [1], 'dtype': 'bool'},
                'next.reward': {'shape': [1], 'dtype': 'float64'},
                'timestamp': {'shape': [1], 'dtype': 'float64'},
                'annotation.human.action.task_description': {'shape': [1], 'dtype': 'int64'},
                'annotation.human.validity': {'shape': [1], 'dtype': 'int64'},
                'task_index': {'shape': [1], 'dtype': 'int64'},
            }
        },
        'modality': {
            # This accurately describes the data structure we're creating.
            # Source data only has right_arm and right_hand.
            'action': {
                'right_arm_qpos': {'start': 0, 'end': 7},
                'right_hand_qpos': {'start': 7, 'end': 13}
            },
            'state': {
                'right_arm_qpos': {'start': 0, 'end': 7},
                'right_hand_qpos': {'start': 7, 'end': 13}
            },
            'video': {
                'ego_view': {'original_key': 'observation.images.ego_view'}
            },
            'annotation': {
                'human.action.task_description': {},
                'human.validity': {}
            }
        },
        'tasks': {
            # Will be filled with placeholder tasks later
        },
        'episodes_meta': [
            # Will be filled in the main loop
        ],
        'stats': {
            # Will be filled in the statistics calculation section
        }
    }

    return ATTRS_TEMPLATE

def convert_zarr_to_lerobotzarr(original_path: Path, new_zarr_path: Path, fps=30):
    """
    Convert original Zarr dataset format to Lerobot Zarr format.
    """
    print(f"Starting conversion: {original_path} -> {new_zarr_path}")

    timestamp_step = 1.0 / fps
    timestamp_start = timestamp_step

    original_root = zarr.open(str(original_path), mode='r')
    # print(original_root.attrs)
    # print(original_root)
    # breakpoint()

    # 1. Create Zarr root directory
    if new_zarr_path.exists():
        print(f"Warning: Zarr path {new_zarr_path} already exists, will be overwritten.")
    root = zarr.open(str(new_zarr_path), mode='w')

    episode_ends = original_root['meta/episode_ends'][:]
    print(f"Found {len(episode_ends)} episodes.")

    original_data = original_root['data']

    # 2. Process each episode in loop
    start_idx = 0
    episodes_meta_list = []
    for i, end_idx in enumerate(tqdm(episode_ends, desc="Processing episodes")):
        
        episode_len = end_idx - start_idx
        
        # Create new episode group
        episode_group = root.create_group(f'episode_{i:06d}')
        
        # --- 3. Process and convert data ---
        
        # 3.1 observation.images.ego_view (direct slicing)
        # This is a large data block, we slice directly from source and write to avoid creating huge copies in memory
        rgb_data = original_data['camera_0.rgb'][start_idx:end_idx]
        episode_group.create_dataset(
            'observation.images.ego_view',
            data=rgb_data,
            chunks=(16, *rgb_data.shape[1:]),  # Set appropriate chunking for image data
            dtype='uint8',
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2),  # 2 means Bit shuffle
        )

        # print(original_data)
        # breakpoint()
        
        # 3.2 observation.state (concatenate, pad, type conversion)
        arm_qpos = original_data['right_arm_qpos'][start_idx:end_idx]
        hand_qpos = original_data['right_hand_qpos'][start_idx:end_idx]
        
        # Concatenate to 13 dimensions
        state = np.concatenate([arm_qpos, hand_qpos], axis=1)
        
        episode_group.create_dataset(
            'observation.state',
            data=state,
            chunks=(64, *state.shape[1:]),
            dtype='float64',
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2) # 2 表示 Bit shuffle
        )

        # 3.3 action (pad, type conversion)
        action = original_data['action'][start_idx:end_idx]

        episode_group.create_dataset(
            'action',
            data=action,
            chunks=(64, *action.shape[1:]),
            dtype='float64',
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2) # 2 表示 Bit shuffle
        )

        # --- 4. Generate new fields ---
        
        # 4.1 next.done
        done_array = np.zeros(episode_len, dtype=bool)
        done_array[-1] = True  # Last step of episode is True
        episode_group.create_dataset('next.done', data=done_array, dtype='bool')
        
        # 4.2 episode_index
        episode_index_array = np.full(episode_len, i, dtype=np.int64)
        episode_group.create_dataset('episode_index', data=episode_index_array, dtype='int64')

        # 4.3 index (global index)
        index_array = np.arange(start_idx, end_idx, dtype=np.int64)
        episode_group.create_dataset('index', data=index_array, dtype='int64')

        # 4.4 Other placeholder fields
        task_description_index_array = np.full(episode_len, i, dtype=np.int64)
        episode_group.create_dataset('annotation.human.action.task_description', data=task_description_index_array)
        episode_group.create_dataset('annotation.human.validity', data=np.ones(episode_len, dtype=np.int64))
        episode_group.create_dataset('next.reward', data=np.zeros(episode_len, dtype=np.float64))

        task_index_array = np.full(episode_len, i, dtype=np.int64)
        episode_group.create_dataset('task_index', data=task_index_array)

        timestamp_array = (np.arange(episode_len) * timestamp_step) + timestamp_start
        episode_group.create_dataset('timestamp', data=timestamp_array, dtype='float64')

        # --- Collect information for Attributes ---
        episodes_meta_list.append({'episode_index': i, 'length': episode_len, 'tasks': ['Put the little yellow duck into the cup.']})
        # Update start index for next episode
        start_idx = end_idx

    # ==============================================================================
    # --- 3. Fill and write attributes (except stats) ---
    # ==============================================================================
    print("\nWriting attributes to the root group...")
    
    # Deep copy template to avoid issues when running function multiple times
    final_attrs = build_attr_template(fps, state.shape[1], action.shape[1])

    # Fill episodes_meta
    final_attrs['episodes_meta'] = episodes_meta_list
    
    # Fill info
    final_attrs['info']['total_episodes'] = len(episode_ends)
    final_attrs['info']['total_frames'] = int(episode_ends[-1])
    # Assume all data is used for training
    final_attrs['info']['splits']['train'] = f'0:100'
    
    # Fill tasks (using placeholder)
    final_attrs['tasks'] = {'0': {'task': 'Put the little yellow duck into the cup.', 'task_index': 0}}

    # Use .put() method to write entire dictionary to Zarr attributes
    root.attrs.put(final_attrs)
    print("Attributes 'episodes_meta', 'modality', 'tasks', 'info' written.")

    # ==============================================================================
    # --- 4. Calculate and write statistics (stats) ---
    # ==============================================================================
    print("\nCalculating statistics... This may take a while.")
    stats_data = {}
    keys_for_stats = list(final_attrs['info']['features'].keys())
    # Remove image data as it's too large and statistics are not usually calculated for it
    keys_for_stats.remove('observation.images.ego_view')

    root_read = zarr.open(new_zarr_path, mode='r')

    for key in tqdm(keys_for_stats, desc="Calculating stats"):
        all_data = np.concatenate([ep[key][:] for _, ep in root_read.groups()])
        axis = 0 if all_data.ndim > 1 else None

        print(key)

        if all_data.dtype == bool:
            # For boolean types, quantiles are just min/max
            stats_data[key] = {
                'min': np.min(all_data, axis=axis).tolist(),
                'max': np.max(all_data, axis=axis).tolist(),
                'mean': np.mean(all_data, axis=axis).tolist(),  # Calculate as 0/1
                'std': np.std(all_data, axis=axis).tolist(),    # Calculate as 0/1
                'q01': np.min(all_data, axis=axis).tolist(),    # Use min instead
                'q99': np.min(all_data, axis=axis).tolist(),    # Use max instead
            }
        else:
            stats_data[key] = {
                'min': np.min(all_data, axis=axis).tolist(),
                'max': np.max(all_data, axis=axis).tolist(),
                'mean': np.mean(all_data, axis=axis).tolist(),
                'std': np.std(all_data, axis=axis).tolist(),
                'q01': np.quantile(all_data, 0.01, axis=axis).tolist(),
                'q99': np.quantile(all_data, 0.99, axis=axis).tolist(),
            }

        print(key, stats_data[key])
        print('-'*50)

    # Update 'stats' attribute in Zarr file
    root.attrs['stats'] = stats_data
    print("Attribute 'stats' calculated and written.")
    print("\nConversion completed successfully!")

    # breakpoint()
    # ==============================================================================
    # --- 5. Final validation ---
    # ==============================================================================
    print("\nVerifying the structure and attributes of the new Zarr file:")
    final_root = zarr.open(new_zarr_path, mode='r')
    print("--- Tree ---")
    print(final_root.tree())
    print("\n--- Attributes (JSON format) ---")
    print(json.dumps(final_root.attrs.asdict(), indent=2))

    print("\nConversion completed successfully!")

if __name__ == '__main__':
    # Usage
    original_dataset_path = Path("/share/dataset/dexhand/pick_duck_cup")
    zarr_dataset_path = Path("/share/dataset/dexhand/pick_duck_cup_lerobot")
    
    convert_zarr_to_lerobotzarr(original_dataset_path, zarr_dataset_path, fps=20)