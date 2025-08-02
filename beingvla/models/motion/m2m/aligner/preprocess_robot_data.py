import zarr
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image

def preprocess_robot_zarr(
    zarr_path: Path,
    output_dir: Path,
    action_chunk_length: int = 16,
    image_save_dir: str = "images",
    # Default instruction updated to include dynamic placeholder {k}
    instruction: str = "What are the next {k} actions for the dexterous hand?",
    split: str = "train"
):
    """
    Preprocess Lerobot-style Zarr datasets into JSONL format for multimodal model training.

    Args:
        zarr_path (Path): Path to input Zarr dataset.
        output_dir (Path): Directory to save output jsonl files and images.
        action_chunk_length (int): Length of action sequence to predict.
        image_save_dir (str): Subdirectory within output_dir to save extracted images.
        instruction (str): Natural language task instruction, can include {k} placeholder.
        split (str): Data split to process ('train' or 'val').
    """
    print(f"Starting to preprocess Zarr dataset: {zarr_path}")
    print(f"Output will be saved to: {output_dir}")

    # 1. Setup paths
    output_dir.mkdir(parents=True, exist_ok=True)
    images_path = output_dir / image_save_dir
    images_path.mkdir(exist_ok=True)
    jsonl_output_path = output_dir / f"{zarr_path.stem}_{split}.jsonl"

    # 2. Load Zarr dataset and save statistics
    try:
        root = zarr.open(str(zarr_path), mode='r')
    except Exception as e:
        print(f"Error opening Zarr dataset: {e}")
        return

    # --- Save stats information ---
    if 'stats' in root.attrs:
        stats_data = root.attrs['stats']
        stats_output_path = output_dir / "stats.json"
        with open(stats_output_path, 'w') as f_stats:
            # Use asdict() in case stats is a special Zarr attribute type
            json.dump(stats_data.asdict() if hasattr(stats_data, 'asdict') else stats_data, f_stats, indent=4)
        print(f"Statistics saved to: {stats_output_path}")
    else:
        print("Warning: 'stats' not found in Zarr attributes.")
    # --------------------------

    # 3. Get episode boundaries from metadata
    episodes_meta = root.attrs['episodes_meta']
    total_samples_written = 0
    
    # --- Format general instruction ---
    # Replace {k} in instruction with actual action_chunk_length
    final_instruction = instruction.format(k=action_chunk_length)
    # --------------------------

    # 4. Open output file for writing
    with open(jsonl_output_path, 'w') as f:
        # Iterate through each episode
        for episode_meta in tqdm(episodes_meta, desc="Processing episodes"):
            episode_index = episode_meta['episode_index']
            episode_key = f'episode_{episode_index:06d}'
            episode_group = root[episode_key]
            episode_length = episode_meta['length']

            # --- Get episode-specific task description ---
            try:
                # Assume each episode has at least one task description in tasks list
                task_description = episode_meta['tasks'][0]
            except (KeyError, IndexError):
                task_description = "No specific task description provided."
                print(f"Warning: Episode {episode_index} missing task description, using placeholder.")
            # ----------------------------------------

            # Iterate through each possible starting frame in episode
            # Stop if remaining frames insufficient for an action chunk
            for frame_idx in range(episode_length - action_chunk_length):
                # --- A. Extract data from Zarr ---

                # Image: Get image from current frame
                image_data = episode_group['observation.images.ego_view'][frame_idx]

                # Proprioception: Get state from current frame
                proprioception_data = episode_group['observation.state'][frame_idx].tolist()

                # Action chunk: Get *future* actions starting from current frame
                action_chunk_data = episode_group['action'][frame_idx : frame_idx + action_chunk_length].tolist()

                # --- B. Save image and prepare path ---
                # Create unique ID for this sample
                sample_id = f"ep{episode_index:06d}_frame{frame_idx:06d}"
                
                # Save image as PNG file
                image_filename = f"{sample_id}.png"
                image_save_path = images_path / image_filename
                Image.fromarray(image_data).save(image_save_path)
                
                # Path stored in jsonl should be relative to output_dir
                relative_image_path = os.path.join(image_save_dir, image_filename)

                # --- C. Build JSONL entry ---
                # Define placeholders
                image_placeholder = "<image>"
                proprio_placeholder = "<prop_context>"
                action_placeholders = "".join([f"<ACTION_CHUNK_{i}>" for i in range(action_chunk_length)])

                # # --- Build prompt containing task description and general instruction ---
                # human_turn_value = (
                #     f"Task: {task_description}\n"
                #     f"{image_placeholder}\n"
                #     f"{proprio_placeholder}\n"
                #     f"{final_instruction}"
                # )
                # human_turn_value = f"In <image>, the right hand starts its motion with <prop_context>.\n The instruction is '{task_description}'. Plan the right hand's motion in {action_chunk_length} steps."
                human_turn_value = f"<image>\n According to the instruction '{task_description}', what's the motion plan for the right hand in {action_chunk_length} steps? <PROP_CONTEXT>"

                human_turn = {
                    "from": "human",
                    "value": human_turn_value
                }
                # ---------------------------------------------

                # GPT turn: Provide action placeholders
                gpt_turn = {
                    "from": "gpt",
                    "value": action_placeholders
                    # "value": ""
                }

                # Final JSON object for this sample
                json_entry = {
                    # Metadata aligned with existing format
                    "id": sample_id,
                    "dataset_name": zarr_path.stem,  # Use dataset folder name
                    "image": relative_image_path,
                    
                    # Core data for model
                    "conversations": [human_turn, gpt_turn],
                    "proprioception": proprioception_data,
                    "action_chunk": action_chunk_data,
                }

                # Write JSON object as new line to file
                f.write(json.dumps(json_entry) + '\n')
                total_samples_written += 1

    print("-" * 50)
    print(f"Preprocessing complete!")
    print(f"Total samples written: {total_samples_written}")
    print(f"JSONL file saved at: {jsonl_output_path}")
    print(f"Images saved at: {images_path}")
    print("-" * 50)


if __name__ == '__main__':
    # --- How to use this script ---
    
    # Define your Lerobot Zarr dataset path
    input_zarr_path = Path("/share/dataset/dexhand/pick_duck_cup_lerobot") 

    # Define where to save preprocessed data
    # Script will automatically create directory if it doesn't exist
    preprocessed_output_dir = Path("/share/dataset/dexhand/sft_data/pick_duck_cup_lerobot")

    # Run preprocessing
    preprocess_robot_zarr(
        zarr_path=input_zarr_path,
        output_dir=preprocessed_output_dir,
        action_chunk_length=16,  # Customize action chunk length
        # No need to pass instruction parameter, it will use the new default value.
        # Of course, you can still override it like below:
        # instruction="Perform the next {k} actions to complete the task."
    )