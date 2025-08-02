import torch
import zarr
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import time  # For timing inference

# Import model, tokenizer and data processing tools
from transformers import AutoTokenizer
from ....vla.being_vla_model import BeingVLAModel
from ....vla.config import BeingVLAConfig
from .....dataset.datasets import build_transform  # Ensure import path is correct

# --- Zarr Reader ---
class ZarrDatasetReader:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.root = zarr.open(str(input_path), mode='r')
        
        # Adapt to different Zarr file structures
        # Assume data is in root directory, not under 'data' and 'meta' groups
        if 'data' in self.root and 'meta' in self.root:
            self.data_group = self.root['data']
            self.episodes_meta = self.root['meta'].attrs.get('episodes_meta', [])
        else:
            # Adapt to preprocess_robot_zarr.py output structure
            self.data_group = self.root 
            self.episodes_meta = self.root.attrs.get('episodes_meta', [])

        self.data_keys = list(self.data_group.keys())
        # Adapt key naming conventions
        self.observation_keys = {
            'image': 'observation.images.ego_view',
            'state': 'observation.state'
        }
        self.action_key = 'action'
    
    def get_episode(self, episode_index):
        episode_key = f'episode_{episode_index:06d}'
        if episode_key in self.data_group:
            return self.data_group[episode_key]
        return None

# --- Main Evaluation Script ---

def evaluate_policy(
    model_path: str,
    zarr_path: str,
    task_description: str,
    action_chunk_length: int,
    device: torch.device
):
    """
    Evaluates the policy model on a Zarr dataset and computes the average L1 error.
    """
    print("--- 1. Loading Model and Tokenizer ---")
    
    # Load model and tokenizer
    try:
        model = BeingVLAModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Move model to specified device
    model.to(device)
    model.eval()

    # **IMPORTANT**: Set special token IDs on model instance
    # Ensure these tokens exist in tokenizer vocabulary
    try:
        model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        model.prop_context_token_id = tokenizer.convert_tokens_to_ids('<PROP_CONTEXT>')
        model.action_chunk_token_ids = tokenizer.convert_tokens_to_ids(
            [f'<ACTION_CHUNK_{i}>' for i in range(action_chunk_length)]
        )
        print("Special token IDs set successfully.")
    except Exception as e:
        print(f"Error setting special token IDs. Make sure they are in the tokenizer vocab: {e}")
        return

    print("\n--- 2. Preparing Data Loader and Preprocessing ---")
    
    # Prepare image preprocessor
    force_image_size = model.config.vlm_config.get('force_image_size', 448)
    image_transform = build_transform(
        is_train=False, 
        input_size=force_image_size,
        normalize_type='imagenet'
    )
    
    # Initialize Zarr dataset reader
    try:
        dataset_reader = ZarrDatasetReader(input_path=zarr_path)
    except Exception as e:
        print(f"Error opening Zarr dataset at {zarr_path}: {e}")
        return

    # --- Initialize evaluation metrics and counters ---
    all_l1_errors = []
    total_inference_time = 0.0
    steps_evaluated = 0
    max_steps = 300  # Set maximum evaluation steps
    # --- End initialization ---
    
    print(f"\n--- 3. Starting Evaluation Loop (up to {max_steps} steps) ---")
    
    # Use torch.no_grad() to disable gradient computation, saving memory and computation
    with torch.no_grad():
        # Iterate through all episodes
        for episode_meta in tqdm(dataset_reader.episodes_meta, desc="Processing episodes"):
            episode_index = episode_meta['episode_index']
            episode_length = episode_meta['length']
            episode_data = dataset_reader.get_episode(episode_index)

            if episode_data is None:
                print(f"Warning: Could not find data for episode {episode_index}. Skipping.")
                continue

            for frame_idx in range(episode_length - action_chunk_length):
                
                # --- a. Prepare input for single sample ---
                image_np = episode_data[dataset_reader.observation_keys['image']][frame_idx]
                image_pil = Image.fromarray(image_np).convert("RGB")
                pixel_values = image_transform(image_pil).unsqueeze(0)
                
                pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
                
                proprioception_np = episode_data[dataset_reader.observation_keys['state']][frame_idx]
                proprioception_tensor = torch.tensor(proprioception_np, dtype=torch.bfloat16, device=device)

                gt_action_chunk_np = episode_data[dataset_reader.action_key][frame_idx : frame_idx + action_chunk_length]
                gt_action_chunk_tensor = torch.tensor(gt_action_chunk_np)

                # --- b. Call model for inference and time it ---
                
                # Start timing
                start_time = time.time()

                predicted_action_chunk = model.get_action(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    proprioception_values=proprioception_tensor,
                    task_description=task_description,
                    action_chunk_length=action_chunk_length,
                    verbose=False
                )

                # print(predicted_action_chunk)
                # print(gt_action_chunk_tensor)
                # breakpoint()
                
                # End timing (including GPU sync)
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Ensure GPU operations complete
                end_time = time.time()
                total_inference_time += (end_time - start_time)
                # --- End modifications ---

                # --- c. Calculate and record L1 error ---
                if predicted_action_chunk is not None:
                    predicted_action_chunk = predicted_action_chunk.cpu()
                    # print(predicted_action_chunk[0] - gt_action_chunk_tensor[0])
                    l1_error = torch.nn.functional.l1_loss(predicted_action_chunk, gt_action_chunk_tensor)
                    all_l1_errors.append(l1_error.item())

                # --- Update counter and check if limit reached ---
                steps_evaluated += 1
                if steps_evaluated >= max_steps:
                    break
            
            if steps_evaluated >= max_steps:
                break
                # --- End initialization ---

    # --- 4. Calculate and report final results ---
    
    print(f"\n--- 4. Evaluation Finished (stopped after {steps_evaluated} steps) ---")
    if not all_l1_errors:
        print("No samples were successfully evaluated.")
    else:
        average_l1_error = np.mean(all_l1_errors)
        std_l1_error = np.std(all_l1_errors)
        
        # Calculate FPS
        fps = steps_evaluated / total_inference_time if total_inference_time > 0 else 0
        
        print(f"Total samples evaluated: {steps_evaluated}")
        print(f"Average L1 Error (MAE): {average_l1_error:.6f}")
        print(f"Standard Deviation of L1 Error: {std_l1_error:.6f}")
        print(f"Total Inference Time: {total_inference_time:.2f} seconds")
        print(f"Inference FPS: {fps:.2f} frames/sec")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a robot policy model on a Zarr dataset.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to the Zarr dataset for evaluation.")
    parser.add_argument("--task_description", type=str, default="What is the next action for the robot dexhand?", help="The task_description for the task.")
    parser.add_argument("--action-chunk-length", type=int, default=16, help="Length of the action chunk to predict.")
    
    args = parser.parse_args()

    # Determine device to run on
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {eval_device}")

    evaluate_policy(
        model_path=args.model_path,
        zarr_path=args.zarr_path,
        task_description=args.task_description,
        action_chunk_length=args.action_chunk_length,
        device=eval_device
    )