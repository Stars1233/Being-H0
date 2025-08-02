# run_server.py

import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

# Ensure import paths are correct
from ....vla.being_vla_model import BeingVLAModel
from .....dataset.datasets import build_transform

# Import server class from our service file
from .internvl_service import InternVLInferenceServer


class PolicyWrapper:
    """
    A wrapper to make InternVL model compatible with our service framework.
    Handles data preprocessing and model invocation.
    """
    def __init__(self, model_path: str, action_chunk_length: int, device: torch.device):
        self.device = device
        self.action_chunk_length = action_chunk_length

        print("--- 1. Loading Model and Tokenizer ---")
        self.model = BeingVLAModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set special token IDs
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.prop_context_token_id = self.tokenizer.convert_tokens_to_ids('<PROP_CONTEXT>')
        self.model.action_chunk_token_ids = self.tokenizer.convert_tokens_to_ids(
            [f'<ACTION_CHUNK_{i}>' for i in range(self.action_chunk_length)]
        )
        print("Special token IDs set successfully.")

        print("--- 2. Preparing Image Preprocessing ---")
        force_image_size = self.model.config.vlm_config.get('force_image_size', 448)
        self.image_transform = build_transform(
            is_train=False,
            input_size=force_image_size,
            normalize_type='imagenet'
        )

    @torch.no_grad()
    def get_action(self, obs_dict: dict) -> np.ndarray:
        """
        Process incoming observation data, invoke model, and return predicted action chunk.
        This method will be called by the server endpoint.

        Args:
            obs_dict (dict): Dictionary containing observation data, must include:
                - 'image' (np.ndarray): RGB image in (H, W, 3) format.
                - 'state' (np.ndarray): 1D proprioception state vector.
                - 'task_description' (str): Text description of the task.

        Returns:
            np.ndarray: Predicted action chunk with shape (action_chunk_length, action_dim).
        """
        # --- a. Data validation and preprocessing ---
        required_keys = ['image', 'state', 'task_description']
        if not all(key in obs_dict for key in required_keys):
            raise ValueError(f"Observation dictionary must contain keys: {required_keys}")

        image_np = obs_dict['image']
        proprioception_np = obs_dict['state']
        task_description = obs_dict['task_description']

        # Image processing
        image_pil = Image.fromarray(image_np).convert("RGB")
        pixel_values = self.image_transform(image_pil).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        
        # Proprioception state processing
        proprioception_tensor = torch.tensor(
            proprioception_np, dtype=torch.bfloat16
        ).to(self.device)

        # --- b. Invoke model ---
        predicted_action_chunk = self.model.get_action(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            proprioception_values=proprioception_tensor,
            task_description=task_description,
            action_chunk_length=self.action_chunk_length,
            verbose=False  # Usually disable verbose logging on server side
        )

        # --- c. Post-process and return ---
        # Convert result from GPU tensor to CPU numpy array for serialization and network transfer
        if predicted_action_chunk is not None:
            return predicted_action_chunk.cpu().to(torch.float32).numpy().tolist()
        else:
            # Return zero array with proper shape on error to avoid client crash
            # Note: action_dim needs to be determined based on your model
            action_dim = proprioception_np.shape[0] 
            return np.zeros((self.action_chunk_length, action_dim), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run the InternVL policy inference server.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--action-chunk-length", type=int, default=16, help="Length of the action chunk to predict.")
    parser.add_argument("--port", type=int, default=5555, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to ('0.0.0.0' for all interfaces).")
    parser.add_argument("--api-token", type=str, default=None, help="Optional API token for authentication.")
    args = parser.parse_args()

    # Determine device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running policy on device: {device}")

    # Create model wrapper instance
    policy = PolicyWrapper(
        model_path=args.model_path,
        action_chunk_length=args.action_chunk_length,
        device=device
    )

    # Create and run server
    server = InternVLInferenceServer(
        policy=policy,
        port=args.port,
        host=args.host,
        api_token=args.api_token
    )
    server.run()

if __name__ == "__main__":
    main()