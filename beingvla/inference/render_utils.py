"""
Clean rendering utilities for BeingVLA inference.
"""

import sys
import contextlib
from io import StringIO
import os
import warnings

# Suppress specific warnings that clutter output
warnings.filterwarnings('ignore', category=UserWarning, module='pyrender')
warnings.filterwarnings('ignore', category=FutureWarning, module='trimesh')

@contextlib.contextmanager
def suppress_verbose_output():
    """Context manager to suppress verbose output during rendering."""
    # Save original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect to StringIO to capture output
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def render_hand_video_clean(hands_data, camera_intrinsic, output_path, raw_frame, fps=15, blend_alpha=0.8):
    """
    Clean wrapper for hand_2d_render that suppresses verbose output.
    
    Args:
        hands_data: Hand mesh data
        camera_intrinsic: Camera intrinsic matrix
        output_path: Output video path
        raw_frame: Background frame
        fps: Frame rate
        blend_alpha: Blending alpha
    
    Returns:
        bool: Success status
    """
    try:
        # Import here to avoid early pyrender initialization
        from beingvla.models.motion.m2m.utils.mano_vis import hand_2d_render_clean
        
        # Use the clean version that doesn't have verbose output
        hand_2d_render_clean(
            hands_data, 
            camera_intrinsic, 
            output_path, 
            raw_frame, 
            fps=fps, 
            blend_alpha=blend_alpha
        )
        
        return True
        
    except ImportError:
        # Fallback to original function with output suppression
        from beingvla.models.motion.m2m.utils.mano_vis import hand_2d_render
        
        with suppress_verbose_output():
            hand_2d_render(
                hands_data, 
                camera_intrinsic, 
                output_path, 
                raw_frame, 
                fps=fps, 
                blend_alpha=blend_alpha
            )
        
        return True
        
    except Exception as e:
        # Re-raise the exception so caller can handle it
        raise e

def get_progress_callback(total_frames, sample_id):
    """
    Create a simple progress callback for rendering.
    
    Args:
        total_frames: Total number of frames
        sample_id: Sample identifier
    
    Returns:
        Callback function
    """
    def callback(frame_idx):
        progress = (frame_idx + 1) / total_frames * 100
        if frame_idx == 0:
            print(f"  Rendering sample {sample_id} ({total_frames} frames)...", end='', flush=True)
        elif frame_idx == total_frames - 1:
            print(f" ✓ Complete")
        elif (frame_idx + 1) % max(1, total_frames // 4) == 0:
            print(f" {progress:.0f}%", end='', flush=True)
    
    return callback