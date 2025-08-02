import torch
import numpy as np
from manotorch.manolayer import ManoLayer
from .mano_utils import rot6d_to_rotmat, rotmat_dist, decode_absolute_pose_from_relative, batch_rot2aa, get_pose12d_from_world_to_camera, batch_rodrigues

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mano_layer_dict = dict(
    left=ManoLayer(mano_assets_root='beingvla/models/motion/mano',use_pca=False, side='left', center_idx=0).to(device),
    right=ManoLayer(mano_assets_root='beingvla/models/motion/mano', use_pca=False, side='right', center_idx=0).to(device)
)

def mano_forward(mano_theta, mano_beta, wrist_rot, wrist_trans, side, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    theta = mano_theta.to(device)
    beta = mano_beta.to(device) if len(mano_beta) == len(mano_theta) else mano_beta.view(-1,10).repeat(theta.shape[0], 1).to(device)
    rot = wrist_rot.to(device)
    trans = wrist_trans.to(device)

    mano_layer = mano_layer_dict['left'] if 'l' in side else mano_layer_dict['right']
    mano_output = mano_layer(torch.cat([rot, theta], dim=-1), beta)
    hand_verts, hand_joints = mano_output.verts, mano_output.joints
    hand_verts += trans.unsqueeze(1)
    hand_joints += trans.unsqueeze(1)

    return hand_verts, hand_joints

def get_single_mano_joints(data, joints_num, side):
    assert data.shape[-1] == 109, f"Expected data shape to be B, T, 109 but got {data.shape}"
    assert joints_num == 21, f"Expected joints_num to be 21 but got {joints_num}"

    # split 109 parameters into 90 + 10 + 6 + 3

    mano_theta = data[..., :90]  # 90 parameters for joint angles
    mano_rot = data[..., 90:96]  # 6 parameters for wrist rotation
    mano_trans = data[..., 96:99]  # 3 parameters for wrist translation
    mano_beta = data[..., 99:109]  # 10 parameters for hand shape
    B, T = mano_theta.shape[:2]

    theta_flatten_9d = rot6d_to_rotmat(mano_theta.reshape(-1, 6))  # T, 3, 3
    theta = batch_rot2aa(theta_flatten_9d).reshape(T, 45)  # T, 45
    beta = mano_beta.reshape(-1, 10)  # T, 10
    rot_flatten_9d = rot6d_to_rotmat(mano_rot.reshape(-1, 6))  # T, 3, 3
    rot = batch_rot2aa(rot_flatten_9d).reshape(T, 3)  # T, 3
    trans = mano_trans.reshape(-1, 3)  # T, 3

    hand_verts, hand_joints = mano_forward(  # T, 21, 3
        theta,
        beta,
        rot,
        trans,
        side
    )

    return hand_verts, hand_joints  # T, 21, 3

def restore_with_mask(x_filtered: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x_filtered: shape (..., D') — only masked-in features
    mask: shape (D,) — bool or 0/1 tensor indicating which dimensions were kept
    Returns:
        x_restored: shape (..., D), with 0s in masked-out positions
    """
    mask = mask.bool().to(x_filtered.device)  # Ensure mask is boolean and on the same device as x_filtered
    D = mask.shape[0]
    x_shape = x_filtered.shape[:-1] + (D,)  # target shape

    x_restored = torch.zeros(*x_shape, dtype=x_filtered.dtype, device=x_filtered.device)

    x_restored[..., mask] = x_filtered
    return x_restored

valid_theta_mask = torch.tensor([1, 1, 1, 1, 0, 1, 0, 0, 1, 
                                0, 1, 1, 1, 0, 1, 0, 0, 1,
                                1, 1, 1, 1, 0, 1, 0, 0, 1,
                                1, 1, 1, 1, 0, 1, 0, 0, 1,
                                1, 1, 1, 1, 1, 1, 0, 0, 1],
                               dtype=torch.float32)

def get_mano_joints(data, sides, default_beta=None):
    """
    Extracts MANO joints from the input data.
    
    Args:
        data (torch.Tensor): B, T, D=109 / 99
        sides (strs): B * (left or right)
        default_beta (torch.Tensor, optional): Default beta parameters for the hand shape. Required if data.shape[-1] == 99. (B, 10)
        
    Returns:
        torch.Tensor: Extracted joint positions: B, T, J, 3
    """
    assert data.shape[-1] in [36, 51, 63, 99, 100, 109, 114, 162, 172], f"Expected data shape to be B, T, 36/51/63/99/100/109/114/162/172 but got {data.shape}"
    if data.shape[-1] in [36, 51, 99, 100, 114, 162]:
        assert default_beta is not None

    if data.shape[-1] in [63]:
        # reshape to B, T, 63 -> B, T, 21, 3
        return data.reshape(data.shape[0], data.shape[1], 21, 3)

    # split 109 parameters into 90 + 10 + 6 + 3
    if data.shape[-1] in [109, 172]:
        mano_theta = data[..., :90]  # 90 parameters for joint angles
        mano_rot = data[..., 90:96]  # 6 parameters for wrist rotation
        mano_trans = data[..., 96:99]  # 3 parameters for wrist translation
        mano_beta = data[..., 99:109]  # 10 parameters for hand shape
        B, T = mano_theta.shape[:2]
    elif data.shape[-1] in [99, 162]:
        mano_theta = data[..., :90]  # 90 parameters for joint angles
        mano_rot = data[..., 90:96]  # 6 parameters for wrist rotation
        mano_trans = data[..., 96:99]  # 3 parameters for wrist translation
        B, T = mano_theta.shape[:2]
        # expand beta from B, 10 to B, T, 10
        mano_beta = default_beta.unsqueeze(1).expand(-1, T, -1)
    elif data.shape[-1] in [51, 114]:
        mano_theta = data[..., :45]
        mano_rot = data[..., 45:48]
        mano_trans = data[..., 48:51]
        B, T = mano_theta.shape[:2]
        mano_beta = default_beta.unsqueeze(1).expand(-1, T, -1)
    elif data.shape[-1] in [36, 100]:
        mano_theta = data[..., :30]
        mano_rot = data[..., 30:33]
        mano_trans = data[..., 33:36]
        mano_theta = restore_with_mask(mano_theta, valid_theta_mask)  # B, T, 45
        B, T = mano_theta.shape[:2]
        mano_beta = default_beta.unsqueeze(1).expand(-1, T, -1)
    else:
        raise NotImplementedError
    
    left_indices = [i for i, x in enumerate(sides) if x == 'left']
    right_indices = [i for i, x in enumerate(sides) if x == 'right']

    final_joints = []
    for side in ['left', 'right']:
        if data.shape[-1] not in [51, 114, 36, 100]:
            theta_flatten_9d = rot6d_to_rotmat(mano_theta[eval(f'{side}_indices')].reshape(-1,6)) # B'*T*15, 3, 3
            theta = batch_rot2aa(theta_flatten_9d).reshape(-1, 45)  # B'*T, 45
            rot_flatten_9d = rot6d_to_rotmat(mano_rot[eval(f'{side}_indices')].reshape(-1,6)) # B'*T, 3, 3
            rot = batch_rot2aa(rot_flatten_9d).reshape(-1, 3) # B'*T, 3
            trans = mano_trans[eval(f'{side}_indices')].reshape(-1, 3)  # B'*T, 3
        else:
            theta = mano_theta[eval(f'{side}_indices')].reshape(-1, 45)
            rot = mano_rot[eval(f'{side}_indices')].reshape(-1, 3)
            trans = mano_trans[eval(f'{side}_indices')].reshape(-1, 3)

        beta = mano_beta[eval(f'{side}_indices')].reshape(-1, 10) # B'*T, 10

        hand_verts, hand_joints = mano_forward( # B'*T, 21, 3
            theta,
            beta,
            rot,
            trans,
            side
        )
        final_joints.append(hand_joints.reshape(-1, T, 21, 3))  # B', T, 21, 3

    hand_joints = torch.cat(final_joints, dim=0)  # (B, T, 21, 3)
    hand_joints = hand_joints.contiguous().view(B, T, 21, 3)  # B, T, J, 3
    return hand_joints



def get_xyz_calculator(motion_feat):
    if motion_feat in ["mano36", "mano51", "mano63", "mano99", "mano100", "mano109", "mano114", "mano162", "mano172"]:
        calculate_xyz = get_mano_joints
    else:
        assert False
        
    return calculate_xyz


@torch.jit.script
def get_pc_from_world_to_camera(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Transform points from the world frame to the camera frame using the extrinsics matrix.

    Args:
        points_world: (T, N, 3) tensor of points in the world frame.
        extrinsics: (4, 4) or (T, 4, 4) tensor of world2camera matrices.

    Returns:
        points_camera: (T, N, 3) tensor of points in the camera frame.
    """
    if points_world.ndim == 2:
        points_world = points_world.unsqueeze(0)  # (1, N, 3)

    if extrinsics.ndim == 2:
        extrinsics = extrinsics.unsqueeze(0)  # (1, 4, 4)

    T, N, _ = points_world.shape
    if extrinsics.shape[0] == 1 and T > 1:
        extrinsics = extrinsics.expand(T, -1, -1)

    ones = torch.ones((T, N, 1), dtype=points_world.dtype, device=points_world.device)
    points_world_hom = torch.cat([points_world, ones], dim=-1)  # (T, N, 4)

    points_camera_hom = torch.bmm(points_world_hom, extrinsics.transpose(1, 2))  # (T, N, 4)

    points_camera = points_camera_hom[:, :, :3]  # (T, N, 3)

    return points_camera