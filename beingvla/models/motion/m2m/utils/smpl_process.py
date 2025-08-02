import smplx
import h5py
import os
from tqdm import tqdm
import torch
import numpy as np
from .quaternion import *
from .constant import *
from .smplx2joints import process_smplx_322_data
from .skeleton import Skeleton
from .constant import t2m_kinematic_chain as kinematic_chain
from .rotation_conversions import aa_to_rot6d


def recover_root_rot_pos(data):
    # Root joint rotation is calculated from rotation velocity (rot_vel)
    # Rotation velocity (rot_vel) is typically Y-axis angular velocity (e.g., changes in body orientation),
    # calculated by cumulative sum (cumsum) to get total rotation angle (r_rot_ang).
    rot_vel = data[..., 0] 
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # Angle → Quaternion
    # Only calculating Y-axis rotation (around vertical axis), so quaternion only has w and y components (x=0, z=0)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # Why use quaternions?
    # Quaternions avoid Gimbal Lock and are suitable for accumulating rotations.
    # Integrating rotation velocity gives angles, which naturally convert to quaternions.

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    """Recover joint positions from rotation parameter data
    data: Input rotation parameter data
    joints_num: Number of joints
    skeleton: Skeleton object containing forward kinematics calculation method
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)  # Extract root joint rotation (quaternion representation) and position from input data

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)  # Convert root joint quaternion rotation to continuous 6D representation
    # Why convert to 6D? 6D representation (first two columns of rotation matrix) is more stable in deep learning tasks,
    # avoiding quaternion ambiguity (q and -q represent the same rotation).
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6  # Extract non-root joint rotation parameters (continuous 6D representation) from input data
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)  # Merge root and non-root joint rotation parameters into a complete joint rotation parameter set
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)  # Use forward kinematics to calculate global position of each joint

    return positions, cont6d_params


def recover_from_ric(data, joints_num):
    """Recover joint positions in global coordinate system from given motion data (containing root node information and local joint coordinates)."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)  # Recover root joint rotation and global position
    positions = data[..., 4:(joints_num - 1) * 3 + 4]  # (k, 21, 3)  Extract local joint positions
    positions = positions.view(positions.shape[:-1] + (-1, 3))  # (k, 21, 3)

    '''Add Y-axis rotation to local joints'''
    """
    Rotate local joint positions to global coordinate system
    qinv(r_rot_quat): Calculate inverse of root joint rotation (i.e., reverse rotation)
    [..., None, :] and expand: Expand quaternion dimensions to match number of joints
    qrot: Rotate local positions with inverse quaternion to cancel root rotation, getting rotation-invariant global positions
    """
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)  # (k, 21, 3)

    '''Add root XZ to joints'''
    """Add root joint XZ position to all joints"""
    positions[..., 0] += r_pos[..., 0:1]  # (k, 21, 3)
    positions[..., 2] += r_pos[..., 2:3]  # (k, 21, 3)

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)  # (k, 22, 3)

    return positions


def angle_axis_to_quaternion_np(angle_axis: np.ndarray) -> np.ndarray:
    """Convert an angle axis to a quaternion.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (numpy.ndarray): array with angle axis.

    Return:
        numpy.ndarray: array with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = np.random.rand(2, 3)  # Nx3
        >>> quaternion = angle_axis_to_quaternion_numpy(angle_axis)  # Nx4
    """
    if not isinstance(angle_axis, np.ndarray):
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}"
                         .format(angle_axis.shape))
    # unpack input and compute conversion
    a0: np.ndarray = angle_axis[..., 0:1]
    a1: np.ndarray = angle_axis[..., 1:2]
    a2: np.ndarray = angle_axis[..., 2:3]
    theta_squared: np.ndarray = a0 * a0 + a1 * a1 + a2 * a2

    theta: np.ndarray = np.sqrt(theta_squared)
    half_theta: np.ndarray = theta * 0.5

    mask: np.ndarray = theta_squared > 0.0
    ones: np.ndarray = np.ones_like(half_theta)

    k_neg: np.ndarray = 0.5 * ones
    k_pos: np.ndarray = np.sin(half_theta) / theta
    k: np.ndarray = np.where(mask, k_pos, k_neg)
    w: np.ndarray = np.where(mask, np.cos(half_theta), ones)

    quaternion_xyz: np.ndarray = np.zeros_like(angle_axis)
    quaternion_xyz[..., 0:1] += a0 * k
    quaternion_xyz[..., 1:2] += a1 * k
    quaternion_xyz[..., 2:3] += a2 * k
    return np.concatenate([w, quaternion_xyz], axis=-1)


def quaternion_to_angle_axis_np(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (numpy.ndarray): array with quaternions.

    Return:
        numpy.ndarray: array with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = np.random.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis_numpy(quaternion)  # Nx3
    """
    if not isinstance(quaternion, np.ndarray):
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: np.ndarray = quaternion[..., 1]
    q2: np.ndarray = quaternion[..., 2]
    q3: np.ndarray = quaternion[..., 3]
    sin_squared_theta: np.ndarray = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: np.ndarray = np.sqrt(sin_squared_theta)
    cos_theta: np.ndarray = quaternion[..., 0]
    two_theta: np.ndarray = 2.0 * np.where(
        cos_theta < 0.0,
        np.arctan2(-sin_theta, -cos_theta),
        np.arctan2(sin_theta, cos_theta))

    k_pos: np.ndarray = two_theta / sin_theta
    k_neg: np.ndarray = 2.0 * np.ones_like(sin_theta)
    k: np.ndarray = np.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: np.ndarray = np.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def recover_root_rot_pos_v2(data):  # v2 version, corresponding to improved smpl_to_dxxx_v2
    # data structure: [root_rotation_6d, root_linear_velocity_xz, root_height, joint_rotations_6d]
    r_rot_6d = data[..., :6]  # Root joint rotation 6D representation
    l_velocity = data[..., 6:8] # xz linear velocity
    root_y = data[..., 8:9]     # root height

    r_rot_quat = cont6d_to_quat(torch.from_numpy(r_rot_6d)).float()  # Convert 6D to quaternion

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = torch.from_numpy(l_velocity[:-1]).float()  # xz velocity, shift by one, accumulate from second frame (compatible even for single frame)
    '''Add Y-axis rotation to root position (not needed here as velocity is already in global coordinate system)'''
    # r_pos = qrot(qinv(r_rot_quat), r_pos)  # No rotation needed as velocity is already in global coordinates

    r_pos = torch.cumsum(r_pos, dim=-2)  # Accumulate position

    r_pos[..., 1] = torch.from_numpy(root_y[:-1, 0]).float()  # root height, shift by one, start from second frame

    return r_rot_quat[:-1], r_pos[:-1]  # Also need to slice output to keep frame count consistent (for single frame duplication, only one frame has valid information)


def d135_to_smpl(data):  # v2 version, corresponding to improved smpl_to_dxxx_v2 and recover_root_rot_pos_v2
    joints_num = 22
    
    r_rot_cont6d = data[..., :6] # (seq_len - 1, 6)
    r_rot_cont6d = r_rot_cont6d.reshape(-1, 1, 6)
    r_rot_quat = cont6d_to_quat(r_rot_cont6d).float() # (seq_len - 1, 1, 4)
    r_rot_quat = r_rot_quat.squeeze(1) # (seq_len - 1, 4)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).float().to(data.device) # (seq_len-1,3)
    # r_pos[..., 1:, [0, 2]] = data[..., 6:8].float().to(data.device) # (seq_len-2, 2)
    r_pos[..., 1:, [0, 2]] = data[:-1, 6:8].float().to(data.device) # (seq_len-1, 2)  <-- Corrected line: Slice data
    r_pos = qrot(qinv(r_rot_quat), r_pos) # (seq_len - 1, 3)
    r_pos = torch.cumsum(r_pos, dim=-2) # (seq_len -1, 3)
    r_pos[..., 1] = data[..., 8:9].float().to(data.device).squeeze(1)  # (seq_len-1,)
    

    start_indx = 6 + 2 + 1
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = cont6d_params.float().to(data.device) # (seq_len-1, (22-1)*6)
    cont6d_params = cont6d_params.view(-1, joints_num - 1, 6) # (seq_len-1, 21, 6)
    quat_params = torch.cat([cont6d_to_quat(r_rot_cont6d).float(), cont6d_to_quat(cont6d_params)], dim=1) # (seq_len - 1, 22, 4)

    aa = quaternion_to_angle_axis(quat_params)
    smpl_data = torch.cat([aa.reshape(-1, 66), r_pos], dim=-1)
    #print("smpl_data:", smpl_data.shape)
    return smpl_data

def smpl_to_d135(smpl_data):  # v2 version, improved root joint rotation handling
    
    if len(smpl_data) == 1:
        smpl_data = np.concatenate([smpl_data, smpl_data], axis=0)

    trans = smpl_data[:, 66:]
    
    # aa_to_quat
    quat_params = angle_axis_to_quaternion_np(smpl_data[:, :66].reshape(-1, 22, 3))  # (seq_len, 22, 4)
    
    cont_6d_params = quaternion_to_cont6d_np(quat_params) # (seq_len, 22, 6)

    r_rot = quat_params[:, 0].copy() # (seq_len, 4)
    r_rot_cont6d = quaternion_to_cont6d_np(r_rot.reshape(-1, 1, 4)).reshape(-1, 6) # (seq_len, 6)

    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (trans[1:, :] - trans[:-1, :]).copy()
    # print(r_rot.shape)
    velocity = qrot_np(r_rot[1:], velocity)

    # Root height
    root_y = trans[:, 1:2] # (seq_len, 1)

    # Root rotation and linear velocity
    # (seq_len-1, 2) linear velovity on xz plane
    l_velocity = velocity[:, [0, 2]] # (seq_len-1, 2)

    root_data = np.concatenate([r_rot_cont6d[:-1], l_velocity, root_y[:-1]], axis=-1) # (seq_len-1, 6+2+1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)  # (seq_len, (22-1)*6)

    # Concatenate all features into a single array
    data = np.concatenate([root_data, rot_data[:-1]], axis=-1) # (seq_len -1, 6+2+1 + (22-1)*6)
    # print(data.shape)
    return data


def d268_to_smpl(data):
    joints_num = 22

    ################
    r_rot_cont6d = data[..., :6] # (seq_len - 1, 6)
    r_rot_cont6d = r_rot_cont6d.reshape(-1, 1, 6)
    r_rot_quat = cont6d_to_quat(r_rot_cont6d).float() # (seq_len - 1, 1, 4)
    r_rot_quat = r_rot_quat.squeeze(1) # (seq_len - 1, 4)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).float().to(data.device) # (seq_len-1,3)
    # r_pos[..., 1:, [0, 2]] = data[..., 6:8].float().to(data.device) # (seq_len-2, 2)
    r_pos[..., 1:, [0, 2]] = data[:-1, 6:8].float().to(data.device) # (seq_len-1, 2)  <-- Corrected line: Slice data
    r_pos = qrot(qinv(r_rot_quat), r_pos) # (seq_len - 1, 3)
    r_pos = torch.cumsum(r_pos, dim=-2) # (seq_len -1, 3)
    r_pos[..., 1] = data[..., 8:9].float().to(data.device).squeeze(1)  # (seq_len-1,)
    ################
    # start_indx = 6 + 2 + 1
    start_indx = 6 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    # cont6d_params = data[..., start_indx:end_indx]
    # cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    # # print(cont6d_params.shape)
    # cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = cont6d_params.float().to(data.device) # (seq_len-1, (22-1)*6)
    cont6d_params = cont6d_params.view(-1, joints_num - 1, 6) # (seq_len-1, 21, 6)
    quat_params = torch.cat([cont6d_to_quat(r_rot_cont6d).float(), cont6d_to_quat(cont6d_params)], dim=1) # (seq_len - 1, 22, 4)


    # quat_params = cont6d_to_quat(cont6d_params)
    aa = quaternion_to_angle_axis(quat_params)
    # print(aa.shape)
    # print(r_pos.shape)

    smpl_data = torch.cat([aa.reshape(-1, 66), r_pos], dim=-1)
    # print(smpl_data.shape)
    # print(asd)

    return smpl_data

def smpl_to_d268(smpl_data, positions):
    if len(smpl_data) == 1:
        smpl_data = np.concatenate([smpl_data, smpl_data], axis=0)
    
    if len(positions) == 1:
        positions = np.concatenate([positions, positions], axis=0)

    trans = smpl_data[:, 66:]

    # trans_init_xz = trans[0] * np.array([1, 0, 1])  # x,z take original values, y takes 0
    # trans = trans - trans_init_xz

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])  # x,z take original values, y takes 0
    positions = positions - root_pose_init_xz                   # After subtraction, place at x,z origin

    positions_b = positions.copy()
    # Store the global positions for further analysis
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        fid_r, fid_l = [8, 11], [7, 10]

        velfactor, heightfactor = np.array(
            [thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z)
                  < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z)
                  < velfactor)).astype(np.float32)
        return feet_l, feet_r
    
    # feet_l, feet_r = foot_detect(positions, feet_thre)
    feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    # r_rot = None
    #####################################

    # aa_to_quat
    quat_params = angle_axis_to_quaternion_np(smpl_data[:, :66].reshape(-1, 22, 3))  # (seq_len, 4)

    cont_6d_params = quaternion_to_cont6d_np(quat_params)

    r_rot = quat_params[:, 0].copy() # (seq_len, 4)
    r_rot_cont6d = quaternion_to_cont6d_np(r_rot.reshape(-1, 1, 4)).reshape(-1, 6) # (seq_len, 6)

    def get_rifke(positions):
        """
        Adjusts the motion capture data to a local pose representation and ensures
        that all poses face in the Z+ direction.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            numpy.ndarray: Adjusted motion capture data in a local pose representation.
        """
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions
    positions = get_rifke(positions)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)
    ############################

    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (trans[1:, :] - trans[:-1, :]).copy()
    velocity = qrot_np(r_rot[1:], velocity)

    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    # Root height
    root_y = trans[:, 1:2]
    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_rot_cont6d[:-1], l_velocity, root_y[:-1]], axis=-1) # (seq_len-1, 6+2+1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Concatenate all features into a single array
    data = root_data
    # data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data

def smpl_to_d130(smpl_data):
    if len(smpl_data) == 1:
        smpl_data = np.concatenate([smpl_data, smpl_data], axis=0)
        
    trans = smpl_data[:, 66:]

    # trans_init_xz = trans[0] * np.array([1, 0, 1])  # x,z take original values, y takes 0
    # trans = trans - trans_init_xz

    # aa_to_quat
    quat_params = angle_axis_to_quaternion_np(smpl_data[:, :66].reshape(-1, 22, 3))  # (seq_len, 4)
    
    cont_6d_params = quaternion_to_cont6d_np(quat_params)

    r_rot = quat_params[:, 0].copy()  # Losing imaginary parts? Will it have impact?

    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (trans[1:, :] - trans[:-1, :]).copy()

    # print(r_rot.shape)
    velocity = qrot_np(r_rot[1:], velocity)

    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    # Root height
    root_y = trans[:, 1:2]
    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])  ## Keep
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Concatenate all features into a single array
    data = root_data
    data = np.concatenate([data, rot_data[:-1]], axis=-1)

    return data


def smpl_to_d263(smpl_data, positions):
    """
    Convert SMPL motion data to a 263-dimensional representation.
    
    Args:
        smpl_data: Input SMPL motion data with shape (seq_len, 72)
        positions: Joint positions with shape (seq_len, joints_num, 3)
    
    Returns:
        np.ndarray: Processed motion data in 263-dimensional representation
    """

    # Ensure we have at least 2 frames for velocity calculations
    if len(smpl_data) == 1:
        smpl_data = np.concatenate([smpl_data, smpl_data], axis=0)
    if len(positions) == 1:
        positions = np.concatenate([positions, positions], axis=0)

    trans = smpl_data[:, 66:]

    """Center the skeleton at the origin in the XZ plane."""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])  # x,z take original values, y takes 0
    positions = positions - root_pose_init_xz   # After subtraction, place at x,z origin
  
    global_positions = positions.copy()  # Store the global positions for further analysis

    def _detect_foot_contacts(positions, threshold):
        """Detect foot contacts based on velocity threshold."""
        fid_r, fid_l = [8, 11], [7, 10]

        def calc_foot_vel(foot_indices):
            vel_x = (positions[1:, foot_indices, 0] - positions[:-1, foot_indices, 0]) ** 2
            vel_y = (positions[1:, foot_indices, 1] - positions[:-1, foot_indices, 1]) ** 2
            vel_z = (positions[1:, foot_indices, 2] - positions[:-1, foot_indices, 2]) ** 2
            return (vel_x + vel_y + vel_z) < threshold
        
        feet_l = calc_foot_vel(fid_l).astype(np.float32)
        feet_r = calc_foot_vel(fid_r).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = _detect_foot_contacts(positions, threshold=0.002) # Extract foot contacts

    # Convert rotations to quaternions and 6D continuous representation
    quat_params = angle_axis_to_quaternion_np(smpl_data[:, :66].reshape(-1, 22, 3))  # (seq_len, 4)
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()

    def _to_local_pose(positions):
        """Convert to local pose representation facing Z+ direction."""
        positions[..., 0] -= positions[:, 0:1, 0]  # # Center positions around root joint
        positions[..., 2] -= positions[:, 0:1, 2]

        # Rotate all poses to face Z+ direction
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions
    
    positions = _to_local_pose(positions)
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Calculate velocities
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1]
                )
    local_vel = local_vel.reshape(len(local_vel), -1)
   
   
    '''Root Velocity'''
    root_lin_vel = (trans[1:] - trans[:-1]).copy()
    root_lin_vel = qrot_np(r_rot[1:], root_lin_vel)
    root_ang_vel = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    root_ang_vel = np.arcsin(root_ang_vel[:, 2:3])

    velocity = (trans[1:, :] - trans[:-1, :]).copy()
    velocity = qrot_np(r_rot[1:], velocity)

    # Root height
    root_y = trans[:, 1:2]
    root_data = np.concatenate([
        root_ang_vel, 
        root_lin_vel[:, [0, 2]],  # Only use x and z components
        root_y[:-1]
    ], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1) # Prepare rotation data

    return np.concatenate([
        root_data,
        ric_data[:-1],
        rot_data[:-1],
        local_vel,
        feet_l,
        feet_r
    ], axis=-1)


def d130_to_smpl(data):
    joints_num = 22

    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    # print(cont6d_params.shape)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    quat_params = cont6d_to_quat(cont6d_params)
    aa = quaternion_to_angle_axis(quat_params)
    # print(aa.shape)
    # print(r_pos.shape)

    smpl_data = torch.cat([aa.reshape(-1, 66), r_pos], dim=-1)
    # print(smpl_data.shape)
    # print(asd)

    return smpl_data


def d263_to_smpl(data):
    joints_num = 22

    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)

    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    quat_params = cont6d_to_quat(cont6d_params)
    aa = quaternion_to_angle_axis(quat_params)
 
    smpl_data = torch.cat([aa.reshape(-1, 66), r_pos], dim=-1)

    return smpl_data


def recover_from_smpl(data, num_joints, feat_pose, smplx_layer, smplx_model, comp_device):
    data = data.reshape(-1, data.shape[-1])
 
    if feat_pose == 'smpl_d130_20':
        smpl_data = d130_to_smpl(torch.tensor(data, dtype=torch.float32))
        lens = len(smpl_data)
        smpl_data = np.concatenate([smpl_data[:, :66], np.zeros((lens, 243)), smpl_data[:, 66:], np.zeros((lens, 10))], axis=-1).reshape(1, -1, 322)
        smpl_data = torch.tensor(smpl_data).to(comp_device)
        vert, joints, pose, faces = process_smplx_322_data(
            smpl_data, smplx_layer, smplx_model, device=comp_device)
        xyz = joints
        xyz = xyz[:, :, :22, :].reshape(-1, 22, 3)#.detach().cpu().numpy()
    elif feat_pose == 'smpl_d135_20':
        smpl_data = d135_to_smpl(torch.tensor(data, dtype=torch.float32))
        lens = len(smpl_data)
        smpl_data = np.concatenate([smpl_data[:, :66], np.zeros((lens, 243)), smpl_data[:, 66:], np.zeros((lens, 10))], axis=-1).reshape(1, -1, 322)
        smpl_data = torch.tensor(smpl_data).to(comp_device)
        vert, joints, pose, faces = process_smplx_322_data(
            smpl_data, smplx_layer, smplx_model, device=comp_device)
        xyz = joints
        xyz = xyz[:, :, :22, :].reshape(-1, 22, 3)#.detach().cpu().numpy()
    elif feat_pose == 'smpl_d263_20':
        smpl_data = d263_to_smpl(torch.tensor(data, dtype=torch.float32))
        lens = len(smpl_data)
        smpl_data = np.concatenate([smpl_data[:, :66], np.zeros((lens, 243)), smpl_data[:, 66:], np.zeros((lens, 10))], axis=-1).reshape(1, -1, 322)
        smpl_data = torch.tensor(smpl_data).to(comp_device)
        vert, joints, pose, faces = process_smplx_322_data(
            smpl_data, smplx_layer, smplx_model, device=comp_device)
        xyz = joints
        xyz = xyz[:, :, :22, :].reshape(-1, 22, 3)#.detach().cpu().numpy()
    elif feat_pose == 'smpl_d268_20':
        smpl_data = d268_to_smpl(torch.tensor(data, dtype=torch.float32))
        lens = len(smpl_data)
        smpl_data = np.concatenate([smpl_data[:, :66], np.zeros((lens, 243)), smpl_data[:, 66:], np.zeros((lens, 10))], axis=-1).reshape(1, -1, 322)
        smpl_data = torch.tensor(smpl_data).to(comp_device)
        vert, joints, pose, faces = process_smplx_322_data(
            smpl_data, smplx_layer, smplx_model, device=comp_device)
        xyz = joints
        xyz = xyz[:, :, :22, :].reshape(-1, 22, 3)#.detach().cpu().numpy()

    return xyz


def get_xyz_calculator(motion_feat):
    if motion_feat in ["hm3d263", "smpl263"]:
        calculate_xyz = recover_from_ric
    else:
        from .smplx2joints import get_smplx_layer
        from functools import partial
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        smplx_layer, smplx_model = get_smplx_layer(device)
        calculate_xyz = partial(recover_from_smpl, 
                                feat_pose=motion_feat, 
                                smplx_layer=smplx_layer,
                                smplx_model=smplx_model,
                                comp_device=device)
        
    return calculate_xyz


l_idx1, l_idx2 = 5, 8  # Lower legs
r_hip, l_hip = 2, 1  # l_hip, r_hip
fid_r, fid_l = [8, 11], [7, 10] # Right/Left foot
face_joint_indx = [2, 1, 17, 16]  # Face direction, r_hip, l_hip, sdr_r, sdr_l
t2m_raw_offsets = np.array(t2m_raw_offsets)
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def joint_to_d263(positions, feet_thre, tgt_offsets):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    # print('joint_to_d263', floor_height)
    positions[:, :, 1] -= floor_height
    #plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)
    #
    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]
    
    '''All initially face Z+'''
    #"""
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)  # forward (3,), rotate around y-axis
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]  # forward (3,)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)
    #plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)
    #pdb.set_trace()
    #"""
    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)
    
    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions
    
    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)  # (seq_len, joints_num, 4)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        r_rot = quat_params[:, 0].copy()  # (seq_len, 4)
      
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # (seq_len - 1, 3)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (seq_len - 1, 4)
        quat_params[1:, 0] = r_velocity
        
        return quat_params, r_velocity, velocity, r_rot  # (seq_len, joints_num, 4)

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)  # (seq_len, joints_num, 4)
        
        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()  # (seq_len, 4)
        
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # (seq_len - 1, 3)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot
    
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)
    
    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    r_velocity = np.arcsin(r_velocity[:, 2:3])  # (seq_len-1, 1) rotation velocity along y-axis
    l_velocity = velocity[:, [0, 2]]  # (seq_len-1, 2) linear velovity on xz plane
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
   
    '''Get Joint Rotation Representation'''
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)  # (seq_len, (joints_num-1) *6) quaternion for skeleton joints

    '''Get Joint Rotation Invariant Position Represention'''
    ric_data = positions[:, 1:].reshape(len(positions), -1)   # (seq_len, (joints_num-1)*3) local joint position

    '''Get Joint Velocity Representation'''
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])  # (seq_len-1, joints_num*3)
    
    local_vel = local_vel.reshape(len(local_vel), -1)
    
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def raw_to_d263(pose_data, body_model, trans_matrix, joint_num, tgt_offsets, device):
    root_orient, pose_body, pose_hand, trans, betas = pose_data
    assert root_orient.shape[-1]==3 and pose_body.shape[-1]==63 and trans.shape[-1]==3 
    if len(root_orient)==1:
        root_orient = root_orient.repeat(2,1)
        pose_body = pose_body.repeat(2,1)
        trans = trans.repeat(2,1)
        betas = betas.repeat(2,1)
        if pose_hand is not None:
            pose_hand = pose_hand.repeat(2,1)
    
    betas = betas.to(device) if betas is not None else None
    body_parms = {
                'root_orient': root_orient.to(device), 'pose_body': pose_body.to(device),
                'trans': trans.to(device), 'betas': betas,
            }
    
    if pose_hand is not None:
        body_parms['pose_hand'] = pose_hand.to(device)
    
    # inter-media processing
    with torch.no_grad():
        body = body_model(**body_parms)
        
    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)[:, :joint_num, :]  # N, 52, 3
    
    # N,52,3 - > N,263
    data, ground_positions, positions, l_velocity = joint_to_d263(pose_seq_np_n, 0.002, tgt_offsets)  # 739,263
    rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joint_num)  # 1,739,22,3
    
    return data, rec_ric_data


from .constant import HUMAN_MODEL_PATH, SMPL_MEAN_FILE
from .body_model.smplify import SMPLify3D
class joint_to_smpl:
    def __init__(self, num_frames, device_id, cuda=False):
        self.device = torch.device("cuda:" + str(device_id) if cuda else "cpu")
        # self.device = torch.device("cpu")
        self.batch_size = num_frames
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = 150
        self.fix_foot = False
        print(HUMAN_MODEL_PATH)
        smplmodel = smplx.create(HUMAN_MODEL_PATH,
                                 model_type="smpl", gender="neutral", ext="pkl",
                                 batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = SMPL_MEAN_FILE

        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                            batch_size=self.batch_size,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)


    def npy2smpl(self, npy_path):
        out_path = npy_path.replace('.npy', '_rot.npy')
        motions = np.load(npy_path, allow_pickle=True)[None][0]
        # print_batch('', motions)
        n_samples = motions['motion'].shape[0]
        all_thetas = []
        for sample_i in tqdm(range(n_samples)):
            thetas, _ = self.joint2smpl(motions['motion'][sample_i].transpose(2, 0, 1))  # [nframes, njoints, 3]
            all_thetas.append(thetas.cpu().numpy())
        motions['motion'] = np.concatenate(all_thetas, axis=0)
        print('motions', motions['motion'].shape)

        print(f'Saving [{out_path}]')
        np.save(out_path, motions)
        exit()

    def joint2smpl(self, input_joints, init_params=None):
        _smplify = self.smplify # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        keypoints_3d = torch.zeros(self.batch_size, self.num_joints, 3).to(self.device)

        # run the whole seqs
        num_seqs = input_joints.shape[0]


        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            # seq_ind=idx
        )
        
        thetas = new_opt_pose.reshape(self.batch_size, 24, 3)
        thetas = aa_to_rot6d(thetas)  # [bs, 24, 6]
        root_loc = torch.tensor(keypoints_3d[:, 0]) # [bs, 3]
        # root_loc = keypoints_3d[:, 0] # [bs, 3]
        root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [bs, 1, 6]
        thetas = torch.cat([thetas, root_loc], dim=1).unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 6, 196]

        return thetas.clone().detach(), {'pose': new_opt_joints[0, :24].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam': new_opt_cam_t.clone().detach()}

