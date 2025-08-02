import torch
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import trange
from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import axis_angle_to_matrix, rotation_to_axis_angle
from .mano_utils import rot6d_to_rotmat, rot6d_to_aa, world_to_camera
from decord import VideoReader
from copy import deepcopy
from PIL import Image

def mano_forward(rot, trans, theta, beta, sides=['right'], relative=False):
    """
    Convert MANO parameters to standard hands_data format
    
    Args:
        rot: dict or tensor, rotation parameters {'left': (N,3,3), 'right': (N,3,3)} or single hand (N,3,3)
        trans: dict or tensor, translation parameters {'left': (N,3), 'right': (N,3)} or single hand (N,3)
        theta: dict or tensor, pose parameters {'left': (N,45), 'right': (N,45)} or single hand (N,45)
        beta: dict or tensor, shape parameters {'left': (N,10), 'right': (N,10)} or single hand (N,10)
        sides: list, list of hands to process, default ['right']
        
    Returns:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
    """
    hands_data = {}
    
    # Convert single hand inputs to dict format
    if not isinstance(rot, dict):
        side = sides[0] if sides else 'right'
        rot = {side: rot}
        trans = {side: trans}
        theta = {side: theta}
        beta = {side: beta}
    
    for side in sides:
        if side not in rot:
            continue
        
        manolayer = ManoLayer(side=side, use_pca=False, ncomps=15, 
                              center_idx=0 if relative else None,
                              mano_assets_root='beingvla/models/motion/mano').to(rot[side].device)
        
        # Prepare mano input
        wrist_pos = rotation_to_axis_angle(rot[side])
        pose_coeffs = torch.cat([wrist_pos, theta[side]], dim=1)
        
        # Expand beta if necessary
        beta_expanded = beta[side]
        if beta_expanded.dim() == 1:
            beta_expanded = beta_expanded.unsqueeze(0).expand(rot[side].shape[0], -1)
        elif beta_expanded.shape[0] == 1:
            beta_expanded = beta_expanded.expand(rot[side].shape[0], -1)
        
        # Mano forward pass
        mano_output = manolayer(pose_coeffs, beta_expanded)
        verts = mano_output.verts + trans[side].unsqueeze(1)
        joints = mano_output.joints + trans[side].unsqueeze(1)
        
        hands_data[side] = {
            'verts': verts,
            'joints': joints,
            'faces': manolayer.th_faces
        }
    
    return hands_data

def project_vertices_to_2d(vertices_3d, camera_intrinsic):
    """
    Project 3D vertices to 2D image plane
    
    Args:
        vertices_3d: (N, 3) tensor of 3D vertices
        camera_intrinsic: (3, 3) camera intrinsic matrix
    
    Returns:
        vertices_2d: (N, 2) numpy array of 2D coordinates
    """
    # Convert to homogeneous coordinates and project
    vertices_2d_homo = (camera_intrinsic @ vertices_3d.T).T
    vertices_2d = vertices_2d_homo[:, :2] / vertices_2d_homo[:, 2:3]
    return vertices_2d.numpy().astype(int)

def get_face_visibility_and_depth(vertices_3d, faces):
    """
    Calculate face depths and visibility
    
    Args:
        vertices_3d: (N, 3) tensor of 3D vertices
        faces: (F, 3) tensor of face indices
    
    Returns:
        face_depths: (F,) array of face depths (z-coordinates)
        face_centers: (F, 3) array of face center coordinates
    """
    # Get face vertices
    face_vertices = vertices_3d[faces]  # (F, 3, 3)
    
    # Calculate face centers and depths
    face_centers = face_vertices.mean(dim=1)  # (F, 3)
    face_depths = face_centers[:, 2].numpy()  # Z-coordinates
    
    return face_depths, face_centers.numpy()

def check_vertices_in_bounds(vertices_2d, image_size):
    """
    Check which vertices are within image bounds
    
    Args:
        vertices_2d: (N, 2) array of 2D coordinates
        image_size: (height, width) tuple
    
    Returns:
        valid_mask: (N,) boolean array
    """
    height, width = image_size
    valid_x = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < width)
    valid_y = (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < height)
    return valid_x & valid_y

def render_hand_mesh(img, vertices_2d, faces, face_depths, hand_side):
    """
    Render hand mesh on image
    
    Args:
        img: (H, W, 3) image array
        vertices_2d: (N, 2) array of 2D vertex coordinates
        faces: (F, 3) array of face indices
        face_depths: (F,) array of face depths
        hand_side: str, 'left' or 'right'
    
    Returns:
        img: Updated image with rendered mesh
    """
    # Define colors for different hands
    if hand_side == 'right':
        edge_color = (0, 100, 255)  # Orange-red for right hand
        fill_color = (100, 150, 255)  # Light orange-red
    else:  # left hand
        edge_color = (255, 100, 0)  # Blue for left hand  
        fill_color = (255, 150, 100)  # Light blue
    
    # Sort faces by depth (back to front rendering)
    depth_order = np.argsort(-face_depths)  # Negative for back-to-front
    
    # Create overlay for transparent rendering
    overlay = img.copy()
    
    # Render each face
    for face_idx in depth_order:
        face = faces[face_idx]
        
        # Get 2D coordinates of face vertices
        face_vertices_2d = vertices_2d[face]
        
        # Check if all vertices of the face are valid
        if np.all(face_vertices_2d >= 0):
            # Draw filled triangle
            cv2.fillPoly(overlay, [face_vertices_2d], fill_color)
            
            # Draw triangle edges
            for i in range(3):
                pt1 = tuple(face_vertices_2d[i])
                pt2 = tuple(face_vertices_2d[(i + 1) % 3])
                cv2.line(img, pt1, pt2, edge_color, 1, cv2.LINE_AA)
    
    # Blend overlay with original image
    alpha = 0.1  # Low transparency for subtle effect
    img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    return img



def calculate_global_bounds(hands_data):
    """    
    Calculate global bounds for all hands in the dataset
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3)}, ...}
    Returns:
        xlim: tuple, (xmin, xmax)
        ylim: tuple, (ymin, ymax)
        zlim: tuple, (zmin, zmax)
    """
    all_verts = []
    all_joints = []
    
    for hand_data in hands_data.values():
        all_verts.append(hand_data['verts'])
        all_joints.append(hand_data['joints'])
    
    all_points = torch.cat([
        torch.cat(all_verts, dim=0).reshape(-1, 3),
        torch.cat(all_joints, dim=0).reshape(-1, 3)
    ], dim=0)
    
    global_min = all_points.min(0)[0].numpy()
    global_max = all_points.max(0)[0].numpy()
    centers = (global_min + global_max) / 2
    ranges = global_max - global_min
    r = np.max(ranges) / 2 * 1.1
    
    xlim = (centers[0] - r, centers[0] + r)
    ylim = (centers[1] + r, centers[1] - r)
    zlim = (centers[2] + r, centers[2] - r)
    
    return xlim, ylim, zlim

def hand_render(hands_data, output_path, fps=30, figsize=(14, 14), view_angle=(20, -60)):
    """
    Render 3D hand animation
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        output_path: str, output video path
        fps: int, frame rate
        figsize: tuple, figure size
        view_angle: tuple, viewing angle (elevation, azimuth)
    """
    
    xlim, ylim, zlim = calculate_global_bounds(hands_data)
    
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    print(f"Creating 3D animation with {max_frames} frames...")
    
    frames = []
    for frame_idx in trange(max_frames, desc="rendering frames"):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            face_color = (0.2, 0.5, 0.8) if side == 'right' else (0.8, 0.5, 0.2)
            joint_color = '#00008B' if side == 'right' else '#8B0000'
            
            verts = hand_data['verts'][frame_idx].numpy()
            joints = hand_data['joints'][frame_idx].numpy()
            
            # plot 3D mesh
            mesh = Poly3DCollection(verts[hand_data['faces']], alpha=0.3)
            mesh.set_facecolor(face_color)
            mesh.set_edgecolor((0.15, 0.15, 0.15, 0.3))
            ax.add_collection3d(mesh)
            
            # plot joints scatter
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                      color=joint_color, s=50, alpha=1.0)
            
            # plot joint connections
            connections = [
                (0, 1, 2, 3, 4),      # Thumb
                (0, 5, 6, 7, 8),      # Index finger
                (0, 9, 10, 11, 12),   # Middle finger
                (0, 13, 14, 15, 16),  # Ring finger
                (0, 17, 18, 19, 20)   # Pinky finger
            ]
            
            for chain in connections:
                for i in range(len(chain)-1):
                    start = joints[chain[i]]
                    end = joints[chain[i+1]]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                           color=joint_color, linewidth=2, alpha=0.6)
        
        # Set view angle and limits
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}', pad=0, y=0.95)
        
        plt.tight_layout()
        
        # Capture the frame
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
        frames.append(rgba[:, :, :3])  # Convert to RGB
        plt.close(fig)
    
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("3D video saved successfully!")

def project_to_2d(points_3d, camera_intrinsic):
    """ Project 3D points to 2D image plane
    Args:
        points_3d: (N, 3) array of 3D points
        camera_intrinsic: (3, 3) camera intrinsic matrix
    Returns:
        points_2d: (N, 2) array of projected 2D points
    """
    # points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
    points_2d = (camera_intrinsic @ points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d.numpy().astype(int)
    
def get_valid_mask(points_2d, image_size):
    width, height = image_size
    return (points_2d[:, 0] >= 0) & (points_2d[:, 0] < height) & \
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < width)

def hand_plot(hands_data, camera_intrinsic, output_path, raw_frames=None, fps=30, image_size=(640, 480), render_mode='mesh' ):
    """
    Project 3D hands to 2D image plane over the original frames
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output video path
        raw_frames: original frames
        fps: int, frame rate
        image_size: tuple, (width, height)
        render_mode: str, 'skeleton', 'contour', 'heatmap'
    """
    
    # Joint connections definition
    connections = [
        (0, 1, 2, 3, 4),      # thumb
        (0, 5, 6, 7, 8),      # index
        (0, 9, 10, 11, 12),   # middle
        (0, 13, 14, 15, 16),  # ring
        (0, 17, 18, 19, 20)   # pinky
    ]
    palm_connections = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
    
    frames = []
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    if raw_frames is not None:
        height = raw_frames[0].shape[0]
        width = raw_frames[0].shape[1]
        image_size = (height, width)
    else:
        height, width = image_size
    print(f"Creating 2D projection video with {max_frames} frames...")
    
    for frame_idx in trange(max_frames, desc="Projecting frames"):
        raw_frame = raw_frames[frame_idx].copy()
        
        if render_mode == 'skeleton':
            # Skeleton mode
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            for side, hand_data in hands_data.items():
                if frame_idx >= hand_data['joints'].shape[0]:
                    continue
                    
                color = (255, 100, 100) if side == 'right' else (100, 100, 255)  
                joints = hand_data['joints'][frame_idx] # (J, 3) tensor
                
                # project 3D joints to 2D
                joints_2d = project_to_2d(joints, camera_intrinsic)
                valid_mask = get_valid_mask(joints_2d, image_size)
                
                # plot skeleton connections
                for chain in connections + palm_connections:
                    for i in range(len(chain)-1):
                        start_idx, end_idx = chain[i], chain[i+1]
                        if valid_mask[start_idx] and valid_mask[end_idx]:
                            start = tuple(joints_2d[start_idx])
                            end = tuple(joints_2d[end_idx])
                            cv2.line(img, start, end, color, 3)
                
                # plot joints
                for i, (x, y) in enumerate(joints_2d):
                    if valid_mask[i]:
                        radius = 8 if i == 0 else 5  # bigger radius for wrist
                        cv2.circle(img, (x, y), radius, color, -1)
                        cv2.circle(img, (x, y), radius, (0, 0, 0), 1)  # black outline
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.addWeighted(raw_frame, 0.5, img, 0.5, 0)
            frames.append(img)
            
        elif render_mode == 'heatmap':
            # Heatmap mode
            heatmap = np.zeros((height, width), dtype=np.float32)
            
            for side, hand_data in hands_data.items():
                if frame_idx >= hand_data['joints'].shape[0]:
                    continue
                
                joints = hand_data['joints'][frame_idx]
                joints_2d = project_to_2d(joints, camera_intrinsic)
                
                # accumulate heatmap for each joint
                for x, y in joints_2d:
                    if 0 <= x < width and 0 <= y < height:
                        sigma = 15
                        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
                        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                        heatmap += gaussian
            
            # normalize heatmap to 0-255 range
            if heatmap.max() > 0:
                heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            else:
                heatmap = heatmap.astype(np.uint8)
            img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.addWeighted(raw_frame, 0.5, img, 0.5, 0)
            frames.append(img)
        
        elif render_mode == 'contour':
            # Contour mode
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            for side, hand_data in hands_data.items():
                if frame_idx >= hand_data['verts'].shape[0]:
                    continue
                    
                color = (255, 0, 0) if side == 'right' else (0, 0, 255)  # BGR format
                fill_color = (255, 200, 200) if side == 'right' else (200, 200, 255)  # Lighter fill
                verts = hand_data['verts'][frame_idx]
                
                # Project vertices to 2D
                verts_2d = project_to_2d(verts, camera_intrinsic)
                valid_mask = get_valid_mask(verts_2d, image_size)
                
                if valid_mask.sum() > 10:  # Ensure enough valid points
                    valid_verts = verts_2d[valid_mask]
                    
                    # Compute convex hull as hand contour
                    hull = cv2.convexHull(valid_verts.astype(np.int32))
                    
                    # Fill the contour with semi-transparent color
                    cv2.fillPoly(img, [hull], fill_color)
                    # Draw contour outline
                    cv2.polylines(img, [hull], True, color, 3)
                
                # Also draw skeleton for better visualization
                joints = hand_data['joints'][frame_idx]
                joints_2d = project_to_2d(joints, camera_intrinsic)
                joint_valid_mask = get_valid_mask(joints_2d, image_size)
                
                # Draw simplified skeleton
                for chain in connections:
                    for i in range(len(chain)-1):
                        start_idx, end_idx = chain[i], chain[i+1]
                        if joint_valid_mask[start_idx] and joint_valid_mask[end_idx]:
                            start = tuple(joints_2d[start_idx])
                            end = tuple(joints_2d[end_idx])
                            cv2.line(img, start, end, color, 2)
                
                # Draw key joint points
                for i, (x, y) in enumerate(joints_2d):
                    if joint_valid_mask[i]:
                        radius = 4 if i == 0 else 3
                        cv2.circle(img, (x, y), radius, color, -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.addWeighted(raw_frame, 0.5, img, 0.5, 0)
            frames.append(img)
        
        elif render_mode == 'mesh':
            # Mesh projection mode    
            for hand_side, hand_data in hands_data.items():
                if frame_idx >= hand_data['verts'].shape[0]:
                    continue
                
                # Get 3D data for current frame
                vertices_3d = hand_data['verts'][frame_idx]  # (N, 3)
                faces = hand_data['faces']  # (F, 3)
                
                # Project vertices to 2D
                vertices_2d = project_vertices_to_2d(vertices_3d, camera_intrinsic)
                
                # Check vertex visibility
                valid_vertices = check_vertices_in_bounds(vertices_2d, image_size)
                visible_vertex_count = valid_vertices.sum()
                
                # Only render if enough vertices are visible
                if visible_vertex_count > 50:  # Increased threshold for better quality
                    # Calculate face depths
                    face_depths, _ = get_face_visibility_and_depth(vertices_3d, faces)
                    
                    # Render the mesh
                    img = render_hand_mesh(raw_frame, vertices_2d, faces, face_depths, hand_side)
                    
                    # Optional: Add key joint markers
                    joints_3d = hand_data['joints'][frame_idx]
                    joints_2d = project_vertices_to_2d(joints_3d, camera_intrinsic)
                    joints_valid = check_vertices_in_bounds(joints_2d, image_size)
                    
                    # Draw key joints (wrist and fingertips)
                    key_joints = [0, 4, 8, 12, 16, 20]
                    joint_color = (0, 50, 200) if hand_side == 'right' else (200, 50, 0)
                    
                    for joint_idx in key_joints:
                        if joint_idx < len(joints_2d) and joints_valid[joint_idx]:
                            center = tuple(joints_2d[joint_idx])
                            radius = 8 if joint_idx == 0 else 5
                            cv2.circle(img, center, radius, joint_color, -1)
                            cv2.circle(img, center, radius, (0, 0, 0), 1)  # Black border
            
                frames.append(img)
        else:
            raise ValueError(f"Unsupported render mode: {render_mode}")
    
    print(f"Saving 2D video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("2D video saved successfully!")

try:
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import pyrender
    import trimesh
except ImportError:
    print("PyRender or Trimesh not installed. Skipping 3D rendering functions.")
    pyrender = None
    trimesh = None  

def create_raymond_lights():
    """Create raymond lighting for the scene"""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    
    return nodes


def hand_2d_render_clean(hands_data, camera_intrinsic, output_path, raw_frames=None, 
                        fps=30, render_res=(640, 480), mesh_colors=None, 
                        background_color=(0.0, 0.0, 0.0), blend_alpha=0.9, valid_region=None):
    """
    Clean version of hand_2d_render without verbose progress output.
    """
    # Same implementation as hand_2d_render but with silent progress
    import pyrender
    import trimesh
    import numpy as np
    import imageio
    
    # Default mesh colors
    if mesh_colors is None:
        mesh_colors = {
            'left': (1.0, 0.5, 0.0),  # 橙色
            'right': (0.0, 0.5, 1.0)  # 蓝色
        }
    
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    
    if raw_frames is not None:
        if len(raw_frames.shape) == 3:
            raw_frames = np.stack([raw_frames] * max_frames, axis=0)
        width, height = raw_frames.shape[-2], raw_frames.shape[-3]
    else:    
        width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    frames = []
    
    # Silent rendering loop with minimal progress indication
    for frame_idx in range(max_frames):
        # Show progress for longer videos (>30 frames)
        if max_frames > 30 and frame_idx % max(1, max_frames // 4) == 0 and frame_idx > 0:
            progress = int(frame_idx / max_frames * 100)
            print(f"\r  Progress: {progress}%", end='', flush=True)
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height,
            point_size=1.0
        )
        
        scene = pyrender.Scene(
            bg_color=[*background_color, 1.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # Add meshes for each hand
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            vertices = hand_data['verts'][frame_idx].numpy()
            faces = hand_data['faces'].numpy()
            
            mesh_color = mesh_colors.get(side, (1.0, 0.8, 0.7))
            vertex_colors = np.array([(*mesh_color, 1.0)] * vertices.shape[0])
            
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
            
            transform = np.eye(4)
            transform[1, 1] = -1  # Flip Y
            transform[2, 2] = -1  # Flip Z
            tri_mesh.apply_transform(transform)
            
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.8,
                alphaMode='OPAQUE',
                baseColorFactor=(*mesh_color, 1.0)
            )
            
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
            scene.add(mesh)
        
        # Add camera
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=np.eye(4))
        
        # Add light
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
        scene.add(light, pose=np.eye(4))
        
        # Render
        color, depth = renderer.render(scene)
        renderer.delete()
        
        # Convert and blend with background
        if raw_frames is not None:
            background = raw_frames[frame_idx]
            # Blend based on alpha
            mask = (color.sum(axis=2) > 0).astype(np.float32)
            mask = np.stack([mask] * 3, axis=2)
            color = color.astype(np.float32) / 255.0
            background = background.astype(np.float32) / 255.0
            blended = color * mask * blend_alpha + background * (1 - mask * blend_alpha)
            color = (blended * 255).astype(np.uint8)
        
        frames.append(color)
    
    # Clear progress line if shown
    if max_frames > 30:
        print(f"\r  Progress: 100%", end='', flush=True)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps, quality=8)


def hand_2d_render(hands_data, camera_intrinsic, output_path, raw_frames=None, 
                   fps=30, render_res=(640, 480), mesh_colors=None, 
                   background_color=(0.0, 0.0, 0.0), blend_alpha=0.9, valid_region=None):
    """
    Render hands using PyRender with high-quality mesh rendering
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output video path
        raw_frames: optional, (N, H, W, 3) background frames
        fps: int, frame rate
        render_res: tuple, (width, height) for rendering resolution
        mesh_colors: dict, colors for each hand {'left': (r,g,b), 'right': (r,g,b)}
        background_color: tuple, (r,g,b) background color
        blend_alpha: float, blending alpha with background frames
    """
    
    # Default mesh colors - 左手橙色，右手蓝色
    if mesh_colors is None:
        mesh_colors = {
            'left': (1.0, 0.5, 0.0),  # 橙色
            'right': (0.0, 0.5, 1.0)  # 蓝色
        }
    
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    
    if raw_frames is not None:
        if len(raw_frames.shape) == 3:
            # If raw_frames is a single frame, expand it to match the number of frames
            raw_frames = np.stack([raw_frames] * max_frames, axis=0)
        width, height = raw_frames.shape[-2], raw_frames.shape[-3]
    else:    
        # Get rendering parameters
        width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # Get total frames
    print(f"Creating PyRender 2D video with {max_frames} frames...")
    
    frames = []
    
    for frame_idx in trange(max_frames, desc="PyRender rendering"):
        # Initialize renderer for this frame
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height,
            point_size=1.0
        )
        
        # Create scene
        scene = pyrender.Scene(
            bg_color=[*background_color, 1.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # Add meshes for each hand
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            # Get vertices and faces for current frame
            vertices = hand_data['verts'][frame_idx].numpy()  # (V, 3)
            faces = hand_data['faces'].numpy()  # (F, 3)
            
            # Create trimesh
            mesh_color = mesh_colors.get(side, (1.0, 0.8, 0.7))
            vertex_colors = np.array([(*mesh_color, 1.0)] * vertices.shape[0])
            
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
            
            # Apply coordinate system transformation (PyRender uses different convention)
            transform = np.eye(4)
            transform[1, 1] = -1  # Flip Y
            transform[2, 2] = -1  # Flip Z
            tri_mesh.apply_transform(transform)
            
            # Create PyRender mesh with material
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.8,
                alphaMode='OPAQUE',
                baseColorFactor=(*mesh_color, 1.0)
            )
            
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
            scene.add(py_mesh, name=f'hand_{side}')
        
        # Set up camera
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )
        
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        # Additional point light at camera position
        point_light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
        light_pose = np.eye(4)
        light_pose[2, 3] = 1.0  # Move light slightly forward
        scene.add(point_light, pose=light_pose)
        
        # Render
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Rendering error at frame {frame_idx}: {e}")
            # Create blank frame on error
            color = np.zeros((height, width, 4), dtype=np.float32)
            color[:, :, 3] = 1.0  # Full alpha
        
        renderer.delete()
        
        # Process the rendered frame
        if raw_frames is not None and frame_idx < len(raw_frames):
            # Blend with background frame
            background = raw_frames[frame_idx].astype(np.float32) / 255.0
            
            # Resize background if needed
            if background.shape[:2] != (height, width):
                background = cv2.resize(
                    background, (width, height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            mask = depth > 0
            hand_color = color[mask]
            background[mask] = (hand_color[:, :3] *hand_color[:, 3:4] * blend_alpha + background[mask] * (1 - hand_color[:, 3:4] * blend_alpha))
            final_frame = (background * 255).astype(np.uint8)
        else:
            final_frame = (color[:, :, :3] * 255).astype(np.uint8)
        
        # final_frame.shape, final_frame.dtype
        if valid_region is not None:
            # Apply valid region mask if provided
            valid_frame = final_frame[valid_region[0]+2:valid_region[1]-3, valid_region[2]+2:valid_region[3]-3]
            # valid_mask = np.zeros((height, width), dtype=bool)
            # valid_mask[:, valid_region[0]:valid_region[1]+1, valid_region[2]:valid_region[3]+1] = True
            # valid_frame = final_frame.copy()
            # valid_frame = valid_frame[valid_mask]
            # valid frame shape: valid_frame.shape
        else:
            valid_frame = final_frame
        frames.append(valid_frame)

    # Save video
    print(f"Saving PyRender video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("PyRender video saved successfully!")

def hand_2d_render_old(hands_data, camera_intrinsic, output_path, raw_frames=None, 
                   fps=30, render_res=(640, 480), mesh_colors=None, 
                   background_color=(0.0, 0.0, 0.0), blend_alpha=0.8):
    """
    Render hands using PyRender with high-quality mesh rendering
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output video path
        raw_frames: optional, (N, H, W, 3) background frames
        fps: int, frame rate
        render_res: tuple, (width, height) for rendering resolution
        mesh_colors: dict, colors for each hand {'left': (r,g,b), 'right': (r,g,b)}
        background_color: tuple, (r,g,b) background color
        blend_alpha: float, blending alpha with background frames
    """
    
    # Default mesh colors  
    if mesh_colors is None:
        mesh_colors = {
            'right': (1.0, 0.8, 0.7),  # Skin tone for right hand
            'left': (0.9, 0.7, 0.6)    # Slightly different for left hand
        }
    
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    
    if raw_frames is not None:
        if len(raw_frames.shape) == 3:
            # If raw_frames is a single frame, expand it to match the number of frames
            raw_frames = np.stack([raw_frames] * max_frames, axis=0)
        width, height = raw_frames.shape[-2], raw_frames.shape[-3]
    else:    
        # Get rendering parameters
        width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # Get total frames
    print(f"Creating PyRender 2D video with {max_frames} frames...")
    
    frames = []
    
    for frame_idx in trange(max_frames, desc="PyRender rendering"):
        # Initialize renderer for this frame
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height,
            point_size=1.0
        )
        
        # Create scene
        scene = pyrender.Scene(
            bg_color=[*background_color, 1.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # Add meshes for each hand
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            # Get vertices and faces for current frame
            vertices = hand_data['verts'][frame_idx].numpy()  # (V, 3)
            faces = hand_data['faces'].numpy()  # (F, 3)
            
            # Create trimesh
            mesh_color = mesh_colors.get(side, (1.0, 0.8, 0.7))
            vertex_colors = np.array([(*mesh_color, 1.0)] * vertices.shape[0])
            
            # Flip faces for left hand to correct normal direction
            # if side == 'left':
            #     faces = faces[:, [0, 2, 1]]
            
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
            
            # Apply coordinate system transformation (PyRender uses different convention)
            transform = np.eye(4)
            transform[1, 1] = -1  # Flip Y
            transform[2, 2] = -1  # Flip Z
            tri_mesh.apply_transform(transform)
            
            # Create PyRender mesh with material
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.8,
                alphaMode='OPAQUE',
                baseColorFactor=(*mesh_color, 1.0)
            )
            
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
            scene.add(py_mesh, name=f'hand_{side}')
        
        # Set up camera
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )
        
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        # Additional point light at camera position
        point_light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
        light_pose = np.eye(4)
        light_pose[2, 3] = 1.0  # Move light slightly forward
        scene.add(point_light, pose=light_pose)
        
        # Render
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Rendering error at frame {frame_idx}: {e}")
            # Create blank frame on error
            color = np.zeros((height, width, 4), dtype=np.float32)
            color[:, :, 3] = 1.0  # Full alpha
        
        renderer.delete()
        
        # Process the rendered frame
        if raw_frames is not None and frame_idx < len(raw_frames):
            # Blend with background frame
            background = raw_frames[frame_idx].astype(np.float32) / 255.0
            
            # Resize background if needed
            if background.shape[:2] != (height, width):
                background = cv2.resize(
                    background, (width, height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            mask = depth > 0
            hand_color = color[mask]
            background[mask] = (hand_color[:, :3] *hand_color[:, 3:4] * blend_alpha + background[mask] * (1 - hand_color[:, 3:4] * blend_alpha))
            final_frame = (background * 255).astype(np.uint8)
        else:
            final_frame = (color[:, :, :3] * 255).astype(np.uint8)
        
        frames.append(final_frame)

    # Save video
    print(f"Saving PyRender video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("PyRender video saved successfully!")

def hand_2d_render_stacked(hands_data, camera_intrinsic, output_path, raw_frames=None, 
                            fps=30, render_res=(640, 480), background_color=(0.0, 0.0, 0.0), 
                            frames_per_image=4):
    """
    渲染手部并将多个时间步叠加在同一张图上。
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output video path
        raw_frames: optional, (N, H, W, 3) background frames
        fps: int, frame rate
        render_res: tuple, (width, height) for rendering resolution
        background_color: tuple, (r,g,b) background color
        frames_per_image: int, number of time steps to render in each image
    """
    
    # Set left and right hand colors
    mesh_base_colors = {
        'left': (1.0, 0.5, 0.0),  # 橙色
        'right': (0.0, 0.5, 1.0)  # 蓝色
    }
    
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    
    if raw_frames is not None:
        if len(raw_frames.shape) == 3:
            raw_frames = np.stack([raw_frames] * max_frames, axis=0)
        width, height = raw_frames.shape[-2], raw_frames.shape[-3]
    else:    
        width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # 计算输出视频的总帧数
    output_frames = max(1, max_frames // frames_per_image)
    print(f"创建包含{output_frames}张图像的视频，每张图像包含{frames_per_image}个时间步...")
    
    frames = []
    
    for output_idx in trange(output_frames, desc="渲染叠加视频"):
        # 为当前输出图像选择均匀分布的帧索引
        start_idx = output_idx * frames_per_image
        end_idx = min(start_idx + frames_per_image, max_frames)
        if end_idx <= start_idx:
            break
            
        # 如果剩余帧数少于frames_per_image，调整步长确保均匀分布
        if end_idx - start_idx < frames_per_image:
            frame_indices = list(range(start_idx, end_idx))
        else:
            frame_indices = [start_idx + i * (end_idx - start_idx) // frames_per_image 
                             for i in range(frames_per_image)]
        
        # 初始化最终图像
        final_frame = np.zeros((height, width, 3), dtype=np.float32)
        final_frame[:] = background_color
        
        # 初始化渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height,
            point_size=1.0
        )
        
        # 渲染每个选定的时间步
        for time_idx, frame_idx in enumerate(frame_indices):
            # 创建当前时间步的场景
            time_scene = pyrender.Scene(
                bg_color=[*background_color, 0.0],  # 透明背景
                ambient_light=(0.3, 0.3, 0.3)
            )
            
            # 添加手部网格
            for side, hand_data in hands_data.items():
                if frame_idx >= hand_data['verts'].shape[0]:
                    continue
                    
                # 获取当前帧的顶点和面
                vertices = hand_data['verts'][frame_idx].numpy()
                faces = hand_data['faces'].numpy()
                
                # 获取基础颜色
                base_color = mesh_base_colors.get(side)
                
                # 计算当前时间步的颜色强度（随时间变深）
                alpha = 0.5 + 0.5 * (time_idx / (frames_per_image - 1))  # 从0.5到1.0
                mesh_color = tuple(c for c in base_color)
                
                # 创建trimesh
                vertex_colors = np.array([(*mesh_color, alpha)] * vertices.shape[0])
                
                tri_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors
                )
                
                # 应用坐标系变换
                transform = np.eye(4)
                transform[1, 1] = -1  # 翻转Y
                transform[2, 2] = -1  # 翻转Z
                tri_mesh.apply_transform(transform)
                
                # 创建PyRender网格
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.1,
                    roughnessFactor=0.8,
                    alphaMode='BLEND',  # 使用BLEND支持透明度
                    baseColorFactor=(*mesh_color, alpha)
                )
                
                py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
                time_scene.add(py_mesh, name=f'hand_{side}_{time_idx}')
            
            # 设置摄像机
            camera = pyrender.IntrinsicsCamera(
                fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
            )
            
            camera_pose = np.eye(4)
            time_scene.add(camera, pose=camera_pose)
            
            # 添加光照
            light_nodes = create_raymond_lights()
            for node in light_nodes:
                time_scene.add_node(node)
            
            # 添加点光源
            point_light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
            light_pose = np.eye(4)
            light_pose[2, 3] = 1.0
            time_scene.add(point_light, pose=light_pose)
            
            # 渲染当前时间步
            try:
                color, depth = renderer.render(time_scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0
            except Exception as e:
                print(f"渲染错误，时间步 {frame_idx}: {e}")
                color = np.zeros((height, width, 4), dtype=np.float32)
            
            # 合并手部到最终图像
            mask = color[:, :, 3] > 0
            final_frame[mask] = color[mask, :3] * color[mask, 3:4] + final_frame[mask] * (1 - color[mask, 3:4])
        
        renderer.delete()
        
        # 添加到输出帧
        frames.append((final_frame * 255).astype(np.uint8))

    # 保存视频
    print(f"保存叠加视频到 {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("叠加视频保存成功!")


def render_hands_transparent_png_sequence(hands_data, camera_intrinsic, output_dir, 
                                         render_res=(1920, 1080), prefix="frame_"):
    """
    渲染带透明背景的手部图片序列，每帧保存为单独的PNG文件。
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_dir: str, 输出目录路径
        render_res: tuple, (width, height) 渲染分辨率
        prefix: str, 文件名前缀
    """
    
    # Set left and right hand colors
    mesh_colors = {
        'left': (1.0, 0.5, 0.0),  # 橙色
        'right': (0.0, 0.5, 1.0)  # 蓝色
    }
    
    # 获取最大帧数
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"创建透明背景手部渲染PNG序列，共 {max_frames} 帧...")
    
    for frame_idx in trange(max_frames, desc="渲染手部PNG"):
        # 初始化渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height
        )
        
        # 创建场景 - 设置完全透明背景
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],  # RGBA全0表示完全透明
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # 添加手部网格
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            # 获取当前帧的顶点和面
            vertices = hand_data['verts'][frame_idx].numpy()
            faces = hand_data['faces'].numpy()
            
            # 获取手部颜色
            mesh_color = mesh_colors.get(side)
            
            # 创建trimesh
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces
            )
            
            # 应用坐标系变换
            transform = np.eye(4)
            transform[1, 1] = -1  # 翻转Y
            transform[2, 2] = -1  # 翻转Z
            tri_mesh.apply_transform(transform)
            
            # 创建材质
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                roughnessFactor=0.6,
                baseColorFactor=(*mesh_color, 1.0)
            )
            
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
            scene.add(py_mesh, name=f'hand_{side}')
        
        # 设置摄像机
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )
        
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # 添加光照
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.eye(4)
        scene.add(light, pose=light_pose)
        
        # 添加点光源
        point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.eye(4)
        light_pose[2, 3] = 2.0
        scene.add(point_light, pose=light_pose)
        
        # 渲染 - 使用RGBA模式
        try:
            # 必须使用flags=pyrender.RenderFlags.RGBA参数来渲染透明通道
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            # 确保color是RGBA格式
            if color.shape[2] == 4:  # 已经是RGBA
                rgba_image = color
            else:  # 如果是RGB，添加Alpha通道
                rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
                rgba_image[:,:,:3] = color
                rgba_image[:,:,3] = 255  # 全不透明
            
            # 保存PNG图片
            filename = f"{prefix}{frame_idx:06d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 使用imageio保存PNG，保留透明度
            imageio.imwrite(filepath, rgba_image)
            
        except Exception as e:
            print(f"渲染错误，帧 {frame_idx}: {e}")
        
        renderer.delete()

    print(f"透明背景手部PNG序列已保存至目录: {output_dir}")
    
    
def render_enhanced_hands_video(hands_data, camera_intrinsic, output_path, 
                                fps=30, render_res=(1920, 1080), background_color=(0.0, 0.0, 0.0)):
    """
    渲染高质量手部视频，左右手不同颜色，强化光照效果。
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output video path
        fps: int, frame rate
        render_res: tuple, (width, height) for rendering resolution
        background_color: tuple, (r,g,b) background color
    """
    
    # Set left and right hand colors
    mesh_colors = {
        'left': (1.0, 0.5, 0.0),  # 橙色
        'right': (0.0, 0.5, 1.0)  # 蓝色
    }
    
    # 获取最大帧数
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    print(f"创建包含 {max_frames} 帧的增强手部渲染视频...")
    
    frames = []
    
    for frame_idx in trange(max_frames, desc="渲染增强手部"):
        # 初始化渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height,
            point_size=1.0
        )
        
        # 创建场景
        scene = pyrender.Scene(
            bg_color=[*background_color, 1.0],
            ambient_light=(0.2, 0.2, 0.2)  # 稍微降低环境光，增强光照对比度
        )
        
        # 添加手部网格
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            # 获取当前帧的顶点和面
            vertices = hand_data['verts'][frame_idx].numpy()
            faces = hand_data['faces'].numpy()
            
            # 获取手部颜色
            mesh_color = mesh_colors.get(side)
            
            # 创建trimesh
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces
            )
            
            # 应用坐标系变换
            transform = np.eye(4)
            transform[1, 1] = -1  # 翻转Y
            transform[2, 2] = -1  # 翻转Z
            tri_mesh.apply_transform(transform)
            
            # 创建高质量材质
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,       # 适度金属感
                roughnessFactor=0.5,      # 中等粗糙度，不太光滑也不太粗糙
                alphaMode='OPAQUE',
                baseColorFactor=(*mesh_color, 1.0)
            )
            
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
            scene.add(py_mesh, name=f'hand_{side}')
        
        # 设置摄像机
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )
        
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # 添加增强光照 - Raymond光照
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            # 增强光照强度
            node.light.intensity = 1.5
            scene.add_node(node)
        
        # 添加额外的填充光
        fill_light = pyrender.DirectionalLight(color=[1.0, 0.9, 0.8], intensity=0.6)  # 暖色填充光
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0.0, -1.0, 0.5]  # 从下方打光
        scene.add(fill_light, pose=light_pose)
        
        # 添加高光点光源
        key_light = pyrender.SpotLight(
            color=[1.0, 1.0, 1.0], 
            intensity=2.0, 
            innerConeAngle=np.pi/6,
            outerConeAngle=np.pi/3
        )
        key_pose = np.eye(4)
        key_pose[:3, 3] = [0.5, 0.5, 2.0]  # 从右上方照射
        scene.add(key_light, pose=key_pose)
        
        # 渲染
        # try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        color = color.astype(np.uint8)
        # except Exception as e:
            # print(f"渲染错误，帧 {frame_idx}: {e}")
            # color = np.zeros((height, width, 3), dtype=np.uint8)
            # color[:] = [int(c * 255) for c in background_color]
        
        renderer.delete()
        frames.append(color)

    # 保存视频
    print(f"保存增强手部渲染视频到 {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("增强手部渲染视频保存成功!")
    
    return frames

def render_stacked_hands_image(hands_data, camera_intrinsic, output_path, raw_frame=None, 
                                render_res=(640, 480), background_color=(0.0, 0.0, 0.0), 
                                frames_to_sample=3):
    """
    渲染一张包含多个时间步手部的图片，每个手使用不透明的渐变颜色。
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output image path
        raw_frame: optional, (H, W, 3) background frame
        render_res: tuple, (width, height) for rendering resolution
        background_color: tuple, (r,g,b) background color
        frames_to_sample: int, number of frames to sample uniformly from the sequence
    """
    
    # Set base colors for left and right hands
    mesh_base_colors = {
        'left': (1.0, 0.5, 0.0),  # 橙色
        'right': (0.0, 0.5, 1.0)  # 蓝色
    }
    
    # 获取最大帧数
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    
    if raw_frame is not None:
        width, height = raw_frame.shape[1], raw_frame.shape[0]
    else:    
        width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # 计算均匀采样的帧索引
    if max_frames < frames_to_sample:
        frame_indices = list(range(max_frames))
    else:
        frame_indices = [int(i * (max_frames - 1) / (frames_to_sample - 1)) for i in range(frames_to_sample)]
    
    print(f"从序列中均匀采样 {len(frame_indices)} 帧: {frame_indices}")
    
    # 初始化最终图像
    if raw_frame is not None:
        final_frame = raw_frame.astype(np.float32) / 255.0
        if final_frame.shape[:2] != (height, width):
            final_frame = cv2.resize(final_frame, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        final_frame = np.zeros((height, width, 3), dtype=np.float32)
        final_frame[:] = background_color
    
    # 初始化渲染器
    renderer = pyrender.OffscreenRenderer(
        viewport_width=width, 
        viewport_height=height,
        point_size=1.0
    )
    
    # 创建Z-buffer来跟踪深度
    z_buffer = np.ones((height, width)) * np.inf
    
    # 按时间步顺序渲染（从早到晚）
    for time_idx, frame_idx in enumerate(frame_indices):
        # 为每个时间步创建单独的场景
        scene = pyrender.Scene(
            bg_color=[*background_color, 0.0],  # 透明背景
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # 添加手部网格
        any_hand_added = False
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            any_hand_added = True
            
            # 获取当前帧的顶点和面
            vertices = hand_data['verts'][frame_idx].numpy()
            faces = hand_data['faces'].numpy()
            
            # 获取基础颜色
            base_color = mesh_base_colors.get(side)
            
            # 计算当前时间步的颜色强度（从浅到深）
            intensity = 1.0 - 0.3 * (time_idx / (len(frame_indices) - 1))  # 从30%亮度到100%亮度
            color = tuple(c * intensity for c in base_color)
            
            # 创建trimesh
            vertex_colors = np.array([(*color, 1.0)] * vertices.shape[0])  # 完全不透明
            
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors
            )
            
            # 应用坐标系变换
            transform = np.eye(4)
            transform[1, 1] = -1  # 翻转Y
            transform[2, 2] = -1  # 翻转Z
            tri_mesh.apply_transform(transform)
            
            # 创建PyRender网格 - 使用不透明材质
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.8,
                alphaMode='OPAQUE',  # 完全不透明
                baseColorFactor=(*color, 1.0)
            )
            
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
            scene.add(py_mesh, name=f'hand_{side}_{time_idx}')
        
        if not any_hand_added:
            continue
            
        # 设置摄像机
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )
        
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # 添加光照
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        # 添加点光源
        point_light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
        light_pose = np.eye(4)
        light_pose[2, 3] = 1.0
        scene.add(point_light, pose=light_pose)
        
        # 渲染当前时间步
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            
            # 创建有效深度掩码（考虑z-buffer）
            valid_mask = (depth < z_buffer) & (color[:,:,3] > 0)
            
            # 更新z-buffer和最终图像
            z_buffer[valid_mask] = depth[valid_mask]
            final_frame[valid_mask] = color[valid_mask, :3]  # 直接覆盖，不混合
            
        except Exception as e:
            print(f"渲染错误，时间步 {frame_idx}: {e}")
    
    renderer.delete()
    
    # 保存最终图像
    final_image = (final_frame * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"渐变颜色手部图像已保存至 {output_path}")
    
    return final_image

from matplotlib import cm

def create_raymond_lights():
    # 创建Raymond光照设置的帮助函数
    thetas = np.pi * np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
    phis = np.pi * np.array([0.0, 2.0/3.0, 4.0/3.0])
    
    nodes = []
    
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        matrix = np.eye(4)
        matrix[:3, 0] = x
        matrix[:3, 1] = y
        matrix[:3, 2] = z
        matrix[:3, 3] = z * 3.0
        
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        node = pyrender.Node(light=light, matrix=matrix)
        nodes.append(node)
    
    return nodes

def hand_2d_heat_render(hands_data, camera_intrinsic, output_path, raw_frames=None, 
                   fps=30, render_res=(640, 480), colormap='jet',
                   palm_vertex_index=0, background_color=(0.0, 0.0, 0.0), blend_alpha=0.8):
    """
    Render hands using PyRender with heatmap coloring based on distance from palm
    
    Args:
        hands_data: dict {'left': {'verts': (N,V,3), 'joints': (N,J,3), 'faces': (F,3)}, ...}
        camera_intrinsic: (3, 3) camera intrinsic matrix
        output_path: str, output video path
        raw_frames: optional, (N, H, W, 3) background frames
        fps: int, frame rate
        render_res: tuple, (width, height) for rendering resolution
        colormap: str, matplotlib colormap name for heatmap (default: 'jet')
        palm_vertex_index: int, index of vertex to use as palm center (default: 0)
        background_color: tuple, (r,g,b) background color
        blend_alpha: float, blending alpha with background frames
    """
    # Get colormap function from matplotlib
    cmap = getattr(cm, colormap)
    
    max_frames = max(hand_data['verts'].shape[0] for hand_data in hands_data.values())
    
    if raw_frames is not None:
        if len(raw_frames.shape) == 3:
            # If raw_frames is a single frame, expand it to match the number of frames
            raw_frames = np.stack([raw_frames] * max_frames, axis=0)
        width, height = raw_frames.shape[-2], raw_frames.shape[-3]
    else:    
        # Get rendering parameters
        width, height = render_res
    
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
    
    # Get total frames
    print(f"Creating PyRender 2D video with {max_frames} frames...")
    
    frames = []
    
    for frame_idx in trange(max_frames, desc="PyRender rendering"):
        # Initialize renderer for this frame
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, 
            viewport_height=height,
            point_size=1.0
        )
        
        # Create scene
        scene = pyrender.Scene(
            bg_color=[*background_color, 1.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        
        # Add meshes for each hand
        for side, hand_data in hands_data.items():
            if frame_idx >= hand_data['verts'].shape[0]:
                continue
                
            # Get vertices and faces for current frame
            vertices = hand_data['verts'][frame_idx].numpy()  # (V, 3)
            faces = hand_data['faces'].numpy()  # (F, 3)
            
            # 计算热力图颜色 - 基于到手心的距离
            # 使用指定的掌心顶点索引或计算手部中心点
            if palm_vertex_index < vertices.shape[0]:
                palm_center = vertices[palm_vertex_index]  # 使用指定的掌心顶点
            else:
                palm_center = np.mean(vertices, axis=0)  # 如果索引无效，使用平均位置
                
            # 计算每个顶点到掌心的距离
            distances = np.linalg.norm(vertices - palm_center, axis=1)
            
            # 归一化距离到 [0,1] 范围
            if distances.max() > distances.min():
                distances_norm = (distances - distances.min()) / (distances.max() - distances.min())
            else:
                distances_norm = np.zeros_like(distances)
                
            # 应用colormap获取颜色
            vertex_colors = cmap(distances_norm)
            
            # Create trimesh
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors  # 使用热力图颜色
            )
            
            # Apply coordinate system transformation (PyRender uses different convention)
            transform = np.eye(4)
            transform[1, 1] = -1  # Flip Y
            transform[2, 2] = -1  # Flip Z
            tri_mesh.apply_transform(transform)
            
            # 创建PyRender网格（不需要额外的材质，因为我们已经设置了顶点颜色）
            py_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene.add(py_mesh, name=f'hand_{side}')
        
        # Set up camera
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=10.0
        )
        
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        # Additional point light at camera position
        point_light = pyrender.PointLight(color=np.ones(3), intensity=2.0)
        light_pose = np.eye(4)
        light_pose[2, 3] = 1.0  # Move light slightly forward
        scene.add(point_light, pose=light_pose)
        
        # Render
        try:
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Rendering error at frame {frame_idx}: {e}")
            # Create blank frame on error
            color = np.zeros((height, width, 4), dtype=np.float32)
            color[:, :, 3] = 1.0  # Full alpha
        
        renderer.delete()
        
        # Process the rendered frame
        if raw_frames is not None and frame_idx < len(raw_frames):
            # Blend with background frame
            background = raw_frames[frame_idx].astype(np.float32) / 255.0
            
            # Resize background if needed
            if background.shape[:2] != (height, width):
                background = cv2.resize(
                    background, (width, height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            mask = depth > 0
            hand_color = color[mask]
            background[mask] = hand_color[:, :3] 
            # (hand_color[:, :3] * hand_color[:, 3:4] * blend_alpha + 
            #                    background[mask] * (1 - hand_color[:, 3:4] * blend_alpha))
            final_frame = (background * 255).astype(np.uint8)
        else:
            final_frame = (color[:, :, :3] * 255).astype(np.uint8)
        
        frames.append(final_frame)

    # Save video
    print(f"Saving PyRender video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    print("PyRender video saved successfully!")
    
def world_hand_to_camera(hands_data, camera_extrinsic):
    hands_data = deepcopy(hands_data)
    hand_rot = hands_data['rot']
    hand_trans = hands_data['trans']
    
    for side in hand_rot.keys():
        side_rot = hand_rot[side]
        side_trans = hand_trans[side]
        side_rot_c, side_trans_c = world_to_camera(camera_extrinsic, rot=side_rot, trans=side_trans)
        hands_data['rot'][side] = side_rot_c
        hands_data['trans'][side] = side_trans_c
    return hands_data
    
def convert_reference(rot, trans, theta, beta, camera_ref, camera_seq, side=None):
    """
    Convert MANO parameters from initial camera coordinate system to target camera sequence
    
    Args:
        rot: dict, rotation parameters {'left': (N,3,3), 'right': (M,3,3)}
        trans: dict, translation parameters {'left': (N,3), 'right': (M,3)}
        theta: dict, pose parameters {'left': (N,45), 'right': (M,45)}
        beta: dict, shape parameters {'left': (N,10), 'right': (M,10)}
        camera_seq: (N, 4, 4) camera extrinsic sequence
        
    Returns:
        rot_new: dict, transformed rotation parameters
        trans_new: dict, transformed translation parameters
        theta: dict, pose parameters (unchanged)
        beta: dict, shape parameters (unchanged)
    """
    rot_new = {}
    trans_new = {}
    
    if not isinstance(rot, dict):
        if side is None:
            side = 'right'
        rot = {side: rot}
        trans = {side: trans}
        theta = {side: theta}
        beta = {side: beta}
        
    # the reference camera is the first one in the sequence
    # reference camera rotation and translation to the world frame
    R_ref_w = camera_ref[:3, :3]  
    t_ref_w = camera_ref[:3, 3]   
    
    for side in rot.keys():
        rot_side_new = []
        trans_side_new = []
        
        for i in range(camera_seq.shape[0]):
            if i < rot[side].shape[0]:
                # target camera rotation and translation to the world frame
                R_target_w = camera_seq[i, :3, :3]  
                t_target_w = camera_seq[i, :3, 3]   
                R_w_target = R_target_w.T
                
                # Convert rotation from reference camera to target camera
                rot_w = R_ref_w @ rot[side][i]  # (3, 3)
                rot_transformed = R_w_target @ rot_w
                
                # Convert translation from reference camera to target camera
                t_w = t_ref_w + R_ref_w @ trans[side][i]  # (3,)
                t_transformed = R_w_target @ (t_w - t_target_w)
                
                rot_side_new.append(rot_transformed)
                trans_side_new.append(t_transformed)
        
        rot_new[side] = torch.stack(rot_side_new) if rot_side_new else rot[side]
        trans_new[side] = torch.stack(trans_side_new) if trans_side_new else trans[side]
    
    return {
        'rot': rot_new,
        'trans': trans_new,
        'theta': theta,  # unchanged
        'beta': beta,    # unchanged
    }
    
def trans_convert_reference(hand_data, camera_ref, camera_seq):
    """
    Convert MANO parameters from initial camera coordinate system to target camera sequence
    
    Args:
        hand_data: dict, containing
            - 'left': {'joints': (N, J, 3), 'verts': (N, V, 3), 'faces': (F, 3)}
            - 'right': {'joints': (M, J, 3), 'verts': (M, V, 3), 'faces': (F, 3)}
        
        camera_ref: (4, 4) camera extrinsic matrix for reference camera
        camera_seq: (4, 4) camera extrinsics for target sequence
        
    Returns:
        new_hand_data: dict, transformed hand data with updated joint and vertex positions
            - 'left': {'joints': (N, J, 3), 'verts': (N, V, 3), 'faces': (F, 3)}
            - 'right': {'joints': (M, J, 3), 'verts': (M, V, 3), 'faces': (F, 3)}
    """
    new_hand_data = {}
        
    # the reference camera is the first one in the sequence
    # reference camera rotation and translation to the world frame
    R_ref_w = camera_ref[:3, :3]  
    t_ref_w = camera_ref[:3, 3]   
    
    R_target_w = camera_seq[:3, :3]  
    t_target_w = camera_seq[:3, 3]
    R_w_target = R_target_w.T
    
    for side in hand_data.keys():
        joints = hand_data[side]['joints']
        verts = hand_data[side]['verts']
        faces = hand_data[side]['faces']
        
        joints_w = torch.bmm(R_ref_w[None, :, :].expand(joints.shape[0], -1, -1),
                             joints.transpose(1, 2)) + t_ref_w[None, :, None]  # (N, J, 3)
        joints_transformed = torch.bmm(R_w_target[None, :, :].expand(joints.shape[0], -1, -1),
                                       joints_w - t_target_w[None, :, None])  # (N, J, 3)
        # R_ref_w @ joints.transpose(1, 2) + t_ref_w[:, None]  # (N, J, 3)
        # joints_transformed = R_target_w @ joints_w.transpose(1, 2) - t_target_w[:, None]  # (N, J, 3)
        
        verts_w = torch.bmm(R_ref_w[None, :, :].expand(verts.shape[0], -1, -1), 
                            verts.transpose(1, 2)) + t_ref_w[None, :, None]  # (N, V, 3)
        verts_transformed = torch.bmm(R_w_target[None, :, :].expand(verts.shape[0], -1, -1),
                                      verts_w - t_target_w[None, :, None])  # (N, V, 3)
        
        # verts_w = R_ref_w @ verts.transpose(1, 2) + t_ref_w[:, None]  # (N, V, 3)
        # verts_transformed = R_target_w @ verts_w.transpose(1, 2) - t_target_w[:, None]  # (N, V, 3)
        new_hand_data[side] = {
            'joints': joints_transformed.transpose(1, 2),  # (N, J, 3)
            'verts': verts_transformed.transpose(1, 2),    # (N, V, 3)
            'faces': faces,  # unchanged
        }
    
    return new_hand_data

def eval_visualize(rot, trans, theta, beta, camera_extrinsics,
                   camera_intrinsic, output_path, raw_frames,
                   side, fps):
    rot_c, trans_c, _ = world_to_camera(camera_extrinsics, rot=rot, trans=trans)
    hand_data = mano_forward(rot_c, trans_c, theta, beta, sides=[side], relative=True)
    hand_2d_render(hand_data, camera_intrinsic, output_path, raw_frames=raw_frames,
        fps=fps,  blend_alpha=0.5)

def vis_hand_plot(rot, trans, theta, beta, sides, image_paths, intrinsic_matrix, camera_extrinsics, output_dir, fps=30):
    print("vis_hand_plot")

    rot_c, trans_c = {}, {}
    for side in sides:
        # print(rot[side].device, trans[side].device, camera_extrinsics.device)
        rot_c[side], trans_c[side], _ = world_to_camera(camera_extrinsics, rot=rot[side], trans=trans[side])
    hand_data = mano_forward(rot_c, trans_c, theta, beta, sides, relative=True)

    if not isinstance(image_paths, np.ndarray):
        raw_frames = []
        for path in image_paths:
            with Image.open(path) as img:
                raw_frames.append(np.array(img))
        raw_frames = np.array(raw_frames)
    else:
        raw_frames = image_paths
    # print(raw_frames.shape)
    # print(intrinsic_matrix)

    # print(raw_frames)
    hand_2d_render(hand_data, torch.Tensor(intrinsic_matrix), 
                    output_path=f"{output_dir}_render.mp4", raw_frames=raw_frames, 
                    fps=fps, blend_alpha=0.5)

    # hand_render(hand_data, output_path=f"{output_dir}.mp4", fps=fps)

def vis_hand_plot_v2(hand_data, image_paths, intrinsic_matrix, output_dir, fps=30):
    print("vis_hand_plot_v2")
    # print(image_paths)
    if not isinstance(image_paths, np.ndarray):
        raw_frames = []
        for path in image_paths:
            with Image.open(path) as img:
                raw_frames.append(np.array(img))
        raw_frames = np.array(raw_frames)
        # raw_frames.shape
    else:
        raw_frames = image_paths

    hand_2d_render(hand_data, torch.Tensor(intrinsic_matrix), 
                output_path=f"{output_dir}_render.mp4", raw_frames=raw_frames, 
                fps=fps, blend_alpha=0.5)
