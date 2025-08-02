import torch
import numpy as np
from scipy import linalg
from .misc import rank0_print


@torch.jit.script
def procrustes_alignment(pred_joints: torch.Tensor, true_joints: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points pred_joints (B, N, 3) closest to a set of 3D points true_joints (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        pred_joints (torch.Tensor): First set of points of shape (B, N, 3).
        true_joints (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = pred_joints.shape[0]
    pred_joints = pred_joints.permute(0, 2, 1)
    true_joints = true_joints.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = pred_joints.mean(dim=2, keepdim=True)
    mu2 = true_joints.mean(dim=2, keepdim=True)
    X1 = pred_joints - mu1
    X2 = true_joints - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    pred_joints_hat = scale*torch.matmul(R, pred_joints) + t

    return pred_joints_hat.permute(0, 2, 1)

def calculate_mpjpe_mano(gt_joints, pred_joints):
    """Calculate Mean Per Joint Position Error (MPJPE) between predicted and ground truth joints.
    
    Args:
        gt_joints: num_poses x num_joints(21) x 3
        pred_joints: num_poses x num_joints(21) x 3
    
    Returns: MPJPE for each pose (num_poses tensor)
    """

    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"
    return 1000 * torch.norm(pred_joints - gt_joints, dim=-1).mean()


def calculate_mpjpe_relative(gt_joints, pred_joints):
    gt_fingers, pred_fingers = gt_joints[:, 1:, :], pred_joints[:, 1:, :]
    gt_wrist, pred_wrist = gt_joints[:, :1, :], pred_joints[:, :1, :]
    gt_relative = gt_fingers - gt_wrist
    pred_relative = pred_fingers - pred_wrist
    return 1000 * torch.norm(pred_relative - gt_relative, dim=-1).mean()


def calculate_p_mpjpe(gt_joints, pred_joints):
    """
    Compute P-MPJPE (Procrustes-aligned MPJPE) using PyTorch.
    
    Args:
        pred_joints: [T, 21, 3], predicted 3D joint positions
        gt_joints: [T, 21, 3], ground truth 3D joint positions
    
    Returns:
        p_mpjpe: scalar, the P-MPJPE error
    """
    '''
    # Step 1: Center the joints (remove translation)
    pred_centered = pred_joints - pred_joints.mean(dim=1, keepdim=True)
    gt_centered = gt_joints - gt_joints.mean(dim=1, keepdim=True)

    # Step 2: Compute optimal rotation (Procrustes alignment) Batch SVD: [T, 3, 3] -> [T, 3, 3]
    H = torch.bmm(pred_centered.transpose(1, 2), gt_centered)
    U, S, V = torch.linalg.svd(H)  # Batch SVD
    R = torch.bmm(V, U.transpose(1, 2))

    # Handle reflection case (ensure proper rotation)
    det = torch.det(R)
    mask = det < 0
    if mask.any():
        V[mask, -1, :] *= -1
        R[mask] = torch.bmm(V[mask], U[mask].transpose(1, 2))

    # Step 3: Apply rotation to predicted joints
    aligned_pred = torch.bmm(pred_centered, R)
    
    # Step 4: Compute error (MPJPE after alignment)
    return 1000 * torch.norm(aligned_pred - gt_centered, dim=-1).mean()  # [T, 21]
    '''
    pred_joints_hat = procrustes_alignment(pred_joints, gt_joints)
    return 1000 * torch.norm(pred_joints_hat - gt_joints, dim=-1).mean()
 

def euclidean_distance_matrix(matrix1, matrix2):
    """Compute pairwise Euclidean distances between two matrices.
    
    Args:
        matrix1: N1 x D tensor
        matrix2: N2 x D tensor
    
    Returns: N1 x N2 distance tensor where dist[i, j] = distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1], "Input matrices must have same feature dimension"
    
    # Compute squared norms for each vector
    sq_norm1 = torch.sum(matrix1**2, dim=1, keepdim=True)  # (N1, 1)
    sq_norm2 = torch.sum(matrix2**2, dim=1)                # (N2,)
    
    # Compute pairwise distances using matrix multiplication
    dists = sq_norm1 - 2 * torch.mm(matrix1, matrix2.t()) + sq_norm2.unsqueeze(0)
    dists = torch.sqrt(torch.clamp_min(dists, 0))  # Ensure numerical stability
    
    return dists


def calculate_mpjpe(gt_joints, pred_joints):
    """Calculate Mean Per Joint Position Error (MPJPE) between predicted and ground truth joints.
    
    Args:
        gt_joints: num_poses x num_joints(22) x 3
        pred_joints: num_poses x num_joints(22) x 3
    
    Returns: MPJPE for each pose (num_poses tensor)
    """

    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)

    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    return torch.norm(pred_joints - gt_joints, dim=-1).mean(-1) # num_poses x num_joints=22 -> num_poses


def calculate_top_k(mat: torch.Tensor, top_k: int) -> torch.Tensor:
    """Compute top-k matching accuracy from a sorted distance matrix.
    
    Args:
        mat: N x N sorted indices matrix (torch.Tensor)
        top_k: Maximum k to consider
    
    Returns: N x top_k boolean tensor indicating matches at each k
    """
    size = mat.size(0)
    gt_mat = torch.arange(size, device=mat.device).unsqueeze(1).expand(size, size)
    bool_mat = (mat == gt_mat)
    
    # Vectorized computation of cumulative matches
    correct_cumulative = bool_mat.cumsum(dim=1) > 0
    return correct_cumulative[:, :top_k]


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    """Calculate R-precision metrics between two embeddings.
    
    Args:
        embedding1: First set of embeddings (N x D torch.Tensor)
        embedding2: Second set of embeddings (N x D torch.Tensor)
        top_k: Maximum k for top-k accuracy
        sum_all: Whether to return summed results
    
    Returns:
        Tuple containing:
        - if sum_all: (summed top-k matches, matching score)
        - else: (top-k boolean matrix, matching score)
    """
    # Compute pairwise distances
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    
    # Get sorted indices and compute top-k matches
    sorted_indices = dist_mat.argsort(dim=1)
    top_k_mat = calculate_top_k(sorted_indices, top_k)
    
    return (top_k_mat.sum(dim=0), matching_score) if sum_all else (top_k_mat, matching_score)


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        rank0_print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

