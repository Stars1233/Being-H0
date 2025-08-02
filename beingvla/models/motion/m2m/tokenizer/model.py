import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .encdec import Encoder, Decoder
from .quantizer.vq import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from .quantizer.rq import ResidualVQ
from .quantizer.grvq import GroupedResidualVQ
from .quantizer.part_grvq import PartGroupedResidualVQ
from .quantizer.lfq import LFQ
from .quantizer.fsq import FSQ
from ..utils.rotation_conversions import rot6d_to_aa


class MotionReconstructionLoss(nn.Module):
    """Handles motion reconstruction loss with different pose configurations."""
    LOSS_MAP = {
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'l1_smooth': nn.SmoothL1Loss
    }

    def __init__(self, recons_loss: str, nb_joints: int, motion_feat: str):
        super().__init__()
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.loss_fn = self.LOSS_MAP[recons_loss]()
        self.nb_joints = nb_joints
        self.motion_feat = motion_feat
        self._setup_forward_vel()

    @property
    def motion_dim(self) -> int:
        return (self.nb_joints - 1) * 12 + 4 + 3 + 4

    def _setup_forward_vel(self):
        """Configure velocity calculation based on pose features."""
        vel_handlers = {
            'hm3d263': self._vel_263,
            'smpl130': self._vel_130,
            'smpl135': self._vel_135,
            'smpl263': self._vel_263,
            'smpl268': self._vel_268,
            'mano51': self._mano_wrist_51,
            'mano114': self._mano_wrist_51,
            'mano99': self._mano_wrist_99,
            'mano162': self._mano_wrist_99,
            'mano109': self._mano_wrist_99,
            'mano36': self._mano_wrist_36,
            'mano100': self._mano_wrist_36,
        }
        self.forward_vel = vel_handlers.get(self.motion_feat, lambda *_: torch.tensor(0.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred[..., :self.motion_dim], target[..., :self.motion_dim])
    
    def _vel_263(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        slice_end = (self.nb_joints - 1) * 3 + 4
        return self.loss_fn(pred[..., 4:slice_end], target[..., 4:slice_end])
    
    def _vel_130(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=pred.device)
    
    def _vel_135(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=pred.device)
    
    def _vel_268(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        slice_end = (self.nb_joints - 1) * 3 + 9
        return self.loss_fn(pred[..., 9:slice_end], target[..., 9:slice_end])

    def _mano_wrist_51(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred[..., 45:51], target[..., 45:51])

    def _mano_wrist_99(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred[..., 90:99], target[..., 90:99])

    def _mano_wrist_36(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred[..., 30:36], target[..., 30:36])


# ==================== Quantizer Models ====================
class BaseVQModel(nn.Module):
    """Base class for vector quantization models."""
    
    def __init__(self, args, dim_motion=None):
        super().__init__()
        """Initialize encoder/decoder and quantizer based on config."""
        self.code_dim = args.code_dim
        self.dim_pose = dim_motion if dim_motion is not None else int(args.motion_feat[4:])
        self.quantizer_name = args.quantizer_name

        if "part" in self.quantizer_name:
            # Reversed
            # Left Hand  = pelvis 0,right_collar 14,right_shoulder 17,right_elbow 19,right_wrist 21           = [0,14,17,19,21]
            # Right Hand = pelvis 0,left_collar 13,left_shoulder 16,left_elbow 18,left_wrist 20               = [0,13,16,18,20]
            # Left leg   = pelvis 0,left_hip 1,left_knee 4,left_ankle 7,left_foot 10                          = [0,1,4,7,10]
            # Right leg  = pelvis 0,right_hip 2,right_knee 5,right_ankle 8,right_foot 11                      = [0,2,5,8,11]
            # Body       = pelvis 0,spine1 3,spine2 6,spine3 9,neck 12,left_collar 13,right_collar 14,head 15 = [0,3,6,9,12,13,14,15]
            if "mano" not in args.motion_feat:
                self.left_hand_indices =self.get_part_indices([3,6,9,13,16,18,20])  
                self.right_hand_indices = self.get_part_indices([3,6,9,14,17,19,21])
                self.left_leg_indices = self.get_part_indices([3,6,9,1,4,7,10])
                self.right_leg_indices = self.get_part_indices([3,6,9,2,5,8,11])
                self.body_indices = self.get_part_indices([3,6,9,12,13,14,15])
                self.part_indices = torch.cat([self.left_hand_indices, self.right_hand_indices, self.left_leg_indices, self.right_leg_indices, self.body_indices])
                self.left_hand_ID = [16,18,20]
                self.right_hand_ID = [17,19,21] 
                self.left_leg_ID = [1,4,7,10]
                self.right_leg_ID = [2,5,8,11]
                self.body_ID = [3,6,9,12,13,14,15]
                dim_pose = 95
            elif args.motion_feat == "mano114":
                self.thumb_indices = self.get_mano_part_indices([1, 2, 3, 4])  # thumb
                self.index_indices = self.get_mano_part_indices([5, 6, 7, 8])
                self.middle_indices = self.get_mano_part_indices([9, 10, 11, 12])
                self.ring_indices = self.get_mano_part_indices([13, 14, 15, 16])
                self.pinky_indices = self.get_mano_part_indices([17, 18, 19, 20])
                self.part_indices = torch.cat([self.thumb_indices, self.index_indices,self.middle_indices, self.ring_indices, self.pinky_indices])
                self.dim_pose = 30
            
        self.encoder = Encoder(
            input_emb_width=self.dim_pose,
            output_emb_width=args.output_emb_width,
            down_t=args.down_t,
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation=args.activate,
            norm=args.norm,
            num_conv_layers=args.num_conv_layers
        )
        self.decoder = Decoder(
            input_emb_width=self.dim_pose,
            output_emb_width=args.output_emb_width,
            down_t=args.down_t,  # Assuming symmetric
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation=args.activate,
            norm=args.norm,
            num_conv_layers=args.num_conv_layers
        )

        self.quantizer = self._create_quantizer(args)

    def _create_quantizer(self, args):
        """Factory method for quantizer creation."""
        levels_dict = {
            256: [8, 6, 5], 512: [8, 8, 8],
            1024: [8, 5, 5, 5], 2048: [8, 8, 6, 5],
            4096: [7, 5, 5, 5, 5], 8192: [8, 6, 6, 5, 5],
            16384: [8, 8, 8, 6, 5], 65536: [8, 8, 8, 5, 5, 5]
        }
        quantizers = {
            "ema_reset": lambda: QuantizeEMAReset(args.nb_code, args.code_dim, args.mu),
            "orig": lambda: Quantizer(args.nb_code, args.code_dim, 1.0),
            "ema": lambda: QuantizeEMA(args.nb_code, args.code_dim),
            "reset": lambda: QuantizeReset(args.nb_code, args.code_dim, args.mu),
            "residualvq": lambda: ResidualVQ(
                codebook_size=args.nb_code,
                dim=args.code_dim,
                num_quantizers=args.num_quantizers,
                shared_codebook=args.shared_codebook
            ),
            "group_residualvq": lambda: GroupedResidualVQ(codebook_size=args.nb_code, dim=args.code_dim, 
                                               num_quantizers=args.num_quantizers, 
                                               groups=args.num_quant_groups,
                                               shared_codebook=args.shared_codebook),
            "part_group_residualvq": lambda: PartGroupedResidualVQ(codebook_size=args.nb_code, dim=args.code_dim, 
                                               num_quantizers=args.num_quantizers, 
                                               groups=2,
                                               shared_codebook=args.shared_codebook),
            "simvq": lambda: SimVQ(codebook_size=args.nb_code, dim=args.code_dim,rotation_trick=True),
            "lfq": lambda: LFQ(codebook_size=args.nb_code, dim=args.code_dim),
            "bsq": lambda: LFQ(codebook_size=args.nb_code, dim=args.code_dim, spherical=True),
            "fsq": lambda: FSQ(levels=levels_dict[args.nb_code], dim=args.code_dim)
        }
        return quantizers[args.quantizer_name]()
    
    def get_part_indices(self, joint_ids):
        root_end = 4
    
        pos_start = root_end
        pos_end = pos_start + 21*3
        pos_indices = torch.as_tensor([pos_start + (j-1) * 3 + k for j in joint_ids for k in range(3)]).reshape(-1,3)

        rot_start = pos_end
        rot_end = rot_start + 21*6
        rot_indices = torch.as_tensor([rot_start + (j-1) * 6 + k for j in joint_ids for k in range(6)]).reshape(-1,6)

        vel_start = rot_end
        vel_end = vel_start + 22*3
        vel_indices = torch.as_tensor([vel_start + j * 3 + k for j in joint_ids for k in range(3)]).reshape(-1,3)

        root_indices = torch.as_tensor([0,1,2,3,259,260,261,262]+[vel_start + k for k in range(3)])

        indices = torch.cat([pos_indices, rot_indices, vel_indices], dim=-1).flatten()
        indices = torch.cat([root_indices, indices])
        return indices
    
    def get_mano_part_indices(self, joint_ids):
        """Get indices for MANO hand parts."""
        pos_start = 51
        pos_indices = torch.as_tensor([pos_start + j * 3 + k for j in joint_ids for k in range(3)]).reshape(-1,3) # 4 * 3

        theta_start = 0
        theta_order_list = [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
        theta_order_map = {i: j for i, j in enumerate(theta_order_list)}

        theta_indices = torch.as_tensor([theta_start + (theta_order_map[j] - 1) * 3 + k for j in joint_ids[:-1] for k in range(3)]).reshape(-1,3) # 3 * 3

        root_indices = torch.as_tensor([45,46,47,48,49,50] + [pos_start + k for k in range(3)])  # 3 + 3 + 3, wrist rot, trans, joint

        indices = torch.cat([pos_indices, theta_indices], dim=0).flatten() # 7*3 = 21
        indices = torch.cat([root_indices, indices]) # 9 + 21 = 30

        return indices


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float() # (bs, T, Jx3) -> (bs, Jx3, T)
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor: 
        return x.permute(0, 2, 1) # (bs, Jx3, T) ->  (bs, T, Jx3)
    
    def encode(self, x: torch.Tensor):
        N, T, _ = x.shape
        try:
            x_encoder = self.encoder(self.preprocess(x))
        except:
            breakpoint()
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder).view(N, -1)
        return code_idx

    def forward_decoder(self, x):
        bs = x.shape[0]
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(bs, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_quant, commit_loss, perplexity = self.quantizer(x_enc)
        x_out = self.postprocess(self.decoder(x_quant))
        return x_out, commit_loss, perplexity
        

# ==================== Specialized Quantizer Models ====================
class ResidualVQModel(BaseVQModel):
    """Residual VQ variant."""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        _, indices, _ = self.quantizer(x_enc)
        return indices
    
    def forward(self, x: torch.Tensor) -> tuple:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        x_quant, indices, commit_loss = self.quantizer(x_enc)
        x_out = self.postprocess(self.decoder(x_quant.permute(0, 2, 1)))
        return x_out, commit_loss.mean(), torch.tensor(-1, device=x.device)

class GroupResidualVQModel(BaseVQModel):
    """Grouped Residual VQ variant."""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1) # B, T/4, D'
        _, indices, _ = self.quantizer(x_enc) # indices: 2, B, T/4, layer
        return indices

    def forward_decoder(self, x):
        bs = x.shape[1]
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(bs, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        x_quant, indices, commit_loss = self.quantizer(x_enc) # indices: 2, B, T/4, layer
        x_out = self.postprocess(self.decoder(x_quant.permute(0, 2, 1)))
        return x_out, commit_loss.mean(), torch.tensor(-1, device=x.device)

class PRQModel(BaseVQModel):
    """Part-based Residual Quantization Model for motion processing."""
    
    def part_divider(self, x: torch.Tensor) -> torch.Tensor:
        """Split input tensor into body part segments."""
        bs, _, t = x.shape
        if self.dim_pose != 95:
            raise NotImplementedError(f"Unsupported pose dimension: {self.dim_pose}")
        return x[:, self.part_indices].reshape(bs*5, self.dim_pose, t)

    def part_combiner(self, x: torch.Tensor) -> torch.Tensor:
        """Recombine body part segments into full motion representation."""
        # 5*bs,t,95->bs,t,263
        bsp, t, d = x.shape
        bs = bsp // 5  # Original batch size
        p = int(bsp / bs)

        # Reshape and separate components
        x_part = x.view(bs, p, t, d)
        x_root = x_part[:, :, :, :11]
        x_joints = x_part[:, :, :, 11:].view(bs, p, t, 7, 12)

        root_data = x_root[:, 4]  # [bs, t, 11]
        joint_data = {
            'left_hand': x_joints[:, 0, :, 4:],  # 9,6,3,13,16,18,20 -> 16,18,20
            'right_hand': x_joints[:, 1, :, 4:], # 9,6,3,14,17,19,21 -> 17,19,21
            'left_leg': x_joints[:, 2, :, 3:], # 9,6,3,1,4,7,10 -> 1,4,7,10
            'right_leg': x_joints[:, 3, :, 3:], # 9,6,3,2,5,8,11 -> 2,5,8,11
            'body': x_joints[:, 4]               # 3,6,9,12,13,14,15
        }

        whole_body = torch.zeros([bs, t, self.num_joints, 12], device=x.device)
        whole_body[:, :, self.left_hand_ID] = joint_data['left_hand']
        whole_body[:, :, self.right_hand_ID] = joint_data['right_hand']
        whole_body[:, :, self.left_leg_ID] = joint_data['left_leg']
        whole_body[:, :, self.right_leg_ID] = joint_data['right_leg']
        whole_body[:, :, self.body_ID] = joint_data['body']

        # Flatten components
        pos = whole_body[:, :, 1:, :3].flatten(2)
        rot = whole_body[:, :, 1:, 3:9].flatten(2)
        vel = whole_body[:, :, 1:, 9:].flatten(2)

        return torch.cat([
            root_data[..., :4],  # root indices
            pos, rot,
            root_data[..., 8:],   # root velocity
            vel,
            root_data[..., 4:8]  # foot indices
        ], dim=-1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Process input through part-based quantization pipeline."""
        x_part = self.part_divider(self.preprocess(x)) # bs,263,t -> bs*5, 95, t

        x_enc = self.encoder(x_part).permute(0, 2, 1)
        x_quant, _, commit_loss = self.quantizer(x_enc)
        
        x_dec = self.decoder(x_quant.permute(0,2,1)) 
        x_part_out = self.postprocess(x_dec) # bs*5,t,95

        x_out = self.part_combiner(x_part_out)
      
        return x_out, commit_loss.mean(), torch.as_tensor(-1, device=x.device)


class FSQModel(BaseVQModel):
    """Finite Scalar Quantization Model."""

    def encode(self, x):
        x_enc = self.encoder(self.preprocess(x))
        _, code_idx, _, _, _, _ = self.quantizer(x_enc)
        return code_idx.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> tuple:
        x_enc = self.encoder(self.preprocess(x)) # 256, 512, 16
        x_quant, _, loss, perplexity, activate, indices = self.quantizer(x_enc)
        
        x_decoder = self.decoder(x_quant)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity


class LFQModel(BaseVQModel):
    """Lookup-Free Quantization Model."""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_encoder = self.encoder(self.preprocess(x))
        _, code_idx, _, _, _, _ = self.quantizer(x_encoder)
        return code_idx.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> tuple:
        x_enc = self.encoder(self.preprocess(x)) # 256, 512, 16
        x_quant, indices, loss = self.quantizer(x_enc)
        
        x_decoder = self.decoder(x_quant)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, torch.as_tensor(-1, device=x.device)
    

class MotionVQModel(PreTrainedModel):
    """Main human motion VQ-VAE model."""
    config_class = PretrainedConfig  # Can customize config class
    
    def __init__(self, args):
        config = PretrainedConfig()
        super().__init__(config)
  
        self.model = self._create_vq_model(args)
        self.dim_pose = int(args.motion_feat[4:])
        self.loss_fn = MotionReconstructionLoss(
            args.recons_loss, 
            args.nb_joints, 
            args.motion_feat
        )

        self.vel_weight = args.vel_weight
        self.commit_weight = args.commit_weight

    def _create_vq_model(self, args):
        """Factory method for VQ model creation."""
        model_map = {
            "residualvq": ResidualVQModel,
            "group_residualvq": GroupResidualVQModel,
            "part_residualvq": PRQModel,
            "fsq": FSQModel
        }
        return model_map.get(args.quantizer_name, BaseVQModel)(args)
    
    def encode(self, x):
        return self.model.encode(x)

    def forward_decoder(self, x):
        return self.model.forward_decoder(x)

    def forward(self, motion: torch.Tensor, **kwargs):
        
        pred_motion, commit_loss, perplexity = self.model(motion.float())

        recon_loss = self.loss_fn(pred_motion, motion)
        vel_loss = self.loss_fn.forward_vel(pred_motion, motion)
        
        total_loss = recon_loss + self.commit_weight * commit_loss + self.vel_weight * vel_loss
        
        return {
            'loss': total_loss,
            'loss_recons': recon_loss,
            'perplexity': perplexity,
            'loss_commit': commit_loss,
            'pred_motion': pred_motion,
        }


class ManoVQModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, args):
        config = PretrainedConfig()
        super().__init__(config)

        self.use_part = args.use_part

        if args.use_part is not None:
            wrist_dim = 9 if args.motion_feat in ["mano109", "mano172", "mano99", "mano162"] else 6
            if args.motion_feat in ["mano109", "mano172"]:
                finger_dim = 100
            elif args.motion_feat in ["mano99", "mano162"]:
                finger_dim = 90
            elif args.motion_feat in ["mano51", "mano114"]:
                finger_dim = 45
            elif args.motion_feat in ["mano36", "mano100"]:
                finger_dim = 30
            else:
                raise ValueError(f"Unsupported motion feature: {args.motion_feat}")
            if args.motion_feat in ["mano172", "mano162", "mano114", "mano100"]:
                finger_dim += 20*3

            if args.use_part == "wrist":
                self.model = self._create_vq_model(args, dim_motion=wrist_dim)
            elif args.use_part == "finger":
                self.model = self._create_vq_model(args, dim_motion=finger_dim)
            else:  # both
                raise NotImplementedError(f"Unsupported use_part: {args.use_part}")
        else:
            self.model = self._create_vq_model(args)

        self.loss_fn = MotionReconstructionLoss(
            args.recons_loss, 
            args.nb_joints, 
            args.motion_feat
        )

        self.vel_weight = args.vel_weight
        self.commit_weight = args.commit_weight

    def _create_vq_model(self, args, dim_motion=None):
        """Factory method for VQ model creation."""
        model_map = {
            "residualvq": ResidualVQModel,
            "group_residualvq": GroupResidualVQModel,
            "part_residualvq": PRQModel,
            "fsq": FSQModel
        }
        return model_map.get(args.quantizer_name, BaseVQModel)(args, dim_motion)
    
    def encode(self, x):
        return self.model.encode(x)

    def forward_decoder(self, x):
        return self.model.forward_decoder(x)

    def forward(self, motion: torch.Tensor, wrist_motion: torch.Tensor, finger_motion: torch.Tensor, **kwargs):

        if self.use_part is None:
            x_motion = motion
        elif self.use_part == "wrist":
            x_motion = wrist_motion
        elif self.use_part == "finger":
            x_motion = finger_motion
        
        pred_motion, commit_loss, perplexity = self.model(x_motion.float())

        recon_loss = self.loss_fn(pred_motion, x_motion)
        if self.use_part is None:
            vel_loss = self.loss_fn.forward_vel(pred_motion, x_motion)
        else:
            vel_loss = torch.tensor(0.0)
        
        total_loss = recon_loss + self.commit_weight * commit_loss + self.vel_weight * vel_loss
        
        return {
            'loss': total_loss,
            'loss_recons': recon_loss,
            'perplexity': perplexity,
            'loss_commit': commit_loss,
            'pred_motion': pred_motion,
            'loss_vel': vel_loss
        }
    