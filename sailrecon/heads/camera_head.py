# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sailrecon.heads.head_act import activate_pose
from sailrecon.layers import Mlp
from sailrecon.layers.block import Block


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True)
        )

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(
        self,
        aggregated_tokens_list: list,
        cam_token_last_layer: torch.Tensor | None,
        num_iterations: int = 4,
    ) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        num_recon = cam_token_last_layer.shape[1]
        num_reloc = pose_tokens.shape[1]
        pose_tokens = torch.cat([cam_token_last_layer, pose_tokens], dim=1)
        pose_tokens = self.token_norm(pose_tokens)

        attention_mask = build_lr_mask(
            S=num_recon + num_reloc,
            no_reloc_list=[i for i in range(num_recon)],
            device=pose_tokens.device,
        )
        pred_pose_enc_list = self.trunk_fn(pose_tokens, attention_mask, num_iterations)
        pred_pose_enc_list_returned = [[] for i in range(num_iterations)]
        for i in range(len(pred_pose_enc_list)):
            pred_pose_enc_list_returned[i] = pred_pose_enc_list[i][:, num_recon:]
        return pred_pose_enc_list_returned

    def trunk_fn(
        self,
        pose_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        num_iterations: int,
    ) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(
                3, dim=-1
            )

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(
                self.adaln_norm(pose_tokens), shift_msa, scale_msa
            )
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for idx_, blk in enumerate(self.trunk):
                pose_tokens_modulated = blk(
                    pose_tokens_modulated, None, ~attention_mask
                )
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(
                self.trunk_norm(pose_tokens_modulated)
            )

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift


def build_lr_mask(S: int, no_reloc_list, device="cpu"):
    """
    Args:
        S (int)          : total number of tokens in the sequence.
        r_idx (Sequence) : indices of r-tokens (unique, 0-based).
        device (str)     : target device for the mask tensor.

    Returns:
        attn_mask (torch.BoolTensor) of shape (1, 1, S, S)
        — ready for F.scaled_dot_product_attention (True == masked).
    """
    # ----
    r_idx = torch.tensor(
        [i for i in range(S) if i not in no_reloc_list], dtype=torch.long, device=device
    )
    l_idx = torch.as_tensor(no_reloc_list, dtype=torch.long, device=device).unique(
        sorted=True
    )

    mask = torch.zeros(S, S, dtype=torch.bool, device=device)

    # ----
    if l_idx.numel() and r_idx.numel():
        mask[l_idx[:, None], r_idx[None, :]] = True

    # ----
    if r_idx.numel() > 1:
        mask[r_idx[:, None], r_idx[None, :]] = True
        mask[r_idx, r_idx] = False

    # ---- 3. 补 batch & head 维度
    return mask.unsqueeze(0).unsqueeze(0)
