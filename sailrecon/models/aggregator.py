# Copyright (c) HKUST SAIL-Lab and Horizon Robotics.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sailrecon.layers import PatchEmbed
from sailrecon.layers.block import Block
from sailrecon.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from sailrecon.layers.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        ## sail_recon
        intermediate_layer_idx=[4, 11, 17, 23],  # used by the DPT head
        min_rank: int = 150,
        kv_cache: bool = False,
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.intermediate_layer_idx = intermediate_layer_idx

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_reloc_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    kv_cache=kv_cache,
                )
                for i in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, embed_dim)
        )
        self.camera_token_reloc = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.register_token_reloc = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, embed_dim)
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.camera_token_reloc, std=1e-6)
        nn.init.normal_(self.register_token_reloc, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False
            )

        self.use_reentrant = False  # hardcoded to False
        self.generator = self._generate_per_rank_generator()

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        no_reloc_list: list,
        reloc_list: list,
        fix_rank: Union[int | None] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        num_recon = len(no_reloc_list)
        num_reloc = len(reloc_list)
        self.num_recon = num_recon
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        if fix_rank is not None:
            self.rank = min(fix_rank, P)
        else:
            self.rank = torch.randint(
                min(self.min_rank, P // 2),
                max(self.min_rank, P // 2),
                (1,),
                generator=self.generator,
            ).item()
        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        camera_token[:, reloc_list] = self.camera_token_reloc[:, 0:1].expand(
            B, num_reloc, *self.camera_token_reloc.shape[2:]
        )
        register_token[:, reloc_list] = self.register_token_reloc[:, 0:1].expand(
            B, num_reloc, *self.register_token_reloc.shape[2:]
        )
        camera_token = camera_token.view(B * S, *camera_token.shape[2:])
        register_token = register_token.view(B * S, *register_token.shape[2:])

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        del camera_token, register_token, patch_tokens

        allow_blk = build_allow_block(
            S, [i for i in range(num_recon)], [i + num_recon for i in range(num_reloc)]
        )  # (S, S)
        attention_mask = expand_to_token(allow_blk, tokens.shape[1])  # (S, S)
        attention_mask = attention_mask[
            (tokens.shape[1] - self.patch_start_idx - self.rank) * num_recon :,
            (tokens.shape[1] - self.patch_start_idx - self.rank) * num_recon :,
        ].to(tokens.device)
        attention_mask.unsqueeze_(0)  # (1, S, S)
        attention_mask.unsqueeze_(0)  # (1, 1, S, S)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2)
                .to(images.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        global_reloc_idx = 0
        concat_inter = None
        output_dict = {}

        for _ in range(self.aa_block_num):
            layer_idx = frame_idx
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    (
                        tokens,
                        frame_idx,
                        frame_intermediates,
                    ) = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    # perform downsampling for reloc frames
                    (
                        downsampled_frame_tokens,
                        downsampled_frame_pos,
                    ) = self.select_scene_repe_for_reloc(
                        tokens.view(B, S, P, C)[:, no_reloc_list],
                        pos.view(B, S, -1, 2)[:, no_reloc_list],
                    )
                    # expand downsampled tokens and positions to match the number of reloc frames
                    downsampled_frame_tokens = downsampled_frame_tokens.view(B, -1, C)
                    downsampled_frame_pos = downsampled_frame_pos.view(B, -1, 2)
                elif attn_type == "global":
                    (
                        reloc_tokens,
                        global_reloc_idx,
                        global_reloc_intermediates,
                    ) = self._process_global_reloc_attention(
                        downsampled_frame_tokens,
                        tokens.view(B, S, P, C)[:, reloc_list].view(B, -1, C),
                        max(num_reloc, 1),
                        num_recon,
                        B,
                        P if num_reloc != 0 else 0,
                        P_prime=min(self.rank + self.patch_start_idx, P),
                        C=C,
                        global_reloc_idx=global_reloc_idx,
                        downsampled_frame_pos=downsampled_frame_pos,
                        reloc_pos=pos.view(B, S, -1, 2)[:, reloc_list].view(B, -1, 2),
                        attention_mask=attention_mask,
                    )
                    (
                        global_tokens,
                        global_idx,
                        global_intermediates,
                    ) = self._process_global_attention(
                        tokens.view(B, S, P, C)[:, no_reloc_list].view(B, -1, C),
                        B,
                        num_recon,
                        P,
                        C,
                        global_idx,
                        pos=pos.view(B, S, P, 2)[:, no_reloc_list].view(B, -1, 2),
                    )
                    tokens = torch.ones(
                        (B, S, P, C), dtype=tokens.dtype, device=tokens.device
                    )
                    tokens[:, no_reloc_list] = global_tokens.view(
                        B, len(no_reloc_list), P, C
                    )
                    tokens[:, reloc_list] = reloc_tokens.view(B, len(reloc_list), P, C)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                if layer_idx + i in self.intermediate_layer_idx and num_reloc > 0:
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat(
                        [
                            frame_intermediates[i][:, reloc_list],
                            global_reloc_intermediates[i],
                        ],
                        dim=-1,
                    )
                    output_dict[layer_idx + i] = concat_inter
            if frame_idx == self.aa_block_num:
                cam_token_last_layer = torch.cat(
                    [
                        frame_intermediates[-1][:, no_reloc_list][:, :, 0],
                        global_intermediates[-1].view(B, len(no_reloc_list), P, C)[
                            :, :, 0
                        ],
                    ],
                    dim=-1,
                ).clone()  # [B, R, P, 2C]
        del concat_inter
        del frame_intermediates
        del global_intermediates
        assert (
            self.depth - 1 in output_dict
        ) or num_reloc == 0, f"Please make sure the last layer ({self.depth - 1}) is in the output_dict: {output_dict.keys()}"
        if num_reloc > 0:
            output_dict[-1] = output_dict[self.depth - 1]

        return output_dict, self.patch_start_idx, cam_token_last_layer

    def forward_with_cache(
        self,
        images: torch.Tensor,
        cameras: torch.Tensor = None,  # dim 9 [quaternion, translation, fx, fy]
        camera_dropout: float = False,
        fix_rank: Union[int | None] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        assert B == 1, "Batch size must be 1 for this model"
        # assert num_recon + num_reloc == S, \
        # f"Expected S ({S}) to be the sum of num_recon ({num_recon}) and num_reloc ({num_reloc})"

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        if fix_rank is not None:
            self.rank = fix_rank
        else:
            self.rank = torch.randint(
                min(self.min_rank, P // 2),
                max(self.min_rank, P // 2),
                (1,),
                generator=self.generator,
            ).item()

        # Expand camera and register tokens to match batch size and sequence length

        camera_token = self.camera_token_reloc[:, 0:1].expand(
            B, S, *self.camera_token_reloc.shape[2:]
        )
        register_token = self.register_token_reloc[:, 0:1].expand(
            B, S, *self.register_token_reloc.shape[2:]
        )
        camera_token = camera_token.view(B * S, *camera_token.shape[2:])
        register_token = register_token.view(B * S, *register_token.shape[2:])

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        del camera_token, register_token, patch_tokens

        allow_blk = build_allow_block(
            self.num_recon + S,
            [i for i in range(self.num_recon)],
            [i + self.num_recon for i in range(S)],
        )  # (S, S)
        attention_mask = expand_to_token(allow_blk, tokens.shape[1])  # (S, S)
        attention_mask = attention_mask[
            -tokens.shape[0] * tokens.shape[1] :,
            -(self.rank + self.patch_start_idx) * self.num_recon
            - tokens.shape[0] * tokens.shape[1] :,
        ].to(tokens.device)
        attention_mask.unsqueeze_(0)  # (1, S, S)
        attention_mask.unsqueeze_(0)  # (1, 1, S, S)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2)
                .to(images.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        output_dict = {}
        frame_idx = 0
        global_reloc_idx = 0
        for _ in range(self.aa_block_num):
            layer_idx = frame_idx
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    (
                        tokens,
                        frame_idx,
                        frame_intermediates,
                    ) = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )

                elif attn_type == "global":
                    # perform global reloc attention
                    (
                        tokens,
                        global_reloc_idx,
                        global_reloc_intermediates,
                    ) = self._process_global_reloc_attention_cache(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        global_reloc_idx,
                        pos=pos,
                        attn_mask=attention_mask,
                    )

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                if layer_idx + i in self.intermediate_layer_idx:
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_reloc_intermediates[i]], dim=-1
                    )
                    output_dict[layer_idx + i] = concat_inter

        assert (
            self.depth - 1 in output_dict
        ), f"Please make sure the last layer ({self.depth - 1}) is in the output_dict: {output_dict.keys()}"
        output_dict[-1] = output_dict[self.depth - 1]

        return output_dict, self.patch_start_idx

    def select_scene_repe_for_reloc(self, tokens, pos):
        extra_token = tokens[:, :, : self.patch_start_idx, :]
        scene_repre_patch = tokens[:, :, self.patch_start_idx :, :]
        extra_pos = pos[:, :, : self.patch_start_idx, :]
        scene_pos = pos[:, :, self.patch_start_idx :, :]
        select_scene, selected_pos = self.random_select_features(
            scene_repre_patch, self.rank, scene_repre_patch.device, pos=scene_pos
        )
        select_scene = torch.cat((extra_token, select_scene), dim=2)
        selected_pos = torch.cat((extra_pos, selected_pos), dim=2)

        return select_scene, selected_pos

    def random_select_features(self, feats, l_prime, device, pos=None):
        b, r, l, c = feats.shape
        if l_prime > l:
            # logger.warning(
            #     f"Requested l_prime ({l_prime}) is greater than l ({l}). "
            # )
            l_prime = l
            # raise ValueError("l_prime cannot be greater than l")

        # Initialize the tensor to store selected features
        selected_features = torch.zeros(
            b, r, l_prime, c, device=device, dtype=feats.dtype
        )
        if pos is not None:
            b_p, r_p, l_p, c_p = pos.shape
            assert (
                b_p == b and r_p == r and l_p == l
            ), "pos must have the same shape as feats"
            # If pos is provided, ensure it has the same shape as feats
            selected_pos = torch.zeros(
                b_p, r_p, l_prime, c_p, device=device, dtype=pos.dtype
            )

        # Generate random indices for each batch sample using the rank-specific generator
        for i in range(b):
            for j in range(r):
                random_indices = torch.randperm(l, generator=self.generator)[
                    :l_prime
                ].to(device)
                selected_features[i][j] = feats[i, j, random_indices, :]
                selected_pos[i][j] = (
                    pos[i, j, random_indices, :] if pos is not None else None
                )
        return selected_features, selected_pos if pos is not None else None

    def _generate_per_rank_generator(self):
        # this way, the randperm will be different for each rank, but deterministic given a fixed number of forward passes (tracked by self.random_generator)
        # and to ensure determinism when resuming from a checkpoint, we only need to save self.random_generator to state_dict
        # generate a per-rank random seed
        per_forward_pass_seed = torch.randint(0, 2**32, (1,)).item()
        world_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        per_rank_seed = per_forward_pass_seed + world_rank

        # Set the seed for the random generator
        per_rank_generator = torch.Generator()
        per_rank_generator.manual_seed(per_rank_seed)
        return per_rank_generator

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_reloc_attention(
        self,
        downsampled_frame_tokens,
        reloc_tokens,
        num_reloc,
        num_recon,
        B,
        P,
        P_prime,
        C,
        global_reloc_idx,
        downsampled_frame_pos=None,
        reloc_pos=None,
        attention_mask=None,
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        # concatenate downsampled tokens with the reloc frame tokens
        assert downsampled_frame_tokens.shape == (B, num_recon * P_prime, C), (
            f"Expected downsampled_frame_tokens shape to be ({B}, {num_recon} * {P_prime}, {C}), "
            f"but got {downsampled_frame_tokens.shape}"
        )
        if reloc_tokens.shape != (B, num_reloc * P, C):
            reloc_tokens = reloc_tokens.view(B, num_reloc, P, C).view(
                B, num_reloc * P, C
            )
        reloc_tokens = torch.cat([downsampled_frame_tokens, reloc_tokens], dim=1)

        if downsampled_frame_pos is not None and downsampled_frame_pos.shape != (
            num_reloc,
            num_recon * P_prime,
            2,
        ):
            assert downsampled_frame_pos.shape == (B, num_recon * P_prime, 2), (
                f"Expected downsampled_frame_pos shape to be (1, {num_recon} * {P_prime}, 2), "
                f"but got {downsampled_frame_pos.shape}"
            )
        if reloc_pos is not None and reloc_pos.shape != (B, num_reloc * P, C):
            reloc_pos = reloc_pos.view(B, num_reloc, P, 2).view(B, num_reloc * P, 2)
        reloc_pos = (
            torch.cat([downsampled_frame_pos, reloc_pos], dim=1)
            if downsampled_frame_pos is not None
            else None
        )

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                reloc_tokens = checkpoint(
                    self.global_reloc_blocks[global_reloc_idx],
                    reloc_tokens,
                    reloc_pos,
                    attention_mask,
                    use_reentrant=self.use_reentrant,
                )
            else:
                reloc_tokens = self.global_reloc_blocks[global_reloc_idx](
                    reloc_tokens, reloc_pos, attention_mask
                )
            global_reloc_idx += 1

            # take out the tokens
            reloc_results = reloc_tokens[:, num_recon * P_prime :]

            intermediates.append(reloc_results.view(B, num_reloc, P, C))

        return reloc_results, global_reloc_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _process_global_reloc_attention_cache(
        self, tokens, B, S, P, C, global_reloc_idx, pos=None, attn_mask=None
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.global_reloc_blocks[global_reloc_idx],
                    tokens,
                    pos,
                    attn_mask,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.global_reloc_blocks[global_reloc_idx](
                    tokens, pos, attn_mask
                )
            global_reloc_idx += 1

            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_reloc_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    # combined = combined.view(B * S, *combined.shape[2:])
    return combined


def build_allow_block(L: int, L_a: list[int], L_b: list[int]) -> torch.BoolTensor:
    allow = torch.zeros(L, L, dtype=torch.bool)

    # 1) L_a can see each other
    idx_a = torch.tensor(L_a)
    allow[idx_a[:, None], idx_a[None, :]] = True

    # 2) Lb can see L_a
    if len(L_b) > 0:
        idx_b = torch.tensor(L_b)
        allow[idx_b[:, None], idx_a[None, :]] = True

        # 3) L_b see L_b
        allow[idx_b, idx_b] = True

    return allow


def expand_to_token(allow_blk: torch.BoolTensor, P: int) -> torch.BoolTensor:
    return allow_blk.repeat_interleave(P, 0).repeat_interleave(P, 1)
