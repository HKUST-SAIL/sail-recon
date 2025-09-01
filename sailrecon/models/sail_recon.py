# Copyright (c) HKUST SAIL-Lab and Horizon Robotics.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from sailrecon.heads.camera_head import CameraHead
from sailrecon.heads.dpt_head import DPTHead
from sailrecon.heads.track_head import TrackHead
from sailrecon.models.aggregator import Aggregator
from sailrecon.utils.geometry import unproject_depth_map_to_point_map
from sailrecon.utils.pose_enc import pose_encoding_to_extri_intri


class SailRecon(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        kv_cache=False,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            kv_cache=kv_cache,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )
        # self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.cam_token_last_layer = None
        self.need_re_forward = False

    def tmp_forward(self, views, no_reloc_list=None, reloc_list=[], fix_rank=300):
        # second time forward, need to clear the kv cache
        if self.need_re_forward:
            for blk in self.aggregator.global_reloc_blocks:
                blk.attn.clear_kv_cache()
        else:
            self.need_re_forward = True

        if isinstance(views, torch.Tensor):
            rgbs = views
        else:
            rgbs = torch.cat([view["img"] for view in views], dim=0)
        if len(rgbs.shape) == 4:
            rgbs = rgbs.unsqueeze(0)
        if no_reloc_list is None:
            no_reloc_list = [i for i in range(len(views))]
        rgb_feats, idx_patch, cam_token_last_layer = self.aggregator(
            rgbs, no_reloc_list, reloc_list, fix_rank=fix_rank
        )
        if self.cam_token_last_layer is None:
            self.cam_token_last_layer = cam_token_last_layer.clone()
        rgb_feats.clear()

        del rgbs, rgb_feats, idx_patch, cam_token_last_layer

    def reloc(
        self,
        views,
        no_reloc_list=None,
        fix_rank=300,
        memory_save=True,
        save_depth=True,
        fast_reloc=False,
        ret_img=False,
    ):
        if isinstance(views, torch.Tensor):
            rgbs = views
        else:
            rgbs = torch.cat([view["img"] for view in views], dim=0)
        if len(rgbs.shape) == 4:
            rgbs = rgbs.unsqueeze(0)

        rgb_feats, idx_patch = self.aggregator.forward_with_cache(
            rgbs, fix_rank=fix_rank
        )

        cam_tokens = rgb_feats[-1]
        # Extract the camera tokens
        cam_tokens = cam_tokens[:, :, 0]

        predictions = {}

        final_results = [{} for _ in range(len(views))]
        # Turn off the AMP for the decoder heads for better stability
        with torch.cuda.amp.autocast(enabled=False):
            # Camera decoder head
            cam_maps = self.camera_head(
                rgb_feats, self.cam_token_last_layer
            )  # [(B, S, 9), ...]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                cam_maps[-1], (rgbs.shape[-2], rgbs.shape[-1])
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            if fast_reloc:
                for key, value in predictions.items():
                    for i in range(len(views)):
                        final_results[i][key] = value[:, i]
                return final_results
            # Xyz point decoder head
            xyz_map, xyz_cnf = self.point_head(
                rgb_feats, images=rgbs, patch_start_idx=idx_patch
            )

            dpt_map, dpt_cnf = self.depth_head(
                rgb_feats, images=rgbs, patch_start_idx=idx_patch
            )

            conf_percentage_count_xyz = []
            B, S, Hn, Wn, C = xyz_map.shape
            for i in range(len(views)):
                conf_percentage_count_perview = []
                for count_value in torch.arange(1, 5.5, 0.25):
                    conf_percentage_count_perview.append(
                        (xyz_cnf[:, i] > count_value).sum(dim=(1, 2)).item() / Hn / Wn
                    )
                conf_percentage_count_xyz.append(conf_percentage_count_perview)

            point_map_by_unprojection = unproject_depth_map_to_point_map(
                dpt_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
            )
            if not memory_save:
                predictions["point_map_by_unprojection"] = point_map_by_unprojection[
                    None
                ]

                predictions["point_map"] = xyz_map
                predictions["rgbs"] = rgbs
                predictions["xyz_cnf"] = xyz_cnf
            if save_depth:
                predictions["depth_map"] = dpt_map
                predictions["dpt_cnf"] = dpt_cnf
            predictions["cam_tokens"] = cam_tokens

            if ret_img:
                predictions["images"] = rgbs

        for key, value in predictions.items():
            for i in range(len(views)):
                final_results[i][key] = value[:, i]
        return final_results

    def co3d_forward(self, views, no_reloc_list=None, reloc_list=None, fix_rank=None):
        rgbs_no_reloc = torch.cat(
            [view["img"] for view in [views[i] for i in no_reloc_list]], dim=0
        )
        rgb_reloc = torch.cat(
            [view["img"] for view in [views[i] for i in reloc_list]], dim=0
        )
        rgbs = torch.cat([rgbs_no_reloc, rgb_reloc], dim=0)
        if len(rgbs.shape) == 4:
            rgbs = rgbs.unsqueeze(0)
        reloc_list = [i + len(no_reloc_list) for i in reloc_list]
        no_reloc_list = [i for i in range(len(no_reloc_list))]
        rgb_feats, idx_patch, cam_token_last_layer = self.aggregator(
            rgbs, no_reloc_list, reloc_list, fix_rank=fix_rank
        )

        with torch.cuda.amp.autocast(dtype=torch.float32):
            cam_maps = self.camera_head(
                rgb_feats, cam_token_last_layer
            )  # [(B, S, 9), ...]
            with torch.cuda.amp.autocast(dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    cam_maps[-1], (rgbs.shape[-2], rgbs.shape[-1])
                )
        return extrinsic
