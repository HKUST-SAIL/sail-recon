# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# croppping utilities
# --------------------------------------------------------
import PIL.Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import numpy as np  # noqa
from easyvolcap.reloc_eval.utils.device import to_numpy
from easyvolcap.reloc_eval.utils.geometry import (  # noqa
    colmap_to_opencv_intrinsics,
    geotrf,
    inv,
    opencv_to_colmap_intrinsics,
)

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ImageList:
    """Convenience class to aply the same operation to a whole set of images."""

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(
    image, depthmap, camera_intrinsics, output_resolution, force=True
):
    """Jointly rescale a (image, depthmap)
    so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(
        tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final
    )

    return image.to_pil(), depthmap, camera_intrinsics


def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    if depthmap is not None:
        depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def bbox_from_intrinsics_in_out(
    input_camera_matrix, output_camera_matrix, output_resolution
):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox


def reciprocal_1d(corres_1_to_2, corres_2_to_1, ret_recip=False):
    is_reciprocal1 = corres_2_to_1[corres_1_to_2] == np.arange(len(corres_1_to_2))
    pos1 = is_reciprocal1.nonzero()[0]
    pos2 = corres_1_to_2[pos1]
    if ret_recip:
        return is_reciprocal1, pos1, pos2
    return pos1, pos2


def generate_non_self_pairs(n):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    pairs = np.stack([i.ravel(), j.ravel()], axis=1)

    mask = pairs[:, 0] != pairs[:, 1]
    filtered_pairs = pairs[mask]

    return filtered_pairs


def unravel_xy(pos, shape):
    # convert (x+W*y) back to 2d (x,y) coordinates
    return np.unravel_index(pos, shape)[0].base[:, ::-1].copy()


def ravel_xy(pos, shape):
    H, W = shape
    with np.errstate(invalid="ignore"):
        qx, qy = pos.reshape(-1, 2).round().astype(np.int32).T
    quantized_pos = qx.clip(min=0, max=W - 1, out=qx) + W * qy.clip(
        min=0, max=H - 1, out=qy
    )
    return quantized_pos


def extract_correspondences_from_pts3d(
    view1, view2, target_n_corres, rng=np.random, ret_xy=True, nneg=0
):
    view1, view2 = to_numpy((view1, view2))
    # project pixels from image1 --> 3d points --> image2 pixels
    shape1, corres1_to_2 = reproject_view(view1["pts3d"], view2)
    shape2, corres2_to_1 = reproject_view(view2["pts3d"], view1)

    # compute reciprocal correspondences:
    # pos1 == valid pixels (correspondences) in image1
    is_reciprocal1, pos1, pos2 = reciprocal_1d(
        corres1_to_2, corres2_to_1, ret_recip=True
    )
    is_reciprocal2 = corres1_to_2[corres2_to_1] == np.arange(len(corres2_to_1))

    if target_n_corres is None:
        if ret_xy:
            pos1 = unravel_xy(pos1, shape1)
            pos2 = unravel_xy(pos2, shape2)
        return pos1, pos2

    available_negatives = min((~is_reciprocal1).sum(), (~is_reciprocal2).sum())
    target_n_positives = int(target_n_corres * (1 - nneg))
    n_positives = min(len(pos1), target_n_positives)
    n_negatives = min(target_n_corres - n_positives, available_negatives)

    if n_negatives + n_positives != target_n_corres:
        # should be really rare => when there are not enough negatives
        # in that case, break nneg and add a few more positives ?
        n_positives = target_n_corres - n_negatives
        assert n_positives <= len(pos1)

    assert n_positives <= len(pos1)
    assert n_positives <= len(pos2)
    assert n_negatives <= (~is_reciprocal1).sum()
    assert n_negatives <= (~is_reciprocal2).sum()
    assert n_positives + n_negatives == target_n_corres

    valid = np.ones(n_positives, dtype=bool)
    if n_positives < len(pos1):
        # random sub-sampling of valid correspondences
        perm = rng.permutation(len(pos1))[:n_positives]
        pos1 = pos1[perm]
        pos2 = pos2[perm]

    if n_negatives > 0:
        # add false correspondences if not enough
        def norm(p):
            return p / p.sum()

        pos1 = np.r_[
            pos1,
            rng.choice(
                shape1[0] * shape1[1],
                size=n_negatives,
                replace=False,
                p=norm(~is_reciprocal1),
            ),
        ]
        pos2 = np.r_[
            pos2,
            rng.choice(
                shape2[0] * shape2[1],
                size=n_negatives,
                replace=False,
                p=norm(~is_reciprocal2),
            ),
        ]
        valid = np.r_[valid, np.zeros(n_negatives, dtype=bool)]

    # convert (x+W*y) back to 2d (x,y) coordinates
    if ret_xy:
        pos1 = unravel_xy(pos1, shape1)
        pos2 = unravel_xy(pos2, shape2)
    return pos1, pos2, valid


def reproject_view(pts3d, view2):
    shape = view2["pts3d"].shape[:2]
    return reproject(
        pts3d, view2["camera_intrinsics"], inv(view2["camera_pose"]), shape
    )


def reproject(pts3d, K, world2cam, shape):
    H, W, THREE = pts3d.shape
    assert THREE == 3

    # reproject in camera2 space
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = geotrf(K @ world2cam[:3], pts3d, norm=1, ncol=2)

    # quantize to pixel positions
    return (H, W), ravel_xy(pos, shape)
