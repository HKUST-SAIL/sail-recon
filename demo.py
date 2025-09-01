# Copyright (c) HKUST SAIL-Lab and Horizon Robotics.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
from tqdm import tqdm

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def demo(args):
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model_dir = args.ckpt
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model = SailRecon(kv_cache=True)
    if model_dir is not None:
        model.load_state_dict(torch.load(model_dir))
    else:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(_URL, model_dir=model_dir)
        )
    model = model.to(device=device)
    model.eval()

    # Load and preprocess example images
    scene_name = "1"
    if args.vid_dir is not None:
        import cv2

        image_names = []
        video_path = args.vid_dir
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        tmp_file = os.path.join("tmp_video", os.path.basename(video_path).split(".")[0])
        os.makedirs(tmp_file, exist_ok=True)
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            image_path = os.path.join(tmp_file, f"{video_frame_num:06}.png")
            cv2.imwrite(image_path, frame)
            image_names.append(image_path)
            video_frame_num += 1
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(video_path).split(".")[0]
    else:
        image_names = os.listdir(args.img_dir)
        image_names = [os.path.join(args.img_dir, f) for f in sorted(image_names)]
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(args.img_dir)

    # anchor image selection
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]

    os.makedirs(os.path.join(args.out_dir, scene_name), exist_ok=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # processing anchor images to build scene representation (kv_cache)
            print("Processing anchor images ...")
            model.tmp_forward(anchor_images)
            # remove the global transformer blocks to save memory during relocalization
            del model.aggregator.global_blocks
            # relocalization on all images
            predictions = []

            with tqdm(total=len(image_names), desc="Relocalizing") as pbar:
                for img_split in images.split(20, dim=0):
                    pbar.update(20)
                    predictions += to_cpu(model.reloc(img_split))

            # save the predicted point cloud and camera poses

            from eval.utils.geometry import save_pointcloud_with_plyfile

            save_pointcloud_with_plyfile(
                predictions, os.path.join(args.out_dir, scene_name, "pred.ply")
            )

            import numpy as np

            from eval.utils.eval_utils import save_kitti_poses

            poses_w2c_estimated = [
                one_result["extrinsic"][0].cpu().numpy() for one_result in predictions
            ]
            poses_c2w_estimated = [
                np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])]))
                for pose in poses_w2c_estimated
            ]

            save_kitti_poses(
                poses_c2w_estimated,
                os.path.join(args.out_dir, scene_name, "pred.txt"),
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--img_dir", type=str, default="samples/kitchen", help="input image folder"
    )
    args.add_argument("--vid_dir", type=str, default=None, help="input video path")
    args.add_argument("--out_dir", type=str, default="outputs", help="output folder")
    args.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model checkpoint"
    )
    args = args.parse_args()
    demo(args)
