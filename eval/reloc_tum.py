import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vggt.models.sail_recon import SailRecon
from vggt.utils.load_fn import load_and_preprocess_images

from eval.datasets.tum import TumDatasetAll
from eval.utils.device import to_cpu
from eval.utils.eval_utils import save_tum_poses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configuration parameters for the reconstruction system"
    )

    parser.add_argument(
        "--recon_img_num",
        type=int,
        default=100,
        help="Number of anchor image for reconstruction, default: 100",
    )
    parser.add_argument(
        "--fixed_rank",
        type=int,
        default=300,
        help="Number of token per image, default: 300",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=20,
        help="Number of relocalization for each forward, default: 20",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default: 42")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/junyuan.deng/projects/GFM/gfm/data/trained_model/reloc_vggtv6_v4/clean.pt",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/horizon-bucket/saturn_v_4dlabel/008_Simulation/001_users/junyuan.deng/evaluation_datasets/tum/unzip",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/reloc_tum_100",
        help="Directory to save output results",
    )
    parser.add_argument(
        "--save_dense_results",
        action="store_true",
        help="Whether to save dense results (depth maps and point clouds)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    scene_name_list = [
        # "rgbd_dataset_freiburg1_360",
        # "rgbd_dataset_freiburg1_desk",
        # "rgbd_dataset_freiburg1_desk2",
        # "rgbd_dataset_freiburg1_plant",
        # "rgbd_dataset_freiburg1_room",
        # "rgbd_dataset_freiburg1_rpy",
        # "rgbd_dataset_freiburg1_teddy",
        # "rgbd_dataset_freiburg1_xyz",
        "rgbd_dataset_freiburg1_floor",
    ]

    for scene_idx in range(len(scene_name_list)):
        print(f"Processing scene: {scene_name_list[scene_idx]}")
        processing_scene = scene_name_list[scene_idx]
        save_dir = os.path.join(args.save_dir, f"{processing_scene}")
        os.makedirs(save_dir, exist_ok=True)

        model = SailRecon(kv_cache=True)
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device=device)
        model.eval()

        with torch.no_grad():
            dataset = TumDatasetAll(
                args.dataset_root, scene_name=scene_name_list[scene_idx]
            )
            test_loader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=4
            )
            train_samples = dataset.get_train_sample(args.recon_img_num)

            # swap the first image to be a more central view for better reconstruction
            if processing_scene == "rgbd_dataset_freiburg1_floor":
                train_samples[0], train_samples[70] = (
                    train_samples[70],
                    train_samples[0],
                )

            for sample in train_samples:
                sample["img"] = sample["img"].unsqueeze(0).to(device)

            no_reloc_list = [i for i in range(len(train_samples))]
            reloc_list = []
            model.tmp_forward(
                train_samples,
                no_reloc_list=no_reloc_list,
                reloc_list=reloc_list,
                fix_rank=args.fixed_rank,
            )
            sample.clear()
            train_samples.clear()
            del model.aggregator.global_blocks, sample, train_samples

            with torch.cuda.amp.autocast(dtype=dtype):
                ## start sail recon
                buffer = []
                all_data = []
                poses_c2w_estimated = []
                result = []
                verify_reconstruction = False
                for img_data in tqdm(test_loader):
                    img_data["img"] = img_data["img"].to(device)

                    buffer.append(img_data)
                    if len(buffer) == args.buffer_size:
                        torch.cuda.empty_cache()
                        result += to_cpu(
                            model.reloc(
                                buffer,
                                no_reloc_list=no_reloc_list,
                                fix_rank=args.fixed_rank,
                                fast_reloc=False,
                            )
                        )
                        all_data += to_cpu(buffer)
                        buffer = []
                        include_first_frame = False

                if len(buffer) > 0:
                    torch.cuda.empty_cache()
                    result += to_cpu(
                        model.reloc(
                            buffer,
                            no_reloc_list=no_reloc_list,
                            fix_rank=args.fixed_rank,
                        )
                    )
                    all_data += to_cpu(buffer)

                poses_w2c_estimated = [
                    one_result["extrinsic"][0].cpu().numpy() for one_result in result
                ]
                poses_c2w_estimated = [
                    np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])]))
                    for pose in poses_w2c_estimated
                ]
                time_stamp = [
                    float(data["instance"][0].split(".png")[0]) for data in all_data
                ]
                save_tum_poses(
                    poses_c2w_estimated,
                    time_stamp,
                    os.path.join(save_dir, "pred_tum.txt"),
                )

                if args.save_dense_results:
                    depth_all = [
                        one_result["depth_map"][0].cpu().numpy()
                        for one_result in result
                    ]
                    point_map_unproj_all = [
                        one_result["point_map_by_unprojection"][0].cpu().numpy()
                        for one_result in result
                    ]
                    point_map_all = [
                        one_result["point_map"][0].cpu().numpy()
                        for one_result in result
                    ]
                    np.savez_compressed(
                        os.path.join(save_dir, "dense_results.npz"),
                        depth_all=depth_all,
                        point_map_unproj_all=point_map_unproj_all,
                        point_map_all=point_map_all,
                        pred_poses=poses_c2w_estimated,
                    )
