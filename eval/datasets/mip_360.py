import glob
import os
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images


class Mip360Dataset(Dataset):
    def __init__(self, root_dir, scene_name="bicycle"):
        self.scene_dir = os.path.join(
            root_dir,
            f"{scene_name}",
        )

        self.test_samples = sorted(
            glob.glob(os.path.join(self.scene_dir, "images_8", "*.JPG"))
        )
        # self.train_samples = sorted(glob.glob(os.path.join(self.train_seqs, "rgb", "*.png")))
        self.all_samples = self.test_samples  # + self.train_samples
        bin_path = os.path.join(self.scene_dir, "sparse", "0", "images.bin")
        self.poses = read_images_bin(bin_path)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        return self._load_sample(self.all_samples[idx])

    def get_train_sample(self, n=4):
        # _rng = np.random.default_rng(seed=777)
        gap = len(self.all_samples) // n
        gap = max(gap, 1)  # Ensure at least one sample is selected
        gap = min(gap, len(self.all_samples))  # Ensure gap does not exceed length
        if gap == 1:
            selected = sorted(
                random.sample(self.all_samples, min(n, len(self.all_samples)))
            )
        else:
            selected = self.all_samples[::gap]
            if len(selected) > n:
                selected = sorted(random.sample(selected, n))
        return [self._load_sample(s) for s in selected]

    def _load_sample(self, rgb_path):
        img_name = os.path.basename(rgb_path)
        color = load_and_preprocess_images([rgb_path])[0]
        pose = torch.from_numpy(self.poses[img_name]).float()

        return dict(
            img=color,
            camera_pose=pose,  # cam2world
            dataset="7Scenes",
            true_shape=torch.tensor([392, 518]),
            label=img_name,
            instance=img_name,
        )


def read_images_bin(bin_path: str | Path):
    bin_path = Path(bin_path)
    poses = {}

    with bin_path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]  # uint64
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.frombuffer(f.read(8 * 4), dtype=np.float64)  # qw,qx,qy,qz
            tvec = np.frombuffer(f.read(8 * 3), dtype=np.float64)  # tx,ty,tz
            cam_id = struct.unpack("<I", f.read(4))[0]  # camera_id

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\0":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8")

            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, 1)

            # world→cam to cam→world
            qw, qx, qy, qz = qvec
            R_wc = np.array(
                [
                    [
                        1 - 2 * qy * qy - 2 * qz * qz,
                        2 * qx * qy + 2 * qz * qw,
                        2 * qx * qz - 2 * qy * qw,
                    ],
                    [
                        2 * qx * qy - 2 * qz * qw,
                        1 - 2 * qx * qx - 2 * qz * qz,
                        2 * qy * qz + 2 * qx * qw,
                    ],
                    [
                        2 * qx * qz + 2 * qy * qw,
                        2 * qy * qz - 2 * qx * qw,
                        1 - 2 * qx * qx - 2 * qy * qy,
                    ],
                ]
            )
            t_wc = -R_wc @ tvec
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R_wc.astype(np.float32)
            c2w[:3, 3] = t_wc.astype(np.float32)

            poses[name] = c2w

    return poses
