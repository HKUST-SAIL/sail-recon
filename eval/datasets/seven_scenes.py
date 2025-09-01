import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images

from eval.utils.eval_utils import uniform_sample


class SevenScenesUnifiedDataset(Dataset):
    def __init__(self, root_dir, scene_name="chess"):
        self.scene_dir = os.path.join(root_dir, f"pgt_7scenes_{scene_name}")

        self.train_seqs = os.path.join(self.scene_dir, "train")
        self.test_seqs = os.path.join(self.scene_dir, "test")

        self.test_samples = sorted(
            glob.glob(os.path.join(self.test_seqs, "rgb", "*.png"))
        )
        self.train_samples = sorted(
            glob.glob(os.path.join(self.train_seqs, "rgb", "*.png"))
        )
        self.all_samples = self.test_samples  # + self.train_samples
        # len_samples = len(self.all_samples)
        # self.all_samples = self.all_samples[::len_samples//200]

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        return self._load_sample(self.all_samples[idx])

    def get_train_sample(self, n=4):
        uniform_sampled = uniform_sample(len(self.all_samples), n)
        selected = [self.all_samples[i] for i in uniform_sampled]
        return [self._load_sample(s) for s in selected]

    def _load_sample(self, rgb_path):
        img_name = os.path.basename(rgb_path)
        color = load_and_preprocess_images([rgb_path])[0]
        pose_path = (
            rgb_path.replace("rgb", "poses")
            .replace("color", "pose")
            .replace(".png", ".txt")
        )
        pose = np.loadtxt(pose_path)
        pose = torch.from_numpy(pose).float()

        return dict(
            img=color,
            camera_pose=pose,  # cam2world
            dataset="7Scenes",
            true_shape=torch.tensor([392, 518]),
            label=img_name,
            instance=img_name,
        )
