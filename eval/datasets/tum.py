import glob
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images

from eval.utils.eval_utils import uniform_sample


class TumDatasetAll(Dataset):
    def __init__(self, root_dir, scene_name="rgbd_dataset_freiburg1_360"):
        self.scene_name = scene_name
        self.scene_all_dir = os.path.join(root_dir, f"{scene_name}", "rgb")

        self.test_samples = sorted(glob.glob(os.path.join(self.scene_all_dir, "*.png")))
        self.all_samples = self.test_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        return self._load_sample(self.all_samples[idx])

    def get_train_sample(self, n=4):
        gap = len(self.all_samples) // n
        gap = max(gap, 1)  # Ensure at least one sample is selected
        gap = min(gap, len(self.all_samples))  # Ensure gap does not exceed length
        if gap == 1:
            uniform_sampled = uniform_sample(len(self.all_samples), n)
            selected = [self.all_samples[i] for i in uniform_sampled]
        else:
            selected = self.all_samples[::gap]
            if len(selected) > n:
                uniform_sampled = uniform_sample(len(selected), n)
                selected = [selected[i] for i in uniform_sampled]
        if self.scene_name == "rgbd_dataset_freiburg1_floor":
            selected += self.all_samples[-20::5]
        return [self._load_sample(s) for s in selected]

    def _load_sample(self, rgb_path):
        img_name = os.path.basename(rgb_path)
        color = load_and_preprocess_images([rgb_path])[0]

        return dict(
            img=color,
            dataset="tnt_all",
            true_shape=torch.tensor([392, 518]),
            label=img_name,
            instance=img_name,
        )
