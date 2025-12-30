<div align="center">
<h1>SAIL-Recon: Large SfM by Augmenting Scene Regression with Localization</h1>
<h2>2026 3DV Oral</h2>
<a href="https://arxiv.org/pdf/2508.17972"><img src="https://img.shields.io/badge/arXiv-2508.17972-b31b1b" alt="arXiv"></a>
<a href="https://hkust-sail.github.io/sail-recon/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/HKUST-SAIL/SAIL-Recon'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>


**[HKUST Spatial Artificial Intelligence Lab](https://github.com/HKUST-SAIL)**; **[Horizon Robotics](https://en.horizon.auto/)**


[Junyuan Deng](https://scholar.google.com/citations?user=KTCPC5IAAAAJ&hl=en), [Heng Li](https://hengli.me/), [Tao Xie](https://github.com/xbillowy), [Weiqiang Ren](https://cn.linkedin.com/in/weiqiang-ren-b2798636), [Qian Zhang](https://cn.linkedin.com/in/qian-zhang-10234b73), [Ping Tan](https://facultyprofiles.hkust.edu.hk/profiles.php?profile=ping-tan-pingtan), [Xiaoyang Guo](https://xy-guo.github.io/)
</div>

![pic1](docs/traj_ply.png)



## Overview

Sail-Recon is a feed-forward Transformer that scales neural scene regression to large-scale Structure-from-Motion by augmenting it with visual localization. From a few anchor views, it constructs a global latent scene representation that encodes both geometry and appearance. Conditioned on this representation, the network directly regresses camera poses, intrinsics, depth maps, and scene coordinate maps for thousands of images in minutes, enabling precise and robust reconstruction without iterative optimization.


## TODO
- [x] Inference Code Release
- [x] Gradio Demo
- [ ] Evaluation Script

## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub) following VGGT.

```bash
git clone https://github.com/HKUST-SAIL/sail-recon.git
cd sail-recon
pip install -e .
```

You can download the demo image (e.g., [Barn](https://drive.google.com/file/d/0B-ePgl6HF260NzQySklGdXZyQzA/view?resourcekey=0-luQ7Jaym5BQL6IjxsgXY9A) from [Tanks & Temples](https://www.tanksandtemples.org/)) and put the images in `examples/demo_image`.

Now, you can try the model demo:
```bash
# Images
python demo.py --img_dir path/to/your/images --out_dir outputs
# Video
python demo.py --vid_dir path/to/your/images --out_dir outputs
```

You can find the ply file and camera pose under `outputs`.

We also provide a Gradio demo for easier usage. You can run the demo by:
```bash
python demo_gradio.py
```
Please note that the Gradio demo is slower than `demo.py` due to the visualization part.


## Evaluation

Please refer to [this](eval/readme.md) for more details.

## Acknowledgements

Thanks to these great repositories:

[ACE0](https://github.com/nianticlabs/acezero) for the PSNR evaluation;

[VGGT](https://github.com/facebookresearch/vggt) for the template of github, gradio and visualization;

[Fast3R](https://github.com/facebookresearch/fast3r) for the training data processing and some utility functions;

And many other inspiring works in the community.

If you find this project useful in your research, please consider citing:
```bibtex
@article{dengli2025sail,
  title={SAIL-Recon: Large SfM by Augmenting Scene Regression with Localization},
  author={Deng, Junyuan and Li, Heng and Xie, Tao and Ren, Weiqiang and Zhang, Qian and Tan, Ping and Guo, Xiaoyang},
  journal       = {2026 International Conference on 3D Vision (3DV).},
  year          = {2026},
  archivePrefix = {IEEE},
  primaryClass  = {cs.CV}
}
```

## License

See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

Please see the license of [VGGT](https://github.com/facebookresearch/vggt) about the other code used in this project.

Please see the license of [ACE0](https://github.com/nianticlabs/acezero) about the evaluation used in this project.

Please see the license of [Fast3R](https://github.com/facebookresearch/fast3r) about the utility functions used in this project.
