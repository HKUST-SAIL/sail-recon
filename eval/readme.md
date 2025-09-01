## Tanks and Temples.

### Image set
1. Data Preparation

    Download the images data from [here](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) (for intermidiated and advanced set, please download from [here](https://github.com/isl-org/TanksAndTemples/issues/35)), and COLMAP results from (here)[https://storage.googleapis.com/niantic-lon-static/research/acezero/colmap_raw.tar.gz]. We thank [ACE0](https://github.com/nianticlabs/acezero) again for providing the COLMAP results.

2. Adjust the parameter in `run_tnt.sh`

    Specify the `dataset_root`, `colmap_dir`, `model_path` and `save_dir` in the file.

3. Get the inference results.

    ```
    sh run_tnt.sh
    ```
### Video set
<details>
<summary>Click to expand</summary>

1. Data Preparation

    Download the video sequence and from [here](https://www.tanksandtemples.org/download/) and get images from video via [this](https://www.tanksandtemples.org/tutorial/).

2. Run Inference

    Replace `docs/demo_image` in `../demo.py` to the path storing images from videl.
</details>

## 7 scenes

1. Data Preparation
    Download the corresponding sequence from [here](https://jonbarron.info/mipnerf360/).


## TUM-RGBD

1. Data Preparation

    Download the corresponding sequence from [here](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).

2. Adjust the parameter in `run_tum.sh`

    Specify the `dataset_root`, `recon_img_num`, `model_path` and `save_dir` in the file.

3. Evaluate the results.

    ```
    sh run_tum.sh
    ```
    Noting that we set the `recon_img_num` to 50 or 100 according to the length of dataset. Please refer to the supplementary of paper for detail.

4. Using evo to evaluate The results

    ```
    evo_ape tum gt_pose.txt pred_tum.txt -vas
    ```


## 7 scenes

1. Download the dataset from [here](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and _Pseudo Ground Truth (PGT)_
(see
the [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Brachmann_On_the_Limits_of_Pseudo_Ground_Truth_in_Visual_Camera_ICCV_2021_paper.html)
,
and [associated code](https://github.com/tsattler/visloc_pseudo_gt_limitations/) for details).


2. Adjust the parameter in `run_7scenes.sh`

    Specify the `dataset_root`, `recon_img_num`, `model_path` and `save_dir` in the file.

3. Evaluate the results.

    ```
    sh run_7scenes.sh
    ```
    You will see a `result.txt` file reporting the evaluation results.


## Mip-NeRF 360

1. Data Preparation

    Download the data from [here](https://jonbarron.info/mipnerf360/).

2. Adjust the parameter in `run_mip.sh`

    Specify the `dataset_root`, `model_path` and `save_dir` in the file.

3. Get the inference results.

    ```
    sh run_mip.sh
    ```

## Co3D-V2

1. We thank VGGT for providing evaluation code of CO3D-V2 dataset. Please see link [here](https://github.com/facebookresearch/vggt/tree/evaluation/evaluation#dataset-preparation) for data preparation and processing.

2. Adjust the parameterco3d_dir in `runco3d_anno_dir_7scenes.sh`

    Specify the `dataset_root`, `recon_img_num`, `model_path`, `recon`, `reloc` and `fixed_rank` in the file.

3. Evaluate the results.

    ```
    sh run_co3d.sh
    ```
    You will see evaluation result in the terminal.
