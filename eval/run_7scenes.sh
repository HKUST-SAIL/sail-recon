CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. \
python reloc_7scenes.py \
    --dataset_root "Path to the dataset" \
    --model_path "Path to the trained model" \
    --recon_img_num "50"\
    --save_dir "Directory to save the results"
