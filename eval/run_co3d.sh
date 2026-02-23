CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. \
python test_co3d.py \
    --model_path /YOUR/MODEL/PATH \
    --co3d_dir /YOUR/CO3D/PATH \
    --co3d_anno_dir /YOUR/CO3D/ANNO/PATH \
    --recon NUMBER_OF_RECON \
    --reloc NUMBER_OF_RELOC \
    --fixed_rank NUMBER_OF_TOKEN \
    --seed 0
