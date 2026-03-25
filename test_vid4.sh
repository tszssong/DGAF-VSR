
CUDA_VISIBLE_DEVICES='2' python  examples/dgafvsr/test_dgafvsr.py \
    --in_path assets/Vid4/Bicubic4xLR/ \
    --model_id ckpts/DGAF_VSR/ \
    --ckpt ckpts/DGAF_VSR/DGAF_VSR_REDS \
    --out_path results/Vid4/dgaf_vid4_s50  \
    --num_inference_steps 50
python evalY.py --en_path results/Vid4/dgaf_vid4_s50 