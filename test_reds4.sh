CUDA_VISIBLE_DEVICES='1' python examples/dgafvsr/test_dgafvsr.py \
    --in_path assets/REDS4/x4/ \
    --model_id ckpts/DGAF_VSR/ \
    --ckpt ckpts/DGAF_VSR/DGAF_VSR_REDS \
    --out_path results/REDS4/dgaf_reds4_s50 \
    --num_inference_steps 50
python eval.py  results/REDS4/dgaf_reds4_s50 
