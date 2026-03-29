wkpath=experiments/DGAF_VSR_REDS

CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch examples/dgafvsr/train_dgafvsr.py \
    --pretrained_model_name_or_path ckpts/DGAF_VSR \
    --output_dir ${wkpath} \
    --dataset_config_path="examples/dgafvsr/dataset/config_reds.yaml"  \
    --dataloader_num_workers 4 \
    --learning_rate 5e-5 \
    --train_batch_size 8 \
    --report_to tensorboard \
    --resume_from_checkpoint latest \
    --checkpointing_steps 10000 
# --debug --debug_path tmp0511
# gt_size = 256, batchsize=2, Memory = 11467MiB
# gt_size = 256, batchsize=8, Memory = 18423MiB / 32768MiB
# gt_size = 320, batchsize=8, Memory = 23163MiB /  32768MiB
# gt_size = 384, batchsize=8, Memory = 29566MiB / 32768MiB
# gt_size = 512, batchsize=4, Memory = 26382MiB / 32510MiB
# gt_size = 640, batchsize=2, Memory = 22864MiB / 32510MiB  
# v1.2: feature_up_nn
# v3.1: flow_up_bi 11.160.135.9
# v3.2: flow_up_nn 234