MODEL_NAME=/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0
VAE_NAME=/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix
DATASET_NAME=/mnt/nfs0/hanzhang/data/style/pet3d/trainDR/train20
W=832 # 31010MiB / 32510MiB
H=1152
BS=1
OUT_PTH=/large/hanzhang/wkspace/styleWK/xllora${W}x${H}x${BS}_$(date "+%m%d-%H%M%S")
OUT_PTH_SCRIPTS=${OUT_PTH}/scripts
mkdir ${OUT_PTH}
mkdir ${OUT_PTH_SCRIPTS}
cp -r $(dirname $0) ${OUT_PTH_SCRIPTS}
CUDA_VISIBLE_DEVICES='2' accelerate launch myTraindiffWH0812.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --inw=${W} --inh=${H} --random_flip \
  --train_batch_size=${BS} \
  --num_train_epochs=1000 --checkpointing_steps=500 \
  --validation_epochs=500 --num_validation_images=1 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=$OUT_PTH \
  --validation_prompt="cute dragon creature" 
