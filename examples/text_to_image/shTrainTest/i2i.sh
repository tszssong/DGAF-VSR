MODEL_NAME=/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0
VAE_NAME=/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix
DATASET_NAME=/mnt/nfs0/hanzhang/data/style/pet3d/trainDR/train20
size=960
bs=1
OUT_PTH=/large/hanzhang/wkspace/styleWK/sdxllorai2i_${size}x${bs}_$(date "+%m%d-%H%M%S")
OUT_PTH_SCRIPTS=${OUT_PTH}/scripts
mkdir ${OUT_PTH}
mkdir ${OUT_PTH_SCRIPTS}
cp -r $(dirname $0) ${OUT_PTH_SCRIPTS}
CUDA_VISIBLE_DEVICES='3' accelerate launch myTrainI2I0809.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --resolution=${size} --random_flip \
  --train_batch_size=${bs} \
  --num_train_epochs=500 --checkpointing_steps=500 \
  --validation_epochs=500 --num_validation_images=1 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=$OUT_PTH \
  --validation_prompt="xhs cartoon, a cat, gray and white, blue eyes, lay, hardwood floor"
