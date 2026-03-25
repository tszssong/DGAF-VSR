MODEL_NAME=/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0
VAE_NAME=/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix
DATASET_NAME=/mnt/nfs0/hanzhang/data/style/pet3d/trainDR/train20
OUT_PTH=/large/hanzhang/wkspace/styleWK/sdxl_lora$(date "+%m%d-%H%M%S")
OUT_PTH_SCRIPTS=${OUT_PTH}/scripts
mkdir ${OUT_PTH}
mkdir ${OUT_PTH_SCRIPTS}
cp -r $(dirname $0) ${OUT_PTH_SCRIPTS}
CUDA_VISIBLE_DEVICES='3' accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --resolution=768 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=5 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=$OUT_PTH \
  --validation_prompt="xhs cartoon, a cat, gray and white, blue eyes, lay, hardwood floor"
