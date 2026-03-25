MODEL_NAME=/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0
VAE_NAME=/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix
DATASET_NAME=/mnt/nfs0/hanzhang/data/style/pet3d/pairDR/train35gt
size=896
bs=1
strength=0.0
OUT_PTH=/mnt/nfs1/hanzhang/LoRA/i2i/0818_${size}x${bs}x${strength}_$(date "+%m%d-%H%M%S")
OUT_PTH_SCRIPTS=${OUT_PTH}/scripts
mkdir ${OUT_PTH}
mkdir ${OUT_PTH_SCRIPTS}
cp -r $(dirname $0) ${OUT_PTH_SCRIPTS}
CUDA_VISIBLE_DEVICES='1' accelerate launch myTrainLoRASDXL0818.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --resolution=${size} --random_flip \
  --train_batch_size=${bs} \
  --num_train_epochs=5000 --checkpointing_steps=500 --latent_save_steps=100 \
  --validation_epochs=2000 --num_validation_images=1 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 --s_i2i $strength  \
  --output_dir=$OUT_PTH \
  --validation_prompt="xhs cartoon, a cat, gray and white, blue eyes, lay, hardwood floor"



  # --mixed_precision="fp16" \ not support yet
