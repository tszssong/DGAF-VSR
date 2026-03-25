# sleep 12m
MODEL_NAME=/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0
VAE_NAME=/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix
DATASET_NAME=/mnt/nfs0/hanzhang/data/style/pet3d/train0923/gt_picked/
bs=1
rank=64
size=960
OUT_PTH=/large/hanzhang/wkspace/styleWK2409/animal3D_${size}x${bs}x${rank}_$(date "+%m%d-%H%M%S")
OUT_PTH_SCRIPTS=${OUT_PTH}/scripts
mkdir ${OUT_PTH}
mkdir ${OUT_PTH_SCRIPTS}
cp -r $(dirname $0)/*.sh ${OUT_PTH_SCRIPTS}/
cp -r $(dirname $0)/*.py ${OUT_PTH_SCRIPTS}/
cp  $DATASET_NAME/metadata.jsonl ${OUT_PTH_SCRIPTS}/
CUDA_VISIBLE_DEVICES='3' accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --resolution=${size} --random_flip \
  --train_batch_size=${bs} --rank ${rank}\
  --num_train_epochs=2000 --checkpointing_steps=1000 \
  --validation_epochs=4000 --num_validation_images=1 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir=$OUT_PTH \
  --validation_prompt="tbs cartoon, a cat, gray and white, blue eyes, lay, hardwood floor"
