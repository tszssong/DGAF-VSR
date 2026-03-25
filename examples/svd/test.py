
# from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, DDPMScheduler
import torch
import cv2
import numpy as np
from PIL import Image


import accelerate
from diffusers import DiffusionPipeline
# pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid")

pipline = DiffusionPipeline.from_pretrained("/mnt/nfs0/hanzhang/models/StableDiffusion/svd_img2vid")
# python examples/svd/test.py
# Loading pipeline components...:   0%|                                                                                                                                                 | 0/5 [00:00<?, ?it/s]
# /mnt/vdb/hanzhang/wkspace/sftp/inpWK/myBrushNet/src/diffusers/models/lora.py:387: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
#   deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)
# /mnt/vdb/hanzhang/wkspace/sftp/inpWK/myBrushNet/src/diffusers/models/lora.py:300: FutureWarning: `LoRACompatibleConv` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleConv` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
#   deprecate("LoRACompatibleConv", "1.0.0", deprecation_message)
# Loading pipeline components...:  80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                           | 4/5 [00:03<00:00,  1.10it/s]
# Traceback (most recent call last):
#   File "/mnt/vdb/hanzhang/wkspace/sftp/inpWK/myBrushNet/examples/svd/test.py", line 10, in <module>
#     pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid")
#   File "/root/anaconda3/envs/diffusers/lib/python3.9/site-packages/huggingface_hub/utils/_validators.py", line 119, in _inner_fn
#     return fn(*args, **kwargs)
#   File "/mnt/vdb/hanzhang/wkspace/sftp/inpWK/myBrushNet/src/diffusers/pipelines/pipeline_utils.py", line 817, in from_pretrained
#     loaded_sub_model = load_sub_model(
#   File "/mnt/vdb/hanzhang/wkspace/sftp/inpWK/myBrushNet/src/diffusers/pipelines/pipeline_loading_utils.py", line 473, in load_sub_model
#     loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
#   File "/root/anaconda3/envs/diffusers/lib/python3.9/site-packages/transformers/modeling_utils.py", line 2970, in from_pretrained
#     raise ImportError(
# ImportError: Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`