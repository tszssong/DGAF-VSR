from diffusers import DiffusionPipeline
import torch

def get_pipe(lora_path):
    pipe = DiffusionPipeline.from_pretrained("/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(lora_path)
    return pipe 

# prompt = "A naruto with green eyes and red legs."
# image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
# image.save("naruto.png")


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : myTest.py
@Author : Meisong
@Time: 2024/08/09 14:52:25
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, shutil, argparse, logging
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/trainDR/train20.lst', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--lora', type=str, default='/large/hanzhang/wkspace/styleWK/sdxl_lora0809-130348', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--save_in', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    pipe = get_pipe(args.lora)
    args.out = os.path.join(args.lora, f"pet3d_s{args.seed}")
    os.makedirs(args.out, exist_ok=True)
    with open(args.lst, 'r') as fr:
        for line in fr.readlines():
            ipth = line.strip()
            iname, post = os.path.basename(ipth).split('.')
            tpth = ipth.replace(f'.{post}', '.txt')
            caption = ''
            with open(tpth, 'r') as fr:
                caption = fr.readline()
            image = pipe(prompt=caption, num_inference_steps=30, guidance_scale=7.5).images[0]
            opth = os.path.join(args.out, f"{iname}.png")
            image.save(opth)
            if args.save_in:
                shutil.copy(ipth, opth.replace('.png', '_in.png'))