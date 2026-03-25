#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : ct2i.py
@Author : Meisong
@Time: 2024/08/15 17:24:16
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
C站下载的模型, 先用命令转换成diffusers可用的:
python convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path /mnt/nfs0/hanzhang/models/civitai/disneyrealcartoonmix_v10.safetensors  \
    --dump_path /mnt/nfs0/hanzhang/models/civitai/disneyrealcartoonmix_v10 --from_safetensors
显存与耗时[分辨率>1024会导致生成多个重复物体]：
size = 1536, 18249MiB / 32510MiB -->  28625MiB / 32510MiB,  1.06s/it[53s/50steps]
size = 1792, 26605MiB / 32510MiB, 1.66s/it[] --> 执行一段时间后OOM
"""

import numpy as np
import PIL, cv2
from PIL import Image
import torch
from diffusers import  StableDiffusionXLPipeline
import random
'''
https://blog.csdn.net/u012063773/article/details/79470009
'''
def random_weight(weight_data):
    total = sum(weight_data.values())    # 权重求和
    ra = random.uniform(0, total)   # 在0与权重和之前获取一个随机数 
    curr_sum = 0
    ret = None
    # keys = weight_data.iterkeys()    # 使用Python2.x中的iterkeys
    keys = weight_data.keys()        # 使用Python3.x中的keys
    for k in keys:
        curr_sum += weight_data[k]             # 在遍历中，累加当前权重值
        if ra <= curr_sum:          # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret

def seed_all(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return torch.Generator("cuda").manual_seed(seed)    # used by diffusers pipe

def get_pipe():
    base_model_path = "/mnt/nfs0/hanzhang/models/civitai/disneyrealcartoonmix_v10"
    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        use_safetensors=True,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe = pipe.to("cuda")
    return pipe

import os, shutil, argparse, logging
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--out', type=str, default='/mnt/nfs0/hanzhang/data/style/t2i/disneyXL', help='')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/Test/style_test_ori.lst', help='')
    parser.add_argument('--fvision', type=str, default='florence2MDcaption', help='florence2caption, florence2Dcaption, florence2MDcaption')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--steps', type=int, default=50, help='')
    parser.add_argument('--inw', type=int, default=1024, help='')
    parser.add_argument('--inh', type=int, default=1344, help='')
    parser.add_argument('--save_in', action='store_true', help='save input')
    args = parser.parse_args()
    
    prefix = os.path.basename(args.lst).split('.')[0]
    
    subpth=f"cDisney_{prefix}_{args.fvision}_{args.inw}x{args.inh}{args.seed}steps{args.steps}"
    args.out = os.path.join(args.out, f'{subpth}')
    logging.info("args: {}".format(args))
    
    os.makedirs(args.out, exist_ok=True)
    
    generator = seed_all(args.seed)
    pipe = get_pipe()
    caption_pre = "modisn disney, "
    caption_post = ", detailed, intricate, high quality"
    
    neg = "EasyNegative, badhandv4, (worst quality, low quality:1.3), low quality, bad anatomy, text, glitch, deformed, mutated, ugly, disfigured,  extra hand, extra leg, extra arm, extra head, extra fingers, extra body parts"
    with open(args.lst, 'r') as fr:
        content_lists = fr.readlines()
        for idx, line in enumerate(content_lists):
            ipth = line.strip()
            iname, post = os.path.basename(ipth).split('.')
            if args.fvision == '':
                tpth = ipth.replace(f'.{post}', '.txt')
            else:
                tpth = ipth.replace(f'.{post}', f'_{args.fvision}.txt')
                
            caption = ''
            with open(tpth, 'r') as fr:
                caption = fr.readline()
            
            prompt = caption_pre + caption + caption_post
            logging.info(f"{idx}: {caption}")
            ret = pipe(
                        prompt = prompt,
                        negative_prompt = neg,
                        height = args.inh,              # 目标生成分辨率 - 高, default 1024
                        width = args.inw,               # 目标生成分辨率 - 宽, default 1024
                        num_inference_steps=args.steps, 
                        guidance_scale=7.5,
                        generator = generator
                        ).images[0]
            opth = os.path.join(args.out, f'disney_{idx:04d}_0823.png')
            ret.save(opth)
            with open(opth.replace('.png', '.txt'), 'a') as fw:
                fw.write(f"prompt = {prompt}\n")
                fw.write(f"neg prompt = {neg}\n")
            with open(opth.replace('.png', '_caption.txt'), 'a') as fw:
                fw.write(f"{caption}\n")
            if args.save_in:
                shutil.copy(ipth, opth.replace('.png', '_in.png'))
    
    