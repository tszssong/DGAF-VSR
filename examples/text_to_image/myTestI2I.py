from diffusers import DiffusionPipeline
from diffusers import  StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import PIL, cv2
from PIL import Image

import numpy as np
def seed_all(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return torch.Generator("cuda").manual_seed(seed)    # used by diffusers pipe

def get_pipe(lora_path, gpu):
    base_model_path = "/mnt/nfs0/hanzhang/models/StableDiffusion/sd-xl-base-1.0"
    # pipe = DiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
    # load SDXL pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.to(f"cuda:{gpu}")
    pipe.load_lora_weights(lora_path)
    return pipe 
'''
    高>宽：裁切上半部分后resize
    宽>高：裁切中间部分后resize
'''
def cropUp_Middle_resize(im, dw, dh):
    orig_h, orig_w, _ = im.shape
    if orig_h/float(orig_w) > dh/float(dw):
        new_h = int(dh*orig_w/dw)
        im = im[:new_h,:,:]
        im = cv2.resize(im, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_w = int(dw*orig_h/dh)
        cw = orig_w//2
        im = im[:,(orig_w-new_w)//2:(orig_w+new_w)//2,:]
        im = cv2.resize(im, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    return im


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
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/input_labeled.lst', help='')
    # parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/testORI.lst', help='')
    # parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/testORI5.lst', help='')
    parser.add_argument('--gt', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/output_wo_watermark', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--lora', type=str, default='/large/hanzhang/wkspace/styleWK/sdxllora_960x1_0809-155733/checkpoint-10000', help='')
    parser.add_argument('--word', type=str, default='xhs cartoon', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--inw', type=int, default=768, help='')
    parser.add_argument('--inh', type=int, default=1024, help='')
    parser.add_argument('--s_i2i', type=float, default=0.6, help='')
    parser.add_argument('--steps', type=int, default=30, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--save_in', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    pipe = get_pipe(args.lora, args.gpu)
    prefix = os.path.basename(args.lst).split('.')[0]
    args.out = os.path.join(args.lora, f"{prefix}_i2i{args.seed}_{args.inw}x{args.inh}_s{args.s_i2i}-{args.steps}")
    os.makedirs(args.out, exist_ok=True)
    logging.info("args: {}".format(args))
    # negative_prompt = None
    negative_prompt = "text, glitch, deformed, mutated, ugly, disfigured, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, mole, freckles, skin spots, normal quality, monochrome, grayscale, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, watermark, mask, nipples, exposed breasts, exposed nipples, nude, naked, visible nipples"
    with open(args.lst, 'r') as fr:
        content_lists = fr.readlines()
        for idx, line in enumerate(content_lists):
            lq_path = line.strip()
            iname, post = os.path.basename(lq_path).split('.')
            gt_path = os.path.join(args.gt, f"{iname}.{post}")
            text_path = lq_path.replace(f'.{post}', '.txt')
            if 'ci' in args.lora:
                text_path = lq_path.replace(f'.{post}', '_ci.txt')
            opth = os.path.join(args.out, f"{iname}_1out.png")
            with open(text_path, 'r') as fr:
                caption = fr.readline()
            caption = f'{args.word}, ' + caption.strip()
            input_image = cv2.imread(lq_path)
            orig_h, orig_w, _ = input_image.shape   # TODO:等比例
            input_image = cropUp_Middle_resize(input_image, args.inw, args.inh)
            if args.save_in:
                cv2.imwrite(opth.replace('_1out.png', "_2in.png"), input_image)
                shutil.copy(lq_path, opth.replace('_1out.png', '_3lq.png'))
                shutil.copy(gt_path, opth.replace('_1out.png', '_0gt.png'))
            logging.info(f"{idx+1}/{len(content_lists)} {iname} size = {orig_w}x{orig_h} to {args.inw}x{args.inh}")
            init_input = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            image = pipe(   prompt=caption, 
                            negative_prompt=negative_prompt,
                            image = init_input,
                            strength = args.s_i2i,
                            height = args.inh,              # 目标生成分辨率 - 高, default 1024
                            width = args.inw,               # 目标生成分辨率 - 宽, default 1024
                            num_inference_steps=args.steps, 
                            guidance_scale=7.5,
                            generator = seed_all(args.seed),
                            ).images[0]
            image.save(opth)
            with open(opth.replace('.png', '.txt'), 'a') as fw:
                fw.write(f"prompt={caption}\n")
            #     fw.write(f"neg prompt={negative_prompt}\n")