from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler, DDPMScheduler
from PIL import Image
import torch
import random
import numpy as np
import cv2
# from controlnet_aux import MidasDetector, ZoeDetector

from depth_anything_v2.dpt import DepthAnythingV2

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
    
    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    controlnet = ControlNetModel.from_pretrained(
        "/mnt/nfs0/hanzhang/models/ControlNet/xinsir-controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    )

    # when test with other base model, you need to change the vae also.
    vae = AutoencoderKL.from_pretrained("/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
        # scheduler=eulera_scheduler,
    )
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    pipe.to(f"cuda:{gpu}")
    pipe.load_lora_weights(lora_path)
    return pipe 

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
    parser.add_argument('--lora', type=str, default='/large/hanzhang/wkspace/styleWK/sdxllora_960x1_0809-155733/checkpoint-500', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--inw', type=int, default=768, help='')
    parser.add_argument('--inh', type=int, default=1024, help='')
    parser.add_argument('--nsteps', type=int, default=30, help='num_inference_steps')
    parser.add_argument('--s_ctl', type=float, default=0.5, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--trigger_word', type=str, default='xhs cartoon, ', help='')
    parser.add_argument('--do_not_use_trigger_word', action='store_true', help='not to use CUDA when available')
    parser.add_argument('--save_in', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    pipe = get_pipe(args.lora, args.gpu)
    if args.do_not_use_trigger_word:
        args.out = os.path.join(args.lora, f"CaptionwoTrigger_s{args.seed}_{args.inw}x{args.inh}_{args.nsteps}")
    else:
        args.out = os.path.join(args.lora, f"Caption_s{args.seed}_{args.inw}x{args.inh}_{args.nsteps}")
    os.makedirs(args.out, exist_ok=True)
    logging.info("args: {}".format(args))
    # negative_prompt = None
    negative_prompt = "text, glitch, deformed, mutated, ugly, disfigured, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, mole, freckles, skin spots, normal quality, monochrome, grayscale, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, watermark, mask, nipples, exposed breasts, exposed nipples, nude, naked, visible nipples"
    
    depth_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    depth_model.load_state_dict(torch.load('/mnt/nfs0/hanzhang/models/depth/depth-anything-v2-large/depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_model.eval()
    depth_model.to(f"cuda:{args.gpu}")
    
    with open(args.lst, 'r') as fr:
        content_lists = fr.readlines()
        for idx, line in enumerate(content_lists):
            lq_path = line.strip()
            iname, post = os.path.basename(lq_path).split('.')
            if args.fvision == '':
                tpth = lq_path.replace(f'.{post}', '.txt')
            else:
                tpth = lq_path.replace(f'.{post}', f'_{args.fvision}.txt')
                
            caption = ''
            with open(tpth, 'r') as fr:
                caption = fr.readline()
            caption = f'{args.word}, {caption}'
           
            input_image = cv2.imread(lq_path)
            orig_h, orig_w, _ = input_image.shape  
            input_image = cropUp_Middle_resize(input_image, args.inw, args.inh)
            depth_img = depth_model.infer_image(input_image)    # (1024, 768)
            controlnet_img = Image.fromarray(depth_img)
            logging.info(f"{idx+1}/{len(content_lists)} {iname}-{caption} {args.inw}x{args.inh}")
            image = pipe(   prompt=caption, 
                            negative_prompt=negative_prompt,
                            height = args.inh,              # 目标生成分辨率 - 高, default 1024
                            width = args.inw,               # 目标生成分辨率 - 宽, default 1024
                            num_inference_steps=args.nsteps, 
                            image=controlnet_img,  
                            controlnet_conditioning_scale=args.s_ctl,
                            guidance_scale=7.5,
                            generator = seed_all(args.seed)
                            ).images[0]
            opth = os.path.join(args.out, f"{iname}.png")
            image.save(opth)
            with open(opth.replace('.png', '.txt'), 'a') as fw:
                fw.write("prompt={caption}\n")
                fw.write("neg prompt={negative_prompt}\n")
            if args.save_in:
                shutil.copy(ipth, opth.replace('.png', '_in.png'))