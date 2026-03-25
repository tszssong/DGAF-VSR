from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLControlNetImg2ImgPipeline
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, DDPMScheduler
from PIL import Image
import torch
import random
import numpy as np
import cv2
from controlnet_aux import MidasDetector, ZoeDetector

from depth_anything_v2.dpt import DepthAnythingV2

# controlnet_conditioning_scale = 1.0  
# prompt = "your prompt, the longer the better, you can describe it as detail as possible"
# negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


# # need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance

# img = cv2.imread("your original image path")



# images = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     image=controlnet_img,
#     controlnet_conditioning_scale=controlnet_conditioning_scale,
#     width=new_width,
#     height=new_height,
#     num_inference_steps=30,
#     ).images

# images[0].save(f"your image save path, png format is usually better than jpg or webp in terms of image quality but got much bigger")

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
    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    controlnet = ControlNetModel.from_pretrained(
        "/mnt/nfs0/hanzhang/models/ControlNet/xinsir-controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16)
    # when test with other base model, you need to change the vae also.
    vae = AutoencoderKL.from_pretrained("/mnt/nfs0/hanzhang/models/ControlNet/madebyollin-sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
        # scheduler=eulera_scheduler,   # default: EulerDiscreteScheduler
    )
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
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
@File : myTestI2ICtl.py
@Author : Meisong
@Time: 2024/08/09 14:52:25
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
i2i + controlnet
"""
import os, shutil, argparse, logging
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s: %(message)s", datefmt="%m/%d %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/input_labeled.lst', help='')
    # parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/testORI.lst', help='')
    # parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/testORI4.lst', help='')
    parser.add_argument('--gt', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/testDR/output_wo_watermark', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--lora', type=str, default='/large/hanzhang/wkspace/styleWK/sdxllora_960x1_0809-155733/checkpoint-10000', help='')
    parser.add_argument('--word', type=str, default='cartoon', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--steps', type=int, default=30, help='')
    parser.add_argument('--inw', type=int, default=768, help='')
    parser.add_argument('--inh', type=int, default=1024, help='')
    parser.add_argument('--s_i2i', type=float, default=0.6, help='')
    parser.add_argument('--s_ctl', type=float, default=0.5, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--save_in', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    
    pipe = get_pipe(args.lora, args.gpu)
    prefix = os.path.basename(args.lst).split('.')[0]
    # args.out = os.path.join(args.lora, f"{prefix}_i2i{args.s_i2i}_da{args.s_ctl}_{args.inw}x{args.inh}-{args.steps}DDPM")
    args.out = os.path.join(args.lora, f"{prefix}_i2i{args.s_i2i}_da{args.s_ctl}_{args.inw}x{args.inh}-{args.steps}")
    os.makedirs(args.out, exist_ok=True)
    logging.info("args: {}".format(args))
    # negative_prompt = None
    negative_prompt = "text, glitch, deformed, mutated, ugly, disfigured, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, mole, freckles, skin spots, normal quality, monochrome, grayscale, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, watermark, mask, nipples, exposed breasts, exposed nipples, nude, naked, visible nipples"
    
    processor_zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
    processor_midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

    depth_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    depth_model.load_state_dict(torch.load('/mnt/nfs0/hanzhang/models/depth/depth-anything-v2-large/depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_model.eval()
    depth_model.to(f"cuda:{args.gpu}")
    
    
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
            caption = f'xhs {args.word}, ' + caption.strip()
            # caption = 'xhs cartoon, ' + caption.strip() #+ "depth of field"
            # if ("0003" in iname) or ("0007" in iname) or ("0010" in iname) or ("0012" in iname) or ("0017" in iname):
            #     caption = caption + ", smile, happy, mouth open"
            #+ ', big eyes, happy' 
            input_image = cv2.imread(lq_path)
            
            
            orig_h, orig_w, _ = input_image.shape  
            input_image = cropUp_Middle_resize(input_image, args.inw, args.inh)
            
            # depth_img = processor_zoe(input_image, output_type='cv2')
            # depth_img = processor_midas(input_image, output_type='cv2') # (704, 512, 3)
            # depth_img = cv2.resize(depth_img, (args.inw, args.inh), interpolation=cv2.INTER_LANCZOS4)
            depth_img = depth_model.infer_image(input_image)    # (1024, 768)
            controlnet_img = Image.fromarray(depth_img)
            
            if args.save_in:
                cv2.imwrite(opth.replace('_1out.png', "_2in.png"), input_image)
                cv2.imwrite(opth.replace('_1out.png', "_3depth.png"), depth_img)
                shutil.copy(lq_path, opth.replace('_1out.png', '_4lq.png'))
                shutil.copy(gt_path, opth.replace('_1out.png', '_0gt.png'))
            logging.info(f"{idx+1}/{len(content_lists)} {iname} size = {orig_w}x{orig_h} to {args.inw}x{args.inh}")
            init_input = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            
            
            image = pipe(   prompt=caption, 
                            negative_prompt=negative_prompt,
                            image=init_input,    
                            strength=args.s_i2i,           # default = 0.8   
                            control_image=controlnet_img,   
                            controlnet_conditioning_scale=args.s_ctl,
                            height = args.inh,              # 目标生成分辨率 - 高, default 1024
                            width = args.inw,               # 目标生成分辨率 - 宽, default 1024
                            num_inference_steps=args.steps, 
                            guidance_scale=7.5,
                            generator = seed_all(args.seed),
                            ).images[0]
            image.save(opth)
            with open(opth.replace('.png', '.txt'), 'a') as fw:
                fw.write(f"prompt={caption}\n")
                # fw.write(f"neg prompt={negative_prompt}\n")