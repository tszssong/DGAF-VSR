from diffusers import  StableDiffusionXLPipeline
import torch
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
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.to(f"cuda:{gpu}")
    pipe.load_lora_weights(lora_path)
    return pipe 

def filter_caption(inp):
    inp = inp.replace('A woman', 'a girl', 1)
    inp = inp.replace('A man', 'a boy', 1)
    inp = inp.replace('woman', 'girl', 1)
    inp = inp.replace('women', 'girls', 1)
    inp = inp.replace('man', 'boy', 1)
    inp = inp.replace('men', 'boys', 1)
    inp = inp.replace('The image shows', '', 1)
    return inp
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
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/Test/test40/lq40.lst', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--lora', type=str, default='/large/hanzhang/wkspace/styleWK/sdxllora_960x1_0809-155733/checkpoint-10000', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--inw', type=int, default=768, help='')
    parser.add_argument('--inh', type=int, default=1024, help='')
    parser.add_argument('--nsteps', type=int, default=30, help='num_inference_steps')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--fvision', type=str, default='', help='florence2caption, florence2Dcaption, florence2MDcaption')
    parser.add_argument('--word', type=str, default='xhs cartoon', help='xhs cartoon, cmodisndisney, cmodisndisneyv1')
    parser.add_argument('--do_not_use_trigger_word', action='store_true', help='not to use CUDA when available')
    parser.add_argument('--save_in', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    pipe = get_pipe(args.lora, args.gpu)
    prefix = os.path.basename(args.lst).split('.')[0]
    args.out = os.path.join(args.lora, f"T2I_{prefix}_{args.fvision}_s{args.seed}_{args.inw}x{args.inh}_{args.nsteps}")
    os.makedirs(args.out, exist_ok=True)
    logging.info("args: {}".format(args))
    # negative_prompt = None
    negative_prompt = "text, glitch, deformed, mutated, ugly, disfigured, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, mole, freckles, skin spots, normal quality, monochrome, grayscale, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature, watermark, mask, nipples, exposed breasts, exposed nipples, nude, naked, visible nipples"
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
            # caption = filter_caption(caption)
            caption = f'{args.word}, {caption}'
            logging.info(f"{idx+1}/{len(content_lists)} {iname}-{caption} {args.inw}x{args.inh}")
            image = pipe(   prompt=caption, 
                            negative_prompt=negative_prompt,
                            height = args.inh,              # 目标生成分辨率 - 高, default 1024
                            width = args.inw,               # 目标生成分辨率 - 宽, default 1024
                            num_inference_steps=args.nsteps, 
                            guidance_scale=7.5,
                            generator = seed_all(args.seed)
                            ).images[0]
            opth = os.path.join(args.out, f"{iname}.png")
            image.save(opth)
            with open(opth.replace('.png', '.txt'), 'a') as fw:
                fw.write(f"prompt={caption}\n")
                fw.write(f"neg prompt={negative_prompt}\n")
            if args.save_in:
                shutil.copy(ipth, opth.replace('.png', '_in.png'))