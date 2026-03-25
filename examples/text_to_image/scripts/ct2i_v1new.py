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
c站demo分辨率为1024x1536,宽高比1:1.5    https://civitai.com/images/1679391
显存与耗时[正方形分辨率(1024x1024)会导致生成多个重复物体]：
768x1024 17833MiB / 32510MiB 25/25 [00:07<00:00
768x1152  21515MiB / 32510MiB 25/25  25/25 [00:09<00:00,  2.53it/s]
1024x1024 28605MiB / 32510MiB 25/25 [00:13<00:00,
25/25 [00:15<00:00,  1.66it/s]
"""

import numpy as np
import PIL, cv2
from PIL import Image
import torch
from diffusers import  StableDiffusionPipeline, DPMSolverMultistepScheduler
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


def prompt_content():
    # list_subjects = ['1 girl', '1 boy', '1 girl and 1 cat', '1 boy and 1 cat', '1 girl and 1 dog', '1 boy and 1 cat']
    # list_subjects = ['a girl, solo', 'a little girl, solo', 'a young girl, solo', 'a boy, solo', 'a little boy, solo', 'a young boy, solo']
    list_subjects = ['a girl', 'a little girl', 'a young girl', 
                     'a boy,', 'a little boy', 'a young boy',
                     'a man', 'a young man', 'two men', 
                     'a woman', 'a young woman', 'two young women', 'three young women', 'two women',
                     'a man and woman', 'two children, a boy and a girl'
                     ]
    subject = random.choice(list_subjects)
    dict_hair_colors = {'black': 30, 'brown': 20, 'yellow':10, 'red':10, 'grey':5, 'blue':5, 'green':5, 'pink':5, 'purple':5}
    hair_color = random_weight(dict_hair_colors)
    hair_style = f'{hair_color} hair'
    if (('girl' in subject) and random.randint(0,5)==0):    # or 'boy' in subject:
        hair_style += ', short hair'
    if 'girl' in subject:
        r = random.randint(0,5)
        if r == 0:
            hair_style += ', hair clasp'
        elif r == 1:
            hair_style += ', bowknot'
       
    if random.randint(0,3)==0:
         hair_style += ', hat'
    r = random.randint(0,5)
    if r == 0:
        hair_style += ', mask'
    elif r == 1:
        hair_style += ', sunglasses'
    elif r == 2:
        hair_style += ', glass'
      
    dict_eye_colors = {'brown': 20, 'yellow':20, 'grey':20, 'blue':20, 'green':20}
    eye_color = random_weight(dict_eye_colors)
    eye_style = f'{eye_color} eyes'
    
    up_wearing = ['T-shirt', 'shirt', 'jacket', 'sweater', 'vest', 'leather jacket']
    down_wearing = ['skirt', 'pants', 'shorts', 'jeans', 'casual pants']
    all_wearing = ['suit', 'sportswear', 'evening gown', 'trench coat', 'pajamas']
    color_wearing = ['red', 'blue', 'green', 'black', 'white', 'grey', 'yellow', 'pink', 'purple',
                     'orange', 'brown', 'beige', 'gold', 'silver', 'peach', 'burgundy', 'olive green',
                     'sky blue', 'coral', 'teal', 'plaid', 'floral']
  
    if random.randint(0,2):
        up_color = random.choice(color_wearing)
        up = random.choice(up_wearing)
        down_color = random.choice(color_wearing)
        down = random.choice(down_wearing)
        wearing = f'{up_color} {up}, {down_color} {down}'
    else:
        color = random.choice(color_wearing)
        all = random.choice(all_wearing)
        wearing = f'{color} {all}'
    dict_bag = {'bag':20, 'school bag':20, 'hand bag':20, 'daypack':20, 'suitcase':10, 'luggage':10}
    if random.randint(0,3)==0:
        bag_type = random_weight(dict_bag)
        wearing = f'{wearing}, {bag_type}'
        
    list_act = ['sitting', 'standing', 'laying', 'walking', 'running', 'riding a bicycle']   
    act = random.choice(list_act)
    dict_body = {'full body':20, 'profile facing the screen':20, 'crossed arms':20}
    if random.randint(0,1):
        body = random_weight(dict_body)
        act = f'{act}, {body}'
    
    if random.randint(0,2) and (('sitting' in act) or ('standing' in act) or ('laying' in act)):    #室内
        dict_scenery = {'office':20, 'bedroom':20,  'classroom':20, 'coffee shop':20, 'indoor':20}
        list_scenery = ['floor', 'bed', 'table', 'desk', 'chair', 'lamp', 'cup', 'window', 'sofa']
    elif random.randint(0,2) and (('sitting' in act) or ('standing' in act) or ('walking' in act) or ( 'riding a bicycle' in act)):  #城市户外
        dict_scenery = {'plaza': 20, 'market':5}
        list_scenery = ['blue sky', 'cloud', 'car', 'road', 'tree', 'flower', 'house', 'building']
    else:                                                                                           #自然场景
        dict_scenery = {'farm':10, 'park':20, 'outdoor':20}
        list_scenery = ['lake', 'river', 'beach', 'mountain', 'snow', 'blue sky', 'cloud', 'grass', 'car', 'road', 'tree', 'flower', 'house']
    scenery = random_weight(dict_scenery)
    item = random.sample(list_scenery, 2)
    scenery = f'{scenery}, {item[0]}, {item[1]}'

    dict_mood = {'happy': 50, 'sad':10, 'angry':10, 'Surprise':20, 'fear':10}
    mood = random_weight(dict_mood)
    
    caption = f'{subject}, {eye_style}, {hair_style}, {wearing}, {act}, {scenery}, {mood}'
    return caption

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
    base_model_path = "/mnt/nfs0/hanzhang/models/civitai/disneyStyleV1_v10"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    # load SDXL pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        use_safetensors=False,
        safety_checker=None,
        torch_dtype=torch.float16,
        add_watermarker=False,
        scheduler = scheduler,
    )
    pipe = pipe.to("cuda")
    return pipe

import os, shutil, argparse, logging
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--out', type=str, default='/mnt/nfs0/hanzhang/data/style/t2i/disney', help='')
    parser.add_argument('--seed', type=int, default=3838446490, help='')
    parser.add_argument('--steps', type=int, default=25, help='')
    parser.add_argument('--inw', type=int, default=768, help='')
    parser.add_argument('--inh', type=int, default=1152, help='')
    parser.add_argument('--save_in', action='store_true', help='save input')
    args = parser.parse_args()
    
    subpth=f"cDisneyV1_{args.inw}x{args.inh}seed{args.seed}steps{args.steps}-0905"
    args.out = os.path.join(args.out, f'{subpth}')
    logging.info("args: {}".format(args))
    
    os.makedirs(args.out, exist_ok=True)
    
    generator = seed_all(args.seed)
    pipe = get_pipe()
    pre = "masterpiece, best quality, 8k, official art, cinematic light, ultra high res, "
    # post = ", happy, perfect eyes, highly detailed beautiful expressive eyes, detailed eyes, detailed, intricate, high quality"
    # caption = "1 girl, brown hair, short hair, flannel shirt, farm scenery" standing sideways
    # post = ", detailed, intricate, high quality"
    post = ''
    neg = "EasyNegative, badhandv4, (worst quality, low quality:1.3), logo, watermark, signature, text"
    # neg = "EasyNegative, badhandv4, (worst quality, low quality:1.3), low quality, bad anatomy, text, glitch, deformed, mutated, ugly, disfigured,  extra hand, extra leg, extra arm, extra head, extra fingers, extra body parts"
    # caps = ["a girl, black hair, red eyes, bag, siting, with her profile facing the screen, looking into the distance.",
    #         'a boy, yellow hair, green eyes, hand bog, standing sideways, fullbody.']
    # for sidx in range(len(caps)):
        # caption = caps[sidx]
    for sidx in range(5000):
        caption = prompt_content()
        prompt = pre + caption + post
        logging.info(f"{sidx}: {caption}")
        ret = pipe(
                    prompt = prompt,
                    negative_prompt = neg,
                    height = args.inh,              # 目标生成分辨率 - 高, default 1024
                    width = args.inw,               # 目标生成分辨率 - 宽, default 1024
                    num_inference_steps=args.steps, 
                    guidance_scale=7.5,
                    generator = generator
                    ).images[0]
        tpth = os.path.join(args.out, f'disney_{sidx:04d}_0823.png')
        ret.save(tpth)
        with open(tpth.replace('.png', '.txt'), 'a') as fw:
            fw.write(f"prompt = {prompt}\n")
            fw.write(f"neg prompt = {neg}\n")
        with open(tpth.replace('.png', '_caption.txt'), 'a') as fw:
            fw.write(f"{caption}\n")
    
    