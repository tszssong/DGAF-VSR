#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : test.py
@Author : Meisong
@Time: 2024/10/10 17:16:42
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, shutil, argparse, logging, cv2
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)

import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/train0923/gt.lst', help='')
    parser.add_argument('--out', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/train0923/gt_', help='')
    parser.add_argument('--scale', type=int, default=2, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--no_cuda', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    
    args.out = args.out + f"x{args.scale}"
    os.makedirs(args.out, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    
    model = RealESRGAN(device, scale=args.scale)
    model.load_weights(f'/mnt/nfs0/hanzhang/models/RealESRGAN/ai-forever-Real-ESRGAN/RealESRGAN_x{args.scale}.pth', download=True)
    with open(args.lst, 'r') as fr:
        for line in tqdm(fr.readlines()):

            path_to_image = line.strip()
            iname = os.path.basename(path_to_image)
            image = Image.open(path_to_image).convert('RGB')

            sr_image = model.predict(image)

            # sr_image.save('srx8_image.png')

            cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
            cv_sr = cv2.cvtColor(np.asarray(sr_image),cv2.COLOR_RGB2BGR)  

            cv_sr_resized = cv2.resize(cv_sr, image.size)
            # save_path = os.path.join(args.out, iname.replace('.png', f'_sr{args.scale}-r.png'))
            save_path = os.path.join(args.out, iname)
            cv2.imwrite(save_path, cv_sr_resized)
            # shutil.copy(path_to_image, os.path.join(args.out, iname))