#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : test.py
@Author : Meisong
@Time: 2024/10/10 17:16:42
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, sys, argparse, logging, cv2
# logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Data Process')
#     parser.add_argument('--inp', type=str, default='', help='')
#     parser.add_argument('--out', type=str, default='', help='')
#     parser.add_argument('--gpu', type=int, default=0, help='')
#     parser.add_argument('--no_cuda', action='store_true', help='not to use CUDA when available')
#     args = parser.parse_args()
#     logging.info("args: {}".format(args))
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = RealESRGAN(device, scale=4)
# model.load_weights('/mnt/nfs0/hanzhang/models/RealESRGAN/ai-forever-Real-ESRGAN/RealESRGAN_x4.pth', download=True)

# path_to_image = '/mnt/nfs0/hanzhang/data/style/pet3d/test/gt/animal_cat_XHS_1724211609830.png'
# image = Image.open(path_to_image).convert('RGB')

# sr_image = model.predict(image)

# sr_image.save('sr_image.png')

# cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
# cv_sr = cv2.cvtColor(np.asarray(sr_image),cv2.COLOR_RGB2BGR)  

# cv_sr_resized = cv2.resize(cv_sr, image.size)
# cv2.imwrite('srx4_image_resized.png', cv_sr_resized)


model = RealESRGAN(device, scale=2)
model.load_weights('/mnt/nfs0/hanzhang/models/RealESRGAN/ai-forever-Real-ESRGAN/RealESRGAN_x2.pth', download=True)

path_to_image = '/mnt/nfs0/hanzhang/data/style/pet3d/test/gt/animal_cat_XHS_1724211609830.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('srx2_image.png')

cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
cv_sr = cv2.cvtColor(np.asarray(sr_image),cv2.COLOR_RGB2BGR)  

cv_sr_resized = cv2.resize(cv_sr, image.size)
cv2.imwrite('srx2_image_resized.png', cv_sr_resized)



model = RealESRGAN(device, scale=8)
model.load_weights('/mnt/nfs0/hanzhang/models/RealESRGAN/ai-forever-Real-ESRGAN/RealESRGAN_x8.pth', download=True)

path_to_image = '/mnt/nfs0/hanzhang/data/style/pet3d/test/gt/animal_cat_XHS_1724211609830.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('srx8_image.png')

cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  
cv_sr = cv2.cvtColor(np.asarray(sr_image),cv2.COLOR_RGB2BGR)  

cv_sr_resized = cv2.resize(cv_sr, image.size)
cv2.imwrite('srx8_image_resized.png', cv_sr_resized)