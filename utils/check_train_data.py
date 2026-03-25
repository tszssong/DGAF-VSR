#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : check_train_data.py
@Author : Meisong
@Time: 2024/12/17 16:21:47
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, sys, argparse, logging
import cv2
from tqdm import tqdm
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--lst', type=str, default='/large/hanzhang/data/ll/DFHQ/DFHQ.lst', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--no_cuda', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    flog = args.lst.replace('.lst', '.error')
    with open(args.lst, 'r') as fr:
        for line in tqdm(fr.readlines()):
            fpath = line.strip()
            im = cv2.imread(fpath)
            try:
                h, w, c = im.shape
                if h<512 or w < 512:
                    with open(flog, 'a') as fw:
                        fw.write(f"{fpath}: {w}x{h}")
            except:
                print(fpath) 
                with open(flog, 'a') as fw:
                    fw.write(line)