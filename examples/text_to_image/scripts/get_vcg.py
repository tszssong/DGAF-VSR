#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : get_vcg.py
@Author : Meisong
@Time: 2024/08/26 13:45:52
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, shutil, argparse, logging
from tqdm import tqdm
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--lst', type=str, default='/large/hanzhang/data/vcg/vcg-00000-img.lst', help='')
    parser.add_argument('--out', type=str, default='/mnt/nfs0/hanzhang/data/style/t2i/vcg/vcg-00000-img_human-animal5k', help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--no_cuda', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    os.makedirs(args.out, exist_ok=True)
    count = 0
    with open(args.lst, 'r') as fr:
        for line in tqdm(fr.readlines()):
            ipth = line.strip()
            iname, post = os.path.basename(ipth).split('.')
            tpth = ipth.replace(
                f'.{post}', f'.title-content'
            )
            # with open(tpth, 'r') as fr:
            #     caption = fr.readline()
            #     if '无人' in caption:
            #         continue
            #     elif not (('人' in caption) or ('儿童' in caption) or ('男子' in caption) or ('女子' in caption)):
            #         continue
            with open(tpth, 'r') as fr:
                caption = fr.readline()
                if ('无人' in caption) and (not '动物'in caption):
                    continue
            with open(args.lst.replace('.lst', '_picked_human_animal.lst'), 'a') as fw:
                fw.write(f'{ipth}:{caption}\n')
            opth = os.path.join(args.out, f'{iname}.{post}')
            shutil.copy(ipth, opth)     
            count += 1
            if count >= 5000: break            