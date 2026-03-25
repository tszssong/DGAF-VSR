#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : ci_filter_disney.py
@Author : Meisong
@Time: 2024/08/20 11:52:24
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, sys, argparse, logging
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/t2i/disneyXL/0820/train400.lst', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--no_cuda', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    
    spth = args.lst.replace('.lst', '.jsonl')
    with open(args.lst, 'r') as fr:
        for line in fr.readlines():
            ipth = line.strip()
            iname, post = os.path.basename(ipth).split('.')
            tpth = ipth.replace('.{post}', '_ci.txt')
            with open(tpth, 'r') as fr:
                for line in fr.readlines():
                    caption = line.strip()
                    caption = f"cmodisndisney, {caption}"
                    print(f"{iname}-before:{caption}")
                    caption = caption.strip()
                    caption = caption.replace('there is ', '', 1)   #从左到右，只替换一次
                    caption = caption.replace('that is ', '', 1)  
                    caption = caption.replace('there are ', '', 1)
                    caption = caption.replace('araffe ', 'a ', 1)
                    caption = caption.replace('arafed ', 'a ', 1)
                    caption = caption.replace('a image of ', '', 1)
                    caption = caption.replace('an image of ', '', 1)
                    caption = caption.replace('a cartoon', 'a', 1)
                    caption = caption.replace('anime', 'a', 1)
                    caption = caption.replace('anime - style', 'a', 1)
                     
            caption = f'{args.word}, ' + caption
            print(f"{iname}-after:{caption}")
            with open(ipth.replace(f".{post}", f"_ci.txt"), 'a') as fw:
                fw.write(f"{caption}")
            with open(spth, 'a') as fw:
                fw.write(f"{{\"file_name\": \"{iname}\", \"text\": \"{caption}\"}}\n")