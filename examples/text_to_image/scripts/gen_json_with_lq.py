#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : gen_json_from_text.py
@Author : Meisong
@Time: 2024/08/09 12:16:24
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, sys, argparse, logging
logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--inp', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/pairDR/train15gt', help='')
    parser.add_argument('--lq', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/pairDR/train15lq', help='')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/pairDR/train15gt.lst', help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    parser.add_argument('--mini', action='store_true', help='极简提示词：xhs cartoon, a cat')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    spth = args.lst.replace('.lst', '.jsonl')
    with open(args.lst, 'r') as fr:
        for line in fr.readlines():
            ipth = line.strip()
            tpth = ipth.replace('.png', '.txt')
            with open(tpth, 'r') as fr:
                for line in fr.readlines():
                    caption = line.strip()
                    # caption = f"xhs cartoon, {caption}"
                    if args.mini:
                        caption = f"xhs cartoon, {caption.split(',')[1]}"
            iname = os.path.basename(ipth)
            with open(spth, 'a') as fw:
                # lq_path = os.path.join(args.lq, iname.replace('.png', '.jpg'))
                lq_path = os.path.join(args.lq, iname)
                fw.write(f"{{\"file_name\": \"{iname}\", \"text\": \"{caption}\", \"lq\": \"{lq_path}\"}}\n")
            