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
    parser.add_argument('--inp', type=str, default='/mnt/nfs0/hanzhang/data/style/crayon/train0926/gt', help='')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/crayon/train0926/gt.lst', help='')
    parser.add_argument('--word', type=str, default='tbs crayon', help='tbs cartoon, cmodisndisney, cmodisndisneyv1')
    parser.add_argument('--fvision', type=str, default='florence2caption', help='florence2caption, florence2Dcaption, florence2MDcaption')
    # parser.add_argument('--mini', action='store_true', help='极简提示词：xhs cartoon, a cat')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    spth = args.lst.replace('.lst', f'_{args.fvision}_replaced.jsonl')
    with open(args.lst, 'r') as fr:
        for line in fr.readlines():
            ipth = line.strip()
            # if 'animal' in ipth:
            #     tpth = ipth.replace('.png', f'_florence2Dcaption.txt')
            # else:
            #     tpth = ipth.replace('.png', f'_florence2MDcaption.txt')
            tpth = ipth.replace('.png', f'_{args.fvision}.txt')
            with open(tpth, 'r') as fr:
                for line in fr.readlines():
                    caption = line.strip()
                    caption = caption.replace("\"", "\'")   #prompt里的双引号转换成单引号，否则训练代码读取jsonl报错
                    caption = caption.replace("drawing of a ", "", 1) 
                    caption = caption.replace("A drawing of ", "", 1)   #item_item_XHS_1725550049087
                    caption = caption.replace("is walking on a leash ", "sitting on a leash ", 1)   #animal_dog_XHS_1725523688764
                    
                    captions = caption.split('. ')
                   
                        
                    caption = f"{args.word}, {caption}"
                    
            iname = os.path.basename(ipth)
            with open(spth, 'a') as fw:
                fw.write(f"{{\"file_name\": \"{iname}\", \"text\": \"{caption}\"}}\n")
            