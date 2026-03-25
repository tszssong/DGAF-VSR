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
    parser.add_argument('--inp', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/trainDR/train41f2', help='')
    parser.add_argument('--lst', type=str, default='/mnt/nfs0/hanzhang/data/style/pet3d/trainDR/train41f2.lst', help='')
    parser.add_argument('--word', type=str, default='tbs cartoon', help='xhs cartoon, cmodisndisney, cmodisndisneyv1')
    parser.add_argument('--fvision', type=str, default='florence2MDcaption', help='florence2caption, florence2Dcaption, florence2MDcaption')
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
                    caption = caption.replace("The image is an illustration of", "The image shows", 1)
                    caption = caption.replace("The image is a digital illustration of", "The image shows", 1)
                    caption = caption.replace("The image is a cartoon illustration of", "The image shows", 1)
                    caption = caption.replace("The image shows a cartoon", "The image shows", 1)
                    caption = caption.replace("The image is a 3D rendering of", "The image shows", 1)
                    caption = caption.replace("The image shows a 3D rendering of", "The image shows", 1)
                    caption = caption.replace("The items are drawn in a simple, cartoon-like style with thin lines and bright colors.", "", 1)
                    caption = caption.replace("The image shows a close-up of a", "A", 1)
                    caption = caption.replace("The image is a close-up of a", "A", 1)
                    caption = caption.replace("The image shows a", "A", 1)
                    caption = caption.replace("The image is a", "A", 1)
                    caption = caption.replace("A cartoon ", "A ", 1)
                    caption = caption.replace(", and the overall mood of the image is cheerful and playful", "", 1)
                    caption = caption.replace(" and the overall mood of the image is cheerful and playful", "", 1)
                    caption = caption.replace(", and the overall mood of the image is peaceful and serene", "", 1)
                    caption = caption.replace(" and the overall mood of the image is peaceful and serene", "", 1)
                    caption = caption.replace(", and the overall scene is peaceful and serene", "", 1)
                    # caption = caption.replace(", and the overall scene is peaceful and serenee", "", 1)
                    
                    captions = caption.split('. ')
                    if ("The overall style" in captions[-1]) or ("the overall style" in captions[-1]) or \
                        ("The image is drawn" in captions[-1]) or ("cartoon-like" in captions[-1]) or \
                        ("crayon" in captions[-1]) or ("crayons" in captions[-1]) or ("animated" in captions[-1]) or\
                             ("3D rendering" in captions[-1]):
                        print(captions[-1])
                        caption = ". ".join(captions[:-1])
                    captions = caption.split('. ')
                    if ("The overall mood" in captions[-1]):
                        print(captions[-1])
                        caption = ". ".join(captions[:-1])
                        
                    caption = f"{args.word}, {caption}"
                    
            iname = os.path.basename(ipth)
            with open(spth, 'a') as fw:
                fw.write(f"{{\"file_name\": \"{iname}\", \"text\": \"{caption}\"}}\n")
            