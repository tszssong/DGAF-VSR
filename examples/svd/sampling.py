#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : sampling.py
@Author : Meisong
@Time: 2024/07/01 20:50:09
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
"""
import os, sys, argparse, logging
import torch

from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder, AutoencoderKL
from diffusers.utils import load_image, export_to_video

logging.basicConfig( format="[%(filename)s:%(lineno)d]%(asctime)s - %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO ,)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--inp', type=str, default='', help='')
    parser.add_argument('--out', type=str, default='', help='')
    parser.add_argument('--w', type=int, default=576, help='')
    parser.add_argument('--h', type=int, default=320, help='')
    parser.add_argument('--nframes', type=int, default=14, help='')
    parser.add_argument('--no_cuda', action='store_true', help='not to use CUDA when available')
    args = parser.parse_args()
    logging.info("args: {}".format(args))

    # pipe = StableVideoDiffusionPipeline.from_pretrained("/mnt/nfs0/hanzhang/models/StableDiffusion/svd_img2vid_xt8", \
    #                                                     torch_dtype=torch.float16, variant="fp16")
    # pipe = StableVideoDiffusionPipeline.from_pretrained("/mnt/nfs0/hanzhang/models/StableDiffusion/svd_img2vid", \
    #                                                     torch_dtype=torch.float16, low_cpu_mem_usage=False)
    # pipe = StableVideoDiffusionPipeline.from_pretrained("/mnt/nfs0/hanzhang/models/StableDiffusion/svd_img2vid", \
    #                                                     torch_dtype=torch.float32, low_cpu_mem_usage=False)
    pipe = StableVideoDiffusionPipeline.from_pretrained("/mnt/nfs0/hanzhang/models/StableDiffusion/svd_img2vid_xt", \
                                                        torch_dtype=torch.float32, low_cpu_mem_usage=False)
    pipe.to("cuda")
    # image = load_image("/mnt/nfs1/hanzhang/data/InPainting/Test/test0619/p1024x576/other_biz_465441404159_2.png")
    image = load_image("/mnt/nfs1/hanzhang/data/InPainting/Test/test0619/p1024x576/other_biz_465709531535_0.png")
    # image = load_image("/mnt/nfs1/hanzhang/data/InPainting/Test/test0619/p320x576/other_biz_465709531535_0.png")
    image = load_image("/large/hanzhang/wkspace/sftp/albumWK/lss.png")
    image = image.resize((args.w, args.h))    
    frames = pipe(image, height=args.h, width=args.w, num_frames=args.nframes, decode_chunk_size=8).frames[0]   #24757MiB / 32510MiB 14 fp32;  30851MiB / 32510MiB 20 fp32
    # frames = pipe(image, num_frames=3, decode_chunk_size=4).frames[0]       # 31054MiB /  32768MiB 7 fp16; 31936MiB /  32768MiB 3 fp32
    logging.info(f"svd ret:{len(frames)}, type={type(frames[0])}")
    for fidx, frame in enumerate(frames):
        save_path = os.path.join('./', f"{fidx:03d}.png")
        frame.save(save_path)
    
    # prefix = os.path.join('./', fname)
    # cmd = f'/usr/bin/ffmpeg  -f image2 -i %03d.png  -pix_fmt yuv420p  -crf 12 out.mp4'
    # os.system(cmd)
    export_to_video(frames, "generated.mp4", fps=7)
       
