#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File : download_snapsshot.py
@Author : Meisong
@Time: 2024/06/27 16:28:50
@Version: 1.0
@Contact: zhengmeisong.zms@alibaba-inc.com
huggingface模型下载，参考[AI之路]使用huggingface_hub优雅解决huggingface大模型下载问题
https://blog.csdn.net/popboy29/article/details/131979434  
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_snapshot_download.py
"""
import os, argparse, logging
from huggingface_hub import snapshot_download
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Process')
    parser.add_argument('--repo_id', type=str, default='stabilityai/stable-video-diffusion-img2vid-xt', help='copy from huggingface')
    parser.add_argument('--local_dir', type=str, default='/mnt/nfs0/hanzhang/models/StableDiffusion/svd_img2vid_xt', help='where to save')
    args = parser.parse_args()
    logging.info("args: {}".format(args))
    
    repo_id = args.repo_id
    local_dir = args.local_dir
    cache_dir = os.path.join(local_dir, 'cache')
    
    while True:
        try:
            snapshot_download(cache_dir = cache_dir,
                            local_dir = local_dir,
                            repo_id = repo_id,
                            local_dir_use_symlinks = False,
                            resume_download=True,
                            allow_patterns =["*.model", "*.json", "*.bin", "*.py", "*.md", "*.txt", "*.safetensors"],
                            etag_timeout = 10000,
                            )
        except Exception as e:
            print(e)
        else:
            print("下载完成")
            break
    
    
    