import torch
from safetensors.torch import load_file
import re

# 1. 定义路径
model_path = "ckpts/DGAF_VSR/DGAF_VSR_REDS_rename/diffusion_pytorch_model.safetensors"

# 2. 加载所有权重 (返回一个字典)
state_dict = load_file(model_path)

print(f"原始键数量: {len(state_dict)}")
print("前5个键:", list(state_dict.keys())[:5])

# 3. 定义重命名规则 (函数)
def rename_key(key):
    """
    根据需求修改键名。
    常见情况：
    - 移除 'unet.' 前缀
    - 替换 'time_embed' -> 'time_embedding'
    - 添加/移除特定层级
    """
    
    # 示例规则 2: 将 'time_embed' 改为 'time_embedding' (常见于 Diffusers 内部)
    if 'brushnet' in key:
        key = key.replace('brushnet', 'dgafnet')
        
    # 示例规则 3: 处理具体的层名 (例如将 'conv_in' 改为 'conv_in.weight')
    # 注意：safetensors 加载时通常已经包含了 .weight/.bias 后缀
    
    return key

# 4. 执行重命名并创建新字典
new_state_dict = {}
for old_key, value in state_dict.items():
    new_key = rename_key(old_key)
    new_state_dict[new_key] = value

print(f"重命名后键数量: {len(new_state_dict)}")
print("前5个新键:", list(new_state_dict.keys())[:5])

# 5. (可选) 保存到新的文件
output_path = "ckpts/DGAF_VSR/DGAF_VSR_REDS_rename/dgaf/diffusion_pytorch_model.safetensors"
from safetensors.torch import save_file
save_file(new_state_dict, output_path)
print(f"已保存至: {output_path}")

# 6. (可选) 加载到模型中测试
# 假设你有一个名为 MyDiffusionModel 的类
# model = MyDiffusionModel()
# missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
# print(f"缺失: {missing_keys}, 意外: {unexpected_keys}")
