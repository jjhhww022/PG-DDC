# util/pgddc_density_utils.py

import numpy as np
import torch
import torch.nn.functional as F_nn
from typing import List, Tuple
import cv2
import os
from transformers import CLIPModel, CLIPProcessor
from util.visualizer import renorm
import torch.nn as nn # 用于创建卷积层

# --- 1. 高效的动态高斯核生成 (优化版) ---
def get_dynamic_gaussian_kernel(prompt_text: str, device: str = "cuda", sigma_scale: float = 1.0, min_kernel_size: int = 3) -> torch.Tensor:
    """
    根据提示文本生成动态高斯核。
    该实现缓存核，避免重复计算。
    Args:
        prompt_text: 提示文本
        device: 设备
        sigma_scale: 核大小缩放因子
        min_kernel_size: 最小核尺寸 (必须为奇数)
    Returns:
        kernel: [1, 1, k, k] 的高斯核
    """
    # 确保最小核尺寸为奇数
    if min_kernel_size % 2 == 0:
        min_kernel_size += 1

    # 简单的提示到 sigma 映射（可根据需要扩展）
    sigma_map = {"small": 2.0, "medium": 4.0, "large": 8.0}
    base_sigma = 4.0 # 默认 sigma
    for size_word in ["small", "large", "medium"]:
        if size_word in prompt_text.lower():
            base_sigma = sigma_map[size_word]
            break
    sigma = base_sigma * sigma_scale

    # 确保 sigma 不会太小，导致 kernel_size 过小
    min_sigma_for_kernel = min_kernel_size / 6.0
    sigma = max(sigma, min_sigma_for_kernel)

    kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
    kernel_size = max(kernel_size, min_kernel_size) # 确保不低于最小值
    print(f"DEBUG: sigma={sigma}, kernel_size={kernel_size}") # 调试信息

    # 使用 torch.linspace 创建坐标
    x_cord = torch.arange(kernel_size, dtype=torch.float, device=device) - (kernel_size - 1) / 2
    # 计算高斯值
    gauss = torch.exp(-x_cord ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    # 构造 2D 核
    kernel_1d = gauss.unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    return kernel_2d.unsqueeze(0).unsqueeze(0)

# --- 2. 语义增强函数 ---
def enhance_feature_with_prompt(img_feat: torch.Tensor, prompt_text: str, clip_model: CLIPModel, clip_processor: CLIPProcessor, device: str):
    """
    使用 CLIP 语义向量增强图像特征。
    Args:
        img_feat: [B, C, H, W] 图像特征
        prompt_text: 提示文本
        clip_model: CLIP 模型
        clip_processor: CLIP 处理器
        device: 设备
    Returns:
        enhanced_feat: [B, C, H, W] 语义增强后的特征
    """
    B, C, H, W = img_feat.shape
    # 获取文本特征
    text_tokens = clip_processor.tokenizer([prompt_text], return_tensors="pt", padding=True).to(device)
    text_embeds = clip_model.get_text_features(
        input_ids=text_tokens["input_ids"],
        attention_mask=text_tokens.get("attention_mask", None)
    ) # [1, D]
    text_embeds = F_nn.normalize(text_embeds, dim=-1) # [1, D]

    # 将图像特征 reshape 为 [B*H*W, C] 并归一化
    img_feat_flat = img_feat.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
    img_feat_flat_norm = F_nn.normalize(img_feat_flat, dim=-1) # [B*H*W, C]

    # 计算相似度矩阵 [B*H*W, 1]
    sim_scores = torch.matmul(img_feat_flat_norm, text_embeds.t()) # [B*H*W, 1]
    sim_scores = torch.sigmoid(sim_scores) # [B*H*W, 1] -> [0, 1]

    # reshape 回 [B, H, W, 1] 并 permute to [B, 1, H, W]
    sim_map = sim_scores.view(B, H, W, 1).permute(0, 3, 1, 2) # [B, 1, H, W]

    # 将相似度图广播到图像特征上
    enhanced_feat = img_feat * sim_map
    return enhanced_feat

# --- 3. 生成 Prompt-DDM 密度图 (优化版) ---
# util/pgddc_density_utils.py (修正版)

def generate_prompt_ddm_density(
        image: torch.Tensor, # [C, H, W]
        prompts: List[str], # ["red bird ."]
        sample_boxes_xyxy: torch.Tensor, # [N, 4] (来自COUNTGD-BOX的检测框)
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        device: str = "cuda",
        sigma_scale: float = 1.0, # 高斯核缩放
) -> torch.Tensor: # [1, H, W]
    """
    核心修改：使用 CLIP 语义增强特征，然后基于增强特征生成密度图。
    """
    # 1. 图像预处理
    img_np = renorm(image.cpu()).permute(1, 2, 0).numpy()
    if np.min(img_np) < 0 or np.max(img_np) > 1:
        img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # 转RGB（CLIP要求）
    img_h, img_w = img_rgb.shape[:2]

    # ... (其他代码不变) ...

    # 5. 基于 COUNTGD-BOX 框和语义响应打点
    # 这部分保持不变，但使用语义响应图来加权
    density_map = torch.zeros((1, img_h, img_w), device=device)
    if sample_boxes_xyxy.size(0) > 0:
        # 将框中心坐标转换为整数索引
        centers = (sample_boxes_xyxy[:, [0, 2]] + sample_boxes_xyxy[:, [1, 3]]) / 2 # [N, 2] (x, y)
        # --- 修正：分别对 x 和 y 坐标进行裁剪 ---
        centers_x = centers[:, 0].long().clamp(0, img_w - 1)
        centers_y = centers[:, 1].long().clamp(0, img_h - 1)
        centers_int = torch.stack((centers_x, centers_y), dim=1) # [N, 2] (x, y)
        # --- 修正结束 ---

        # 获取对应的语义响应值 (假设 semantic_density 已计算)
        # semantic_values = semantic_density[0, centers_int[:, 1], centers_int[:, 0]] # [N]
        # 这里我们暂时不使用语义响应，直接打点
        semantic_values = torch.ones(centers_int.size(0), device=device) # [N]

        # 过滤：仅保留与提示相关的点 (阈值可调，这里简化)
        valid_mask = semantic_values > 0.5 # [N]
        valid_centers = centers_int[valid_mask] # [M, 2]
        valid_semantic_values = semantic_values[valid_mask] # [M]

        # 打点
        if valid_centers.size(0) > 0:
            # <--- 添加索引范围检查 (修正后理论上不会触发) --->
            y_coords = valid_centers[:, 1] # y is row index
            x_coords = valid_centers[:, 0] # x is col index
            if y_coords.max() >= img_h or x_coords.max() >= img_w or y_coords.min() < 0 or x_coords.min() < 0:
                print(f"Warning: Index out of bounds in density_map. y range: [{y_coords.min()}, {y_coords.max()}], x range: [{x_coords.min()}, {x_coords.max()}], image size: ({img_h}, {img_w})")
                print(f"  valid_centers: {valid_centers}")
                print(f"  sample_boxes_xyxy: {sample_boxes_xyxy}")
                # 裁剪坐标以避免越界 (理论上不需要，因为 clamp 已经处理)
                y_coords = torch.clamp(y_coords, 0, img_h - 1)
                x_coords = torch.clamp(x_coords, 0, img_w - 1)
            # <--- 检查结束 --->
            density_map[0, y_coords, x_coords] = valid_semantic_values

        # 6. 高斯平滑
        gauss_kernel = get_dynamic_gaussian_kernel(prompts[0], device=device, sigma_scale=sigma_scale)
        print(f"DEBUG: density_map shape: {density_map.shape}, gauss_kernel shape: {gauss_kernel.shape}") # 调试信息
        density_map = F_nn.conv2d(density_map, gauss_kernel, padding="same")
    # # 5. 基于 COUNTGD-BOX 框和语义响应打点
    # density_map = torch.zeros((1, img_h, img_w), device=device)
    # if sample_boxes_xyxy.size(0) > 0:
    #
    #     # ================== 🚨 终极修复2：彻底分离 X 和 Y 计算 ==================
    #     centers_x = ((sample_boxes_xyxy[:, 0] + sample_boxes_xyxy[:, 2]) / 2.0).long().clamp(0, img_w - 1)
    #     centers_y = ((sample_boxes_xyxy[:, 1] + sample_boxes_xyxy[:, 3]) / 2.0).long().clamp(0, img_h - 1)
    #     centers_int = torch.stack((centers_x, centers_y), dim=1)  # [N, 2] (x, y)
    #     # ====================================================================
    #
    #     semantic_values = torch.ones(centers_int.size(0), device=device)  # [N]
    #
    #     valid_mask = semantic_values > 0.5  # [N]
    #     valid_centers = centers_int[valid_mask]  # [M, 2]
    #     valid_semantic_values = semantic_values[valid_mask]  # [M]
    #
    #     if valid_centers.size(0) > 0:
    #         y_coords = valid_centers[:, 1]
    #         x_coords = valid_centers[:, 0]
    #         density_map[0, y_coords, x_coords] = valid_semantic_values
    #
    #     # 6. 高斯平滑
    #     gauss_kernel = get_dynamic_gaussian_kernel(prompts[0], device=device, sigma_scale=sigma_scale)
    #
    #     # ================== 🚨 终极修复3：升维满足 conv2d ==================
    #     density_map = density_map.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
    #     density_map = F_nn.conv2d(density_map, gauss_kernel, padding="same")
    #     density_map = density_map.squeeze(0)  # 恢复回 [1, H, W]
    #     # =================================================================

        # 7. 稀疏化 (可选，如果有效)
        # density_map = sparse_density_map(density_map)

    return density_map

# --- 4. 稀疏化函数 (可选) ---
def sparse_density_map(density_map: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    将密度图稀疏化，置零低于阈值的元素。
    Args:
        density_map: [B, 1, H, W] 输入密度图
        threshold: 稀疏化阈值
    Returns:
        sparse_map: [B, 1, H, W] 稀疏化后的密度图
    """
    sparse_map = density_map.clone()
    sparse_map[sparse_map < threshold] = 0.0
    return sparse_map

# --- 5. 初始化 CLIP (保持不变) ---
def init_clip(device: str = "cuda") -> Tuple[CLIPModel, CLIPProcessor]:
    """初始化CLIP模型（彻底移除accelerate依赖）"""
    LOCAL_CLIP_PATH = "/data1/wangjh/dataset/PG-DDC/pretrain/clip-vit-base-patch32"

    # 验证本地文件
    required_files = ["config.json", "pytorch_model.bin", "preprocessor_config.json", "tokenizer.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(LOCAL_CLIP_PATH, f))]
    if missing_files:
        raise FileNotFoundError(f"CLIP文件缺失：{missing_files}，路径：{LOCAL_CLIP_PATH}")

    # 强制离线
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    clip_model = CLIPModel.from_pretrained(
        pretrained_model_name_or_path=LOCAL_CLIP_PATH,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).to(device)

    clip_processor = CLIPProcessor.from_pretrained(
        LOCAL_CLIP_PATH,
        local_files_only=True
    )

    clip_model.eval()
    print(f"CLIP加载成功！设备：{next(clip_model.parameters()).device}")
    return clip_model, clip_processor

torch.set_grad_enabled(False)