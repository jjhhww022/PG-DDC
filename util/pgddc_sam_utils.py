# util/pgddc_sam_utils.py

import numpy as np
from typing import Tuple
import torch
from util.pgddc_density_utils import get_dynamic_gaussian_kernel # 如果需要
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
from skimage.feature import peak_local_max

def calculate_density_ratio(density_map):
    """计算密度图的区域占比 (优化版)"""
    total_pixels = density_map.numel()
    # 使用均值作为阈值
    mean_val = density_map.mean()
    non_zero_pixels = (density_map > mean_val).sum().item()
    return non_zero_pixels / total_pixels

def dynamic_brp(density_ratio, min_ratio=0.1, max_ratio=0.8):
    """动态框保留比例 (优化版)"""
    # 避免极端值
    density_ratio = np.clip(density_ratio, 0.0, 1.0)
    keep_ratio = max(min_ratio, max_ratio - density_ratio * (max_ratio - min_ratio))
    return keep_ratio

def filter_foreground_by_density_peak(foreground_tokens, density_map, feat_stride=16):
    """
    密度峰值过滤 (优化版)
    Args:
        foreground_tokens: SAM2输出的前景tokens
        density_map: [1, H, W] 密度图
        feat_stride: SAM2特征步长
    Returns:
        filtered_tokens: 过滤后的前景tokens
    """
    # 假设 foreground_tokens 是一个列表或张量，包含前景点的坐标
    # 这里需要根据 SAM2 的实际输出格式进行调整
    # 例如，如果 tokens 是 [N, 2] (x, y) 坐标
    if isinstance(foreground_tokens, torch.Tensor):
        tokens = foreground_tokens.cpu().numpy()
    else:
        tokens = np.array(foreground_tokens)

    if tokens.size == 0:
        return foreground_tokens

    # 将 token 坐标映射到密度图空间 (假设 token 坐标是原图坐标)
    # 如果 token 是特征图坐标，则需要乘以 feat_stride
    # mapped_tokens = tokens * feat_stride # 如果是特征图坐标

    # 获取对应密度图的值
    H, W = density_map.shape[-2], density_map.shape[-1]
    x_coords = np.clip(tokens[:, 0], 0, W - 1).astype(int)
    y_coords = np.clip(tokens[:, 1], 0, H - 1).astype(int)
    density_values = density_map[0, y_coords, x_coords].cpu().numpy()

    # 设定阈值过滤
    threshold = np.percentile(density_values, 50) # 例如，只保留密度值高于中位数的点
    filtered_mask = density_values >= threshold
    filtered_tokens = tokens[filtered_mask]

    # 返回过滤后的 tokens (转换回原始格式)
    if isinstance(foreground_tokens, torch.Tensor):
        return torch.from_numpy(filtered_tokens).to(foreground_tokens.device)
    else:
        return filtered_tokens.tolist() # 或者保持 numpy 格式

def refine_foreground_with_ada_sam(
        sam_predictor: SAM2ImagePredictor,
        image: np.ndarray,
        density_map: torch.Tensor,
        boxes: torch.Tensor,
        device: str = "cuda",
        peak_threshold: float = 0.01,  # 极低阈值，最大化峰值召回
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    适配SAM2的前景掩码优化函数 (优化版)
    """
    # 1. 密度图预处理
    density_map_np = density_map.squeeze(0).cpu().numpy()
    img_h, img_w = image.shape[:2]

    if density_map_np.shape != (img_h, img_w):
        density_map_np = cv2.resize(density_map_np, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    density_min = density_map_np.min()
    density_max = density_map_np.max()
    if density_max - density_min < 1e-8:
        density_map_np = np.zeros_like(density_map_np)
    else:
        density_map_np = (density_map_np - density_min) / (density_max - density_min)

    density_map_np = cv2.GaussianBlur(density_map_np, (5, 5), 1.5)
    density_map_np = cv2.normalize(density_map_np, None, 0, 1, cv2.NORM_MINMAX)

    # 2. 峰值检测 (优化)
    peak_coords = peak_local_max(
        density_map_np,
        min_distance=1,
        threshold_abs=peak_threshold,
        threshold_rel=0.001,
        exclude_border=False,
        num_peaks=np.inf
    )
    peak_coords = peak_coords[
        (peak_coords[:, 0] >= 0) &
        (peak_coords[:, 0] < img_h) &
        (peak_coords[:, 1] >= 0) &
        (peak_coords[:, 1] < img_w)
        ]

    # 3. 峰值数量限制
    max_peaks = 500
    if len(peak_coords) > max_peaks:
        peak_densities = density_map_np[peak_coords[:, 0], peak_coords[:, 1]]
        sorted_idx = np.argsort(peak_densities)[-max_peaks:]
        peak_coords = peak_coords[sorted_idx]
    print(f"过滤后有效峰值数：{len(peak_coords)}")

    # 4. 无峰值时的兜底策略
    if len(peak_coords) == 0:
        print("未检测到有效峰值，基于检测框生成兜底掩码")
        peak_mask = np.zeros((img_h, img_w), dtype=np.bool_)
        if len(boxes) > 0:
            boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            for box in boxes_np:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                peak_mask[y1:y2, x1:x2] = True
        return peak_mask, boxes

    # 5. 生成峰值掩码
    peak_mask = np.zeros((img_h, img_w), dtype=np.bool_)
    peak_mask[peak_coords[:, 0], peak_coords[:, 1]] = True

    # 6. SAM2 推理 (优化)
    sam_predictor.set_image(image)
    peak_points_xy = peak_coords[:, [1, 0]]  # peak_coords是[y,x] → 转为[x,y]

    # 合并峰值点和检测框角点
    box_points = []
    if len(boxes) > 0:
        boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
        for box in boxes_np:
            x1, y1, x2, y2 = map(int, box)
            box_points.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    box_points = np.array(box_points)
    box_points = box_points[
        (box_points[:, 1] >= 0) & (box_points[:, 1] < img_h) &
        (box_points[:, 0] >= 0) & (box_points[:, 0] < img_w)
        ]
    total_points = np.vstack([peak_points_xy, box_points[:20]])[:max_peaks + 20] # 限制总数

    # 7. 点提示格式调整
    total_points_tensor = torch.from_numpy(total_points).to(device, dtype=torch.float32)
    total_points_tensor[:, 0] /= img_w
    total_points_tensor[:, 1] /= img_h
    point_coords = total_points_tensor.unsqueeze(0)
    point_labels = torch.ones((1, len(total_points)), device=device, dtype=torch.int32)

    # 8. SAM2 Predict (异常处理)
    try:
        masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
            return_logits=True
        )
    except Exception as e:
        print(f"SAM2 predict失败：{e}，使用峰值掩码兜底")
        masks = torch.tensor(peak_mask).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        scores = torch.ones(1, device=device, dtype=torch.float32)
        logits = torch.zeros_like(masks)

    # 9. 掩码后处理
    if isinstance(masks, torch.Tensor):
        sam_mask = masks.squeeze(0).cpu().numpy() > 0.5
    else:
        sam_mask = masks.squeeze(0) > 0.5
    refined_mask = sam_mask | peak_mask
    refined_mask = refined_mask.astype(np.bool_)

    # 10. 筛选检测框 (优化)
    if len(boxes) > 0:
        score_threshold = 0.25
        scores_np = scores.cpu().numpy().flatten() if isinstance(scores, torch.Tensor) else np.array(scores, dtype=np.float32).flatten()

        if len(scores_np) == 0:
            scores_np = np.array([0.5], dtype=np.float32)

        box_num = len(boxes)
        score_num = len(scores_np)

        if score_num > box_num:
            scores_np = scores_np[:box_num]
        elif score_num < box_num:
            pad_scores = np.ones(box_num - score_num, dtype=np.float32) * 0.5
            scores_np = np.concatenate([scores_np, pad_scores])

        selected_boxes = boxes[scores_np > score_threshold]
    else:
        selected_boxes = boxes

    return refined_mask, selected_boxes
