# util/pgddc_stage2_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import peak_local_max
from typing import Optional, Tuple, Dict, List
import cv2
import os
import sys # 需要导入 sys

class ArgsNamespace:
    def __init__(self):
        self.small = False  
        self.dropout = 0   
        self.alternate_corr = False 
        self.mixed_precision = False 
        if self.small:
            self.corr_levels = 4
            self.corr_radius = 3
        else:
            self.corr_levels = 4
            self.corr_radius = 4

    def __contains__(self, key):
        return hasattr(self, key)

def initialize_raft_lite(device="cuda"):
    raft_path = "/data1/wangjh/dataset/PG-DDC/RAFT" 
    core_path = os.path.join(raft_path, "core") 
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

    try:
        from raft import RAFT 
        args = ArgsNamespace()
        model = RAFT(args)
        checkpoint_path = os.path.join(raft_path, "models", "raft-kitti.pth") 
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'module' in state_dict:
            state_dict = state_dict['module']
        elif any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', ''
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("模型加载成功")
        return model
    except ImportError as e:
        print(f"RAFT-Lite 模块导入失败: {e}")
        print(" 请确保 RAFT 仓库已正确克隆，并且 core 目录在 Python 路径中。")
        # 返回一个模拟器用于调试
        class DummyRAFT:
            def __call__(self, image1, image2):
                h, w = image1.shape[-2], image1.shape[-1]
                return torch.zeros((image1.shape[0], 2, h, w), device=image1.device)
        return DummyRAFT()
    except FileNotFoundError:
        print(f"RAFT 权重文件未找到: {checkpoint_path}")
        # 返回一个模拟器用于调试
        class DummyRAFT:
            def __call__(self, image1, image2):
                h, w = image1.shape[-2], image1.shape[-1]
                return torch.zeros((image1.shape[0], 2, h, w), device=image1.device)
        return DummyRAFT()

def compute_optical_flow(raft_model, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        raft_model: RAFT-Lite 模型实例
        img1: [B, 3, H, W] 的第一帧图像张量
        img2: [B, 3, H, W] 的第二帧图像张量
    Returns:
        flow: [B, 2, H, W] 的光流场（x, y 方向）
    """
    with torch.no_grad():
        flow = raft_model(img1, img2)
    return flow

def warp_density_map(density_map: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Args:
        density_map: [B, 1, H, W] 的密度图
        flow: [B, 2, H, W] 的光流场
    Returns:
        warped_density: [B, 1, H, W] 形变后的密度图
    """
    B, C, H, W = density_map.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=0).float().to(density_map.device) # [2, H, W]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1) # [B, 2, H, W]

    warped_grid = grid + flow # [B, 2, H, W]

    warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
    warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0

    warped_grid = warped_grid.permute(0, 2, 3, 1) # [B, H, W, 2]

    warped_density = F.grid_sample(
        density_map,
        warped_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return warped_density

def calculate_temporal_residual(current_density: torch.Tensor,
                               prev_densities: list,
                               flows_to_current: list) -> torch.Tensor:
    """
    Args:
        current_density: [1, H, W] 当前帧密度图
        prev_densities: List[[1, H, W]] 历史帧密度图列表 (例如 [t-1, t-2])
        flows_to_current: List[[1, 2, H, W]] 从历史帧到当前帧的光流列表 (例如 [flow_t-1_to_t, flow_t-2_to_t])
    Returns:
        residual: [1, H, W] 加权残差图
    """
    residual = torch.zeros_like(current_density)
    if len(prev_densities) > 0 and len(flows_to_current) > 0:
        for i in range(min(len(prev_densities), len(flows_to_current))):
            prev_d = prev_densities[i]
            flow = flows_to_current[i]
            warped_prev_d = warp_density_map(prev_d.unsqueeze(0), flow).squeeze(0)
            residual += (current_density - warped_prev_d)

    # threshold = torch.quantile(torch.abs(residual), 0.1)
    # residual[torch.abs(residual) < threshold] = 0.0

    return residual

def detect_density_peaks(density_map: torch.Tensor, threshold: float = 0.01, min_distance: int = 1) -> torch.Tensor:
    """
    Args:
        density_map: [1, H, W] 密度图
        threshold: 峰值检测绝对阈值
        min_distance: 峰值间最小距离
    Returns:
        peaks_tensor: [K, 2] 峰值坐标 (y, x)
    """
    if density_map.dim() == 3 and density_map.shape[0] == 1:
        density_map = density_map.squeeze(0) # [H, W]
    elif density_map.dim() == 2:
        pass 
    else:
        raise ValueError(f"Expected density_map to be [1, H, W] or [H, W], got {density_map.shape}")

    density_np = density_map.cpu().numpy()
    peaks = peak_local_max(density_np, min_distance=min_distance, threshold_abs=threshold)

    if len(peaks) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=density_map.device)

    return torch.from_numpy(peaks).long().to(density_map.device)

def refine_boxes_with_density_peaks(
    current_boxes: torch.Tensor, # [N, 4] (xyxy)
    density_peaks: torch.Tensor, # [K, 2] (y, x)
    image_size: tuple,
    peak_threshold: float = 0.01,
    max_distance_pixels: int = 5
) -> torch.Tensor:
    """
    Args:
        current_boxes: [N, 4] 当前检测框 (xyxy format)
        density_peaks: [K, 2] 密度图峰值坐标 (y, x format)
        image_size: (H, W) 图像尺寸
        peak_threshold: 峰值阈值 (用于补全，虽然在函数内部未直接使用，但作为接口参数保留)
        max_distance_pixels: 修正距离阈值
    Returns:
        refined_boxes: [M, 4] 修正/补全后的框 (xyxy format)
    """
    H, W = image_size
    refined_boxes = []
    used_peaks = set()

    if current_boxes.size(0) > 0:
        boxes_centers = (current_boxes[:, [0, 2]] + current_boxes[:, [1, 3]]) / 2 # [N, 2] (x, y)

        if density_peaks.size(0) == 0:
            for i in range(current_boxes.size(0)):
                refined_boxes.append(current_boxes[i].cpu().numpy().tolist())
        else:
            for i, (cx, cy) in enumerate(boxes_centers):
                distances = torch.sqrt((density_peaks[:, 1] - cx)**2 + (density_peaks[:, 0] - cy)**2) # (y, x) -> (x, y)
                if distances.numel() > 0: 
                    min_dist_idx = torch.argmin(distances)
                    min_dist = distances[min_dist_idx]

                    if min_dist.item() <= max_distance_pixels:
                        new_cx, new_cy = density_peaks[min_dist_idx, 1], density_peaks[min_dist_idx, 0] # (x, y)
                        width = current_boxes[i, 2] - current_boxes[i, 0]
                        height = current_boxes[i, 3] - current_boxes[i, 1]
                        new_x1 = int(max(0, new_cx - width / 2))
                        new_y1 = int(max(0, new_cy - height / 2))
                        new_x2 = int(min(W, new_cx + width / 2))
                        new_y2 = int(min(H, new_cy + height / 2))
                        refined_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        used_peaks.add(min_dist_idx.item())
                    else:
                        refined_boxes.append(current_boxes[i].cpu().numpy().tolist())
                else:
                    refined_boxes.append(current_boxes[i].cpu().numpy().tolist())

    if refined_boxes:
        return torch.tensor(refined_boxes, dtype=torch.float32, device=current_boxes.device)
    else:
        return torch.empty((0, 4), dtype=torch.float32, device=current_boxes.device)

class PGDDCStage2Inference:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device
        self.raft_model = initialize_raft_lite(device=device)
        self.video_history: Dict[str, List[Dict[str, torch.Tensor]]] = {} # key: video_id, value: list of {'image': img, 'density': density, 'refined_boxes': boxes}

    def process_frame(self, image: torch.Tensor, prompt_text: str, current_boxes: torch.Tensor, frame_id: int, video_name: str):
        """
        Args:
            image: [3, H, W] 当前帧图像张量
            prompt_text: 文本提示
            current_boxes: [N, 4] 当前帧的检测框 (xyxy format)
            frame_id: 帧索引
            video_name: 视频ID
        Returns:
            refined_boxes: [M, 4] 修正后的检测框 (xyxy format)
            final_density: [1, H, W] 最终密度图
        """
        print("PGDDCStage2Inference.process_frame called. Logic should be implemented in engine_fscd147.py using provided utils.")
        return current_boxes, torch.zeros((1, image.shape[1], image.shape[2]), device=image.device) # Placeholder

    def reset_video(self, video_name: str):
        if video_name in self.video_history:
            del self.video_history[video_name]

    def process_single_frame_for_engine(self, image: torch.Tensor, prompt_text: str, current_boxes: torch.Tensor,
                                        current_density_map: torch.Tensor, video_id: str, sam_predictor=None):
        print(f"Applying PG-DDC Stage 2 (Density-Detection Loop) on {video_id} frame...")

        density_peaks = detect_density_peaks(current_density_map, threshold=0.01, min_distance=1)

        refined_boxes = refine_boxes_with_density_peaks(
            current_boxes=current_boxes,
            density_peaks=density_peaks,
            image_size=(image.shape[1], image.shape[2]),
            peak_threshold=0.01,
            max_distance_pixels=5
        )

        sam_masks = None
        if sam_predictor is not None:
            img = image.cpu().float()
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = img * std + mean
            img_rgb = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            img_rgb = (img_rgb * 255).astype(np.uint8)

            try:
                from util.pgddc_sam_utils import refine_foreground_with_ada_sam
                sam_masks, refined_boxes = refine_foreground_with_ada_sam(
                    sam_predictor=sam_predictor,
                    image=img_rgb,
                    density_map=current_density_map,
                    boxes=refined_boxes,
                    device=refined_boxes.device
                )
            except Exception as e:
                print(f"⚠️ Ada-SAM 调用失败 (将仅返回框): {e}")

        print(f"Loop Refined Count: {refined_boxes.size(0)} (from {current_boxes.size(0)} input boxes)")

        return refined_boxes, sam_masks

    def process_video_sequence(self, video_frames: List[torch.Tensor], video_boxes: List[torch.Tensor], video_prompt: str, video_name: str):
        """
        Args:
            video_frames: List of [3, H, W] 图像张量
            video_boxes: List of [N, 4] 检测框张量 (对应每帧)
            video_prompt: 视频的文本提示
            video_name: 视频ID
        Returns:
            List of [M, 4] 修正后的检测框张量 (对应每帧)
            List of [1, H, W] 最终密度图张量 (对应每帧)
        """
        print(f"Processing video sequence: {video_name}")
        self.reset_video(video_name) # 清空历史缓存

        refined_boxes_list = []
        final_density_list = []

        for frame_idx, (frame, boxes) in enumerate(zip(video_frames, video_boxes)):
            if video_name not in self.video_history:
                self.video_history[video_name] = []

            prev_frames = self.video_history[video_name]
            final_density = initial_density_map # Placeholder:

            if len(prev_frames) >= 1: 
                prev_densities = [f['density'] for f in prev_frames[-2:]] 
                prev_images = [f['image'] for f in prev_frames[-2:]]
                flows_to_current = []
                for prev_img in prev_images:
                    flow = compute_optical_flow(self.raft_model, prev_img.unsqueeze(0), frame.unsqueeze(0))
                    flows_to_current.append(flow.squeeze(0))

                temporal_residual = calculate_temporal_residual(initial_density_map, prev_densities, flows_to_current)
                final_density = initial_density_map + temporal_residual

            refined_boxes = refine_boxes_with_density_peaks(
                current_boxes=boxes,
                density_peaks=detect_density_peaks(final_density, threshold=0.01, min_distance=1),
                image_size=(frame.shape[1], frame.shape[2]),
                peak_threshold=0.01,
                max_distance_pixels=5
            )

            self.video_history[video_name].append({
                'image': frame,
                'density': final_density,
                'refined_boxes': refined_boxes
            })
            if len(self.video_history[video_name]) > 3:
                self.video_history[video_name].pop(0)

            refined_boxes_list.append(refined_boxes)
            final_density_list.append(final_density)

        print(f"Completed processing video sequence: {video_name}")
        return refined_boxes_list, final_density_list
