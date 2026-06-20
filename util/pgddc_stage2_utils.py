# util/pgddc_stage2_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from skimage.feature import peak_local_max
from typing import Optional, Tuple, Dict, List
import cv2
import os
import sys # 需要导入 sys

# --- 自定义 ArgsNamespace 类 ---
class ArgsNamespace:
    def __init__(self):
        self.small = False  # KITTI 模型通常不是 small 版本
        self.dropout = 0    # 或者 0.25，取决于训练时的设置，通常 KITTI 模型会使用 dropout
        self.alternate_corr = False # 通常默认为 False
        self.mixed_precision = False # 如果需要 FP16 推理，设为 True
        # 根据 raft.py 中的 __init__ 逻辑，设置 corr_levels 和 corr_radius
        if self.small:
            self.corr_levels = 4
            self.corr_radius = 3
        else:
            self.corr_levels = 4
            self.corr_radius = 4

    def __contains__(self, key):
        # 实现 'in' 操作符
        return hasattr(self, key)
# --- 类定义结束 ---

# --- 1. 光流与密度图形变 ---
def initialize_raft_lite(device="cuda"):
    """
    初始化轻量化的 RAFT-Lite 模型。
    假设 RAFT 仓库位于 /data1/wangjh/dataset/PG-DDC/RAFT
    """
    raft_path = "/data1/wangjh/dataset/PG-DDC/RAFT" # RAFT 仓库根目录
    core_path = os.path.join(raft_path, "core") # RAFT 核心代码目录

    # 将 RAFT 的 core 目录添加到 Python 路径
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

    try:
        # 尝试导入 RAFT 模型
        from raft import RAFT # 从 core 目录导入
        # 创建 args 对象
        args = ArgsNamespace()
        # 加载模型
        model = RAFT(args)
        # 加载模型权重
        checkpoint_path = os.path.join(raft_path, "models", "raft-kitti.pth") # 根据你的权重文件名调整
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # --- 处理 'module.' 前缀 ---
        if 'module' in state_dict:
            state_dict = state_dict['module']
        elif any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') # 移除 'module.' 前缀
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        # --- 处理结束 ---
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("RAFT-Lite 模型加载成功")
        return model
    except ImportError as e:
        print(f"RAFT-Lite 模块导入失败: {e}")
        print(" 请确保 RAFT 仓库已正确克隆，并且 core 目录在 Python 路径中。")
        # 返回一个模拟器用于调试
        class DummyRAFT:
            def __call__(self, image1, image2):
                # 返回零光流作为占位符
                h, w = image1.shape[-2], image1.shape[-1]
                return torch.zeros((image1.shape[0], 2, h, w), device=image1.device)
        return DummyRAFT()
    except FileNotFoundError:
        print(f"RAFT 权重文件未找到: {checkpoint_path}")
        # 返回一个模拟器用于调试
        class DummyRAFT:
            def __call__(self, image1, image2):
                # 返回零光流作为占位符
                h, w = image1.shape[-2], image1.shape[-1]
                return torch.zeros((image1.shape[0], 2, h, w), device=image1.device)
        return DummyRAFT()

def compute_optical_flow(raft_model, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    使用 RAFT-Lite 计算两帧之间的光流。
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
    根据光流场对密度图进行形变（warping）。
    Args:
        density_map: [B, 1, H, W] 的密度图
        flow: [B, 2, H, W] 的光流场
    Returns:
        warped_density: [B, 1, H, W] 形变后的密度图
    """
    B, C, H, W = density_map.shape
    # 创建网格坐标
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=0).float().to(density_map.device) # [2, H, W]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1) # [B, 2, H, W]

    # 应用光流
    warped_grid = grid + flow # [B, 2, H, W]

    # 归一化到 [-1, 1] 范围，以适应 grid_sample
    warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
    warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0

    # 调整维度顺序
    warped_grid = warped_grid.permute(0, 2, 3, 1) # [B, H, W, 2]

    # 使用 grid_sample 进行形变
    warped_density = F.grid_sample(
        density_map,
        warped_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return warped_density

# --- 2. Lite-TCF (轻量化长时序融合) ---
def calculate_temporal_residual(current_density: torch.Tensor,
                               prev_densities: list,
                               flows_to_current: list) -> torch.Tensor:
    """
    计算当前帧与历史帧的时序残差。
    Args:
        current_density: [1, H, W] 当前帧密度图
        prev_densities: List[[1, H, W]] 历史帧密度图列表 (例如 [t-1, t-2])
        flows_to_current: List[[1, 2, H, W]] 从历史帧到当前帧的光流列表 (例如 [flow_t-1_to_t, flow_t-2_to_t])
    Returns:
        residual: [1, H, W] 加权残差图
    """
    # 简单实现：融合 t-1 和 t-2 帧
    residual = torch.zeros_like(current_density)
    if len(prev_densities) > 0 and len(flows_to_current) > 0:
        for i in range(min(len(prev_densities), len(flows_to_current))):
            prev_d = prev_densities[i]
            flow = flows_to_current[i]
            warped_prev_d = warp_density_map(prev_d.unsqueeze(0), flow).squeeze(0)
            # 残差 = 当前 - 形变历史
            residual += (current_density - warped_prev_d)

    # 可选：残差稀疏化
    # threshold = torch.quantile(torch.abs(residual), 0.1)
    # residual[torch.abs(residual) < threshold] = 0.0

    return residual

# --- 3. 密度峰值检测 (优化版) ---
def detect_density_peaks(density_map: torch.Tensor, threshold: float = 0.01, min_distance: int = 1) -> torch.Tensor:
    """
    从密度图中高效检测峰值点。
    Args:
        density_map: [1, H, W] 密度图
        threshold: 峰值检测绝对阈值
        min_distance: 峰值间最小距离
    Returns:
        peaks_tensor: [K, 2] 峰值坐标 (y, x)
    """
    # 确保输入是 [1, H, W] 格式
    if density_map.dim() == 3 and density_map.shape[0] == 1:
        density_map = density_map.squeeze(0) # [H, W]
    elif density_map.dim() == 2:
        pass # 已经是 [H, W]
    else:
        raise ValueError(f"Expected density_map to be [1, H, W] or [H, W], got {density_map.shape}")

    density_np = density_map.cpu().numpy()
    peaks = peak_local_max(density_np, min_distance=min_distance, threshold_abs=threshold)

    if len(peaks) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=density_map.device)

    return torch.from_numpy(peaks).long().to(density_map.device)

# --- 4. 目标框修正与补全 (优化版) ---
def refine_boxes_with_density_peaks(
    current_boxes: torch.Tensor, # [N, 4] (xyxy)
    density_peaks: torch.Tensor, # [K, 2] (y, x)
    image_size: tuple,
    peak_threshold: float = 0.01,
    max_distance_pixels: int = 5
) -> torch.Tensor:
    """
    使用密度峰值修正现有框或补全漏检框。
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
        # 修正现有框
        boxes_centers = (current_boxes[:, [0, 2]] + current_boxes[:, [1, 3]]) / 2 # [N, 2] (x, y)

        # --- 修改：检查 density_peaks 是否为空 ---
        if density_peaks.size(0) == 0:
            # 如果没有峰值，则直接保留所有原始框
            for i in range(current_boxes.size(0)):
                refined_boxes.append(current_boxes[i].cpu().numpy().tolist())
        else:
            # --- 修改结束 ---
            for i, (cx, cy) in enumerate(boxes_centers):
                # 计算当前框中心到所有峰值的距离
                distances = torch.sqrt((density_peaks[:, 1] - cx)**2 + (density_peaks[:, 0] - cy)**2) # (y, x) -> (x, y)
                # --- 修改：添加对 distances 张量是否为空的检查 ---
                if distances.numel() > 0: # 确保 distances 不为空
                    min_dist_idx = torch.argmin(distances)
                    min_dist = distances[min_dist_idx]

                    if min_dist.item() <= max_distance_pixels:
                        # 修正框中心
                        new_cx, new_cy = density_peaks[min_dist_idx, 1], density_peaks[min_dist_idx, 0] # (x, y)
                        width = current_boxes[i, 2] - current_boxes[i, 0]
                        height = current_boxes[i, 3] - current_boxes[i, 1]
                        # --- 修改：确保 max/min 的结果是标量，然后转换为 int ---
                        new_x1 = int(max(0, new_cx - width / 2))
                        new_y1 = int(max(0, new_cy - height / 2))
                        new_x2 = int(min(W, new_cx + width / 2))
                        new_y2 = int(min(H, new_cy + height / 2))
                        # --- 修改结束 ---
                        refined_boxes.append([new_x1, new_y1, new_x2, new_y2]) # <--- 添加时不再调用 .item()
                        used_peaks.add(min_dist_idx.item())
                    else:
                        # 保留原框
                        refined_boxes.append(current_boxes[i].cpu().numpy().tolist())
                else:
                    # 如果 distances 为空（理论上不应该，因为 density_peaks 不为空），保留原框
                    refined_boxes.append(current_boxes[i].cpu().numpy().tolist())
        # --- 修改结束 ---

    # # 补全漏检目标
    # # --- 修改：同样检查 density_peaks 是否为空 ---
    # if density_peaks.size(0) > 0: # 只有在有峰值时才尝试补全
    #     for i, (y, x) in enumerate(density_peaks):
    #         if i not in used_peaks:
    #             size = 10  # 小尺寸框
    #             # --- 修改：确保 max/min 的结果是标量，然后转换为 int ---
    #             x1, y1 = int(max(0, x - size // 2)), int(max(0, y - size // 2))
    #             x2, y2 = int(min(W, x + size // 2)), int(min(H, y + size // 2))
    #             # --- 修改结束 ---
    #             refined_boxes.append([x1, y1, x2, y2]) # <--- 添加时不再调用 .item()
    # # --- 修改结束 ---

    if refined_boxes:
        return torch.tensor(refined_boxes, dtype=torch.float32, device=current_boxes.device)
    else:
        return torch.empty((0, 4), dtype=torch.float32, device=current_boxes.device)

# --- 5. PG-DDC Stage 2 推理器封装类 (优化版) ---
class PGDDCStage2Inference:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device

        # 初始化 CLIP 和 SAM2 (复用第一阶段) - ❌ 这些应该在 engine_fscd147.py 中处理
        # from util.pgddc_density_utils import init_clip
        # from util.pgddc_sam_utils import get_sam # <-- 删除此行
        # self.clip_model, self.clip_processor = init_clip(device=device)
        # self.sam_predictor = get_sam(sam_checkpoint=args.sam_model_path, model_cfg="sam2_hiera_l.yaml", device=device) # <-- 删除此行

        # 初始化 RAFT-Lite
        self.raft_model = initialize_raft_lite(device=device)

        # 缓存每个视频的历史帧信息
        self.video_history: Dict[str, List[Dict[str, torch.Tensor]]] = {} # key: video_id, value: list of {'image': img, 'density': density, 'refined_boxes': boxes}

    def process_frame(self, image: torch.Tensor, prompt_text: str, current_boxes: torch.Tensor, frame_id: int, video_name: str):
        """
        处理单帧图像，应用 Lite-TCF 和 Density-Detection Loop。
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
        # 1. 生成 Prompt-DDM 密度图 (基于原始框或 CLIP 过滤后的框)
        # 这部分逻辑需要在 engine_fscd147.py 的 get_count_errs 中完成，然后传入 density_map
        # initial_density_map = generate_prompt_ddm_density(...) # 在 engine 中调用
        # 为了演示，我们假设 initial_density_map 已经存在或作为参数传入
        # print("Warning: process_frame called, but density map generation is complex in single-image eval flow.")
        # ... (需要 density_map 才能继续) ...
        # 这个方法在当前框架下难以完整实现，因为 density_map 获取困难。

        # --- 实际应用中，你需要将 Lite-TCF 和 Loop 逻辑集成到 engine_fscd147.py 的 get_count_errs 中 ---
        # 该方法可以保留，但主要逻辑应在 engine 中实现，以访问所有中间结果和模型输出。
        # 这个类的主要作用是提供一个组织良好的结构和工具函数。
        # 例如，engine 中可以这样使用：
        # if video_name not in self.video_history:
        #     self.video_history[video_name] = []
        # prev_frames = self.video_history[video_name]
        # if len(prev_frames) >= 1:
        #     prev_density = prev_frames[-1]['density']
        #     flow = compute_optical_flow(self.raft_model, prev_frames[-1]['image'].unsqueeze(0), image.unsqueeze(0))
        #     warped_prev_density = warp_density_map(prev_density.unsqueeze(0), flow).squeeze(0)
        #     residual = current_density - warped_prev_density
        #     final_density = current_density + residual
        # else:
        #     final_density = current_density
        #
        # refined_boxes = refine_boxes_with_density_peaks(current_boxes, detect_density_peaks(final_density), (image.shape[1], image.shape[2]))
        #
        # # 更新缓存
        # self.video_history[video_name].append({'image': image, 'density': final_density, 'refined_boxes': refined_boxes})
        # if len(self.video_history[video_name]) > 3: # 保留最近3帧
        #     self.video_history[video_name].pop(0)
        # return refined_boxes, final_density
        # ---
        # 因此，这个方法可以作为一个高级接口，但内部需要接收中间结果。
        # 为了保持接口一致性，我们保留它，但内部逻辑需要在 engine 中实现。
        # 结论：这个类的主要价值在于提供 detect_density_peaks, refine_boxes_with_density_peaks, calculate_temporal_residual 等工具函数。
        # 完整的视频处理逻辑应在 engine 中，利用这些工具函数。
        # 这个方法可以作为一个占位符，或者接收所有必要的中间结果作为参数。
        print("PGDDCStage2Inference.process_frame called. Logic should be implemented in engine_fscd147.py using provided utils.")
        return current_boxes, torch.zeros((1, image.shape[1], image.shape[2]), device=image.device) # Placeholder

    def reset_video(self, video_name: str):
        """重置指定视频的历史帧缓存"""
        if video_name in self.video_history:
            del self.video_history[video_name]

    # --- 新增：用于 engine_fscd147.py 的单帧处理方法 (仅应用 Loop, 需要外部提供密度图) ---
    def process_single_frame_for_engine(self, image: torch.Tensor, prompt_text: str, current_boxes: torch.Tensor,
                                        current_density_map: torch.Tensor, video_id: str, sam_predictor=None):
        """
        为 engine_fscd147.py 设计的单帧处理接口。
        🚨 致命关键：看上面这一行，必须有 sam_predictor=None 🚨
        """
        print(f"Applying PG-DDC Stage 2 (Density-Detection Loop) on {video_id} frame...")

        # 1. 从 Prompt-DDM 密度图检测峰值
        density_peaks = detect_density_peaks(current_density_map, threshold=0.01, min_distance=1)

        # 2. 使用峰值初步修正/补全检测框
        refined_boxes = refine_boxes_with_density_peaks(
            current_boxes=current_boxes,
            density_peaks=density_peaks,
            image_size=(image.shape[1], image.shape[2]),
            peak_threshold=0.01,
            max_distance_pixels=5
        )

        sam_masks = None
        # 3. 🚨 核心：如果传入了 SAM 模型，就调用它生成高清掩码！
        if sam_predictor is not None:
            # 将 Tensor 反归一化并转为 RGB 的 numpy 数组，供 SAM 使用
            img = image.cpu().float()
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = img * std + mean
            img_rgb = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            img_rgb = (img_rgb * 255).astype(np.uint8)

            try:
                # 导入我们写好的带截胡画图功能的 SAM 工具函数
                from util.pgddc_sam_utils import refine_foreground_with_ada_sam
                # 真正调用 SAM 生成掩码，并在里面自动保存图片
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

        # 🚨 务必使用逗号，同时返回 框(boxes) 和 掩码(masks)
        return refined_boxes, sam_masks

    # --- 新增：用于视频处理的完整流程方法 (需要在 engine 外部调用) ---
    def process_video_sequence(self, video_frames: List[torch.Tensor], video_boxes: List[torch.Tensor], video_prompt: str, video_name: str):
        """
        处理整个视频序列，应用 Lite-TCF 和 Density-Detection Loop。
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
            # --- 1. 获取当前帧的密度图 (需要外部提供或在此处生成) ---
            # 这里需要调用 generate_prompt_ddm_density
            # initial_density_map = generate_prompt_ddm_density(...)
            # 为了演示，我们假设 initial_density_map 已计算好
            # initial_density_map = torch.rand(1, frame.shape[1], frame.shape[2], device=frame.device) # Placeholder
            # --- 1. END ---

            # --- 2. Lite-TCF: 计算时序残差并融合 ---
            if video_name not in self.video_history:
                self.video_history[video_name] = []

            prev_frames = self.video_history[video_name]
            final_density = initial_density_map # Placeholder: 实际应为 initial_density_map

            if len(prev_frames) >= 1: # 至少需要一帧历史
                # 准备历史帧和光流
                prev_densities = [f['density'] for f in prev_frames[-2:]] # 取最近2帧
                prev_images = [f['image'] for f in prev_frames[-2:]]
                flows_to_current = []
                for prev_img in prev_images:
                    flow = compute_optical_flow(self.raft_model, prev_img.unsqueeze(0), frame.unsqueeze(0))
                    flows_to_current.append(flow.squeeze(0))

                # 计算残差
                temporal_residual = calculate_temporal_residual(initial_density_map, prev_densities, flows_to_current)
                # 融合残差
                final_density = initial_density_map + temporal_residual

            # --- 3. Density-Detection Loop: 修正检测框 (基于融合后的密度图) ---
            refined_boxes = refine_boxes_with_density_peaks(
                current_boxes=boxes,
                density_peaks=detect_density_peaks(final_density, threshold=0.01, min_distance=1),
                image_size=(frame.shape[1], frame.shape[2]),
                peak_threshold=0.01,
                max_distance_pixels=5
            )

            # --- 4. 更新缓存 ---
            self.video_history[video_name].append({
                'image': frame,
                'density': final_density,
                'refined_boxes': refined_boxes
            })
            if len(self.video_history[video_name]) > 3: # 保留最近3帧
                self.video_history[video_name].pop(0)

            refined_boxes_list.append(refined_boxes)
            final_density_list.append(final_density)

        print(f"Completed processing video sequence: {video_name}")
        return refined_boxes_list, final_density_list
