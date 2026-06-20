import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GroundingDINO.groundingdino import GroundingDINO  # 复用CountVid的COUNTGD-BOX
from util.density_utils import enhance_feature_with_prompt, get_dynamic_gaussian_kernel, sparse_density_map
from util.sam_utils import calculate_scene_density_ratio, dynamic_brp, filter_foreground_by_density_peak


class PromptDDM(nn.Module):
    def __init__(self, emac_model, clip_model, clip_processor, device="cuda", feat_dim=256):
        super().__init__()
        self.emac = emac_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

        self.density_conv = nn.Conv2d(feat_dim, 1, kernel_size=1)
        nn.init.normal_(self.density_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.density_conv.bias, 0.0)
        self.density_conv = self.density_conv.to(device)

    def forward(self, img: torch.Tensor, prompt_text: str, countgd_boxes: torch.Tensor):
        img_feat = self.emac.extract_feature(img)  # [B, C, H, W]

        enhanced_feat = enhance_feature_with_prompt(
            img_feat, prompt_text, self.clip_model, self.clip_processor, self.device
        )

        semantic_density = torch.sigmoid(self.density_conv(enhanced_feat))  # [B,1,H,W]

        final_density = torch.zeros_like(semantic_density)
        B, _, H, W = final_density.shape
        for b in range(B):
            for box in countgd_boxes[b]:
                x1, y1, x2, y2 = box.int()
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cx < W and 0 <= cy < H:
                    semantic_value = semantic_density[b, 0, cy, cx].item()
                    weight = semantic_value if semantic_value > 0.5 else 0.0  # 不匹配则丢弃！
                    if weight > 0:
                        final_density[b, 0, cy, cx] += weight  # 或直接 = 1.0

        gauss_kernel = get_dynamic_gaussian_kernel(prompt_text, device=self.device)
        final_density = F.conv2d(final_density, gauss_kernel, padding="same")

        return final_density


class AdaSAM(nn.Module):
    def __init__(self, sam_model, feat_stride=16, device="cuda"):
        super().__init__()
        self.sam = sam_model
        self.feat_stride = feat_stride
        self.device = device

    def forward(self, img: torch.Tensor, density_map: torch.Tensor):
        # 1. 计算场景密度R
        R = calculate_scene_density_ratio(density_map)

        # 2. 动态BRP（添加安全检查）
        R = torch.clamp(R, 0.0, 1.0)
        brp = dynamic_brp(R)
        brp = torch.clamp(brp, 0.05, 0.4)  # 限制在合理范围

        # 3. 获取SAM原始输出
        sam_output = self.sam(img, background_keep_prob=brp)

        # 4. 密度峰值过滤（但保留原始结构）
        filtered_tokens = filter_foreground_by_density_peak(
            sam_output["foreground_tokens"],
            density_map,
            self.feat_stride
        )

        # 5. 返回完整结果，但更新前景tokens
        sam_output["foreground_tokens"] = filtered_tokens

        # 6. 添加诊断信息
        sam_output["sam_params"] = {
            "brp_used": brp.item(),
            "density_ratio": R.item(),
            "original_tokens": len(sam_output["foreground_tokens"]),
            "filtered_tokens": len(filtered_tokens)
        }

        return sam_output