"""Footprint mask head that fuses RoI appearance and geometry cues."""

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from ..mask_heads import FCNMaskHead


@HEADS.register_module()
class VisualFeatureFootprintMaskHead(FCNMaskHead):
    """Predict footprint masks from RoI features, roof logits and offsets.

    The three inputs are fused at the roof-mask resolution. RoI features keep
    the visual evidence outside the predicted roof, while the roof logits and
    roof-to-footprint offset provide explicit geometry cues.
    """

    requires_roi_features = True

    def __init__(
            self,
            roi_in_channels=256,
            visual_channels=64,
            roof_channels=1,
            num_convs=8,
            visual_norm_cfg=dict(type="GN", num_groups=8),
            **kwargs,
    ):
        offset_channels = 2
        fused_channels = visual_channels + roof_channels + offset_channels
        if "in_channels" in kwargs:
            raise ValueError(
                "in_channels is derived from visual_channels, roof_channels "
                "and the two offset-map channels; do not set it explicitly")
        super().__init__(
            in_channels=fused_channels, num_convs=num_convs, **kwargs)

        self.roi_in_channels = roi_in_channels
        self.visual_channels = visual_channels
        self.roof_channels = roof_channels
        self.offset_channels = offset_channels

        self.visual_projection = ConvModule(
            roi_in_channels,
            visual_channels,
            kernel_size=1,
            norm_cfg=visual_norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

    @auto_fp16()
    def forward(self, roi_features, roof_logits, offsets):
        if roi_features.size(0) != roof_logits.size(0) or offsets.size(
                0) != roof_logits.size(0):
            raise ValueError(
                "RoI features, roof logits and offsets must have the same batch size"
            )
        if roi_features.size(1) != self.roi_in_channels:
            raise ValueError(
                f"expected {self.roi_in_channels} RoI channels, got {roi_features.size(1)}"
            )
        if roof_logits.size(1) != self.roof_channels:
            raise ValueError(
                f"expected {self.roof_channels} roof channels, got {roof_logits.size(1)}"
            )
        if offsets.ndim != 2 or offsets.size(1) != 2:
            raise ValueError("offsets must have shape [N, 2]")

        output_size = roof_logits.shape[-2:]
        visual_features = self.visual_projection(roi_features)
        visual_features = F.interpolate(
            visual_features,
            size=output_size,
            mode="bilinear",
            align_corners=False)
        offset_features = offsets[:, :, None, None].expand(
            -1, -1, output_size[0], output_size[1])

        x = torch.cat([visual_features, roof_logits, offset_features], dim=1)
        return super().forward(x)

    @force_fp32(apply_to=("mask_pred", ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = super().loss(mask_pred, mask_targets, labels)
        return {"loss_footprint_mask_visual": loss["loss_mask"]}
