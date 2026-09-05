"""Geometry-guided footprint mask refinement head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from ..mask_heads import FCNMaskHead


@HEADS.register_module()
class GeometryGuidedFootprintMaskHead(FCNMaskHead):
    """Refine an offset-shifted roof prior with RoI visual features.

    ``offsets`` keep the same encoded representation used by the original
    footprint head. ``decoded_offsets`` are image-space footprint-to-roof
    vectors. They are divided by the RoI size to translate the roof
    probability map into footprint coordinates with ``grid_sample``.
    """

    requires_roi_features = True
    requires_geometry_inputs = True

    def __init__(
        self,
        roi_in_channels=256,
        visual_channels=64,
        roof_channels=1,
        num_convs=8,
        visual_norm_cfg=dict(type="GN", num_groups=8),
        prior_eps=1e-4,
        residual_refinement=True,
        **kwargs,
    ):
        offset_channels = 2
        prior_channels = roof_channels
        fused_channels = (
            visual_channels + roof_channels + offset_channels + prior_channels)
        if "in_channels" in kwargs:
            raise ValueError(
                "in_channels is derived from visual, roof, offset-map and "
                "shifted-prior channels; do not set it explicitly")
        super().__init__(
            in_channels=fused_channels, num_convs=num_convs, **kwargs)

        self.roi_in_channels = roi_in_channels
        self.visual_channels = visual_channels
        self.roof_channels = roof_channels
        self.offset_channels = offset_channels
        self.prior_channels = prior_channels
        self.prior_eps = prior_eps
        self.residual_refinement = residual_refinement

        self.visual_projection = ConvModule(
            roi_in_channels,
            visual_channels,
            kernel_size=1,
            norm_cfg=visual_norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

    def init_weights(self):
        super().init_weights()
        if self.residual_refinement:
            nn.init.constant_(self.conv_logits.weight, 0)
            nn.init.constant_(self.conv_logits.bias, 0)

    @auto_fp16()
    def forward(self, roi_features, roof_logits, offsets, decoded_offsets,
                roi_boxes):
        self._check_inputs(roi_features, roof_logits, offsets, decoded_offsets,
                           roi_boxes)

        output_size = roof_logits.shape[-2:]
        visual_features = self.visual_projection(roi_features)
        visual_features = F.interpolate(
            visual_features,
            size=output_size,
            mode="bilinear",
            align_corners=False)
        offset_maps = offsets[:, :, None, None].expand(-1, -1, output_size[0],
                                                       output_size[1])
        shifted_prior_logits = self.build_shifted_roof_prior(
            roof_logits, decoded_offsets, roi_boxes)

        fused_features = torch.cat([
            visual_features,
            roof_logits,
            offset_maps,
            shifted_prior_logits,
        ],
                                   dim=1)
        correction_logits = super().forward(fused_features)
        if self.residual_refinement:
            return shifted_prior_logits + correction_logits
        return correction_logits

    def build_shifted_roof_prior(self, roof_logits, decoded_offsets,
                                 roi_boxes):
        """Translate roof probabilities opposite to footprint-to-roof offsets."""
        num_rois, _, height, width = roof_logits.shape
        if num_rois == 0:
            return roof_logits

        dtype = roof_logits.dtype
        device = roof_logits.device
        x_coordinates = torch.linspace(
            -1 + 1 / width, 1 - 1 / width, width, dtype=dtype, device=device)
        y_coordinates = torch.linspace(
            -1 + 1 / height,
            1 - 1 / height,
            height,
            dtype=dtype,
            device=device)
        grid_y, grid_x = torch.meshgrid(
            y_coordinates, x_coordinates, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        sampling_grid = base_grid.unsqueeze(0).expand(num_rois, -1, -1,
                                                      -1).clone()

        roi_widths = (roi_boxes[:, 2] - roi_boxes[:, 0]).clamp(min=1)
        roi_heights = (roi_boxes[:, 3] - roi_boxes[:, 1]).clamp(min=1)
        # grid_sample maps an output location to its source location. Adding
        # the footprint-to-roof offset therefore shifts the output toward the
        # footprint, matching offset_roof_to_footprint().
        sampling_grid[...,
                      0] += (2 * decoded_offsets[:, 0] / roi_widths)[:, None,
                                                                     None]
        sampling_grid[...,
                      1] += (2 * decoded_offsets[:, 1] / roi_heights)[:, None,
                                                                      None]

        shifted_probabilities = F.grid_sample(
            roof_logits.sigmoid(),
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False)
        shifted_probabilities = shifted_probabilities.clamp(
            min=self.prior_eps, max=1 - self.prior_eps)
        return torch.log(shifted_probabilities) - torch.log1p(
            -shifted_probabilities)

    def _check_inputs(self, roi_features, roof_logits, offsets,
                      decoded_offsets, roi_boxes):
        batch_size = roof_logits.size(0)
        if any(
                value.size(0) != batch_size
                for value in (roi_features, offsets, decoded_offsets,
                              roi_boxes)):
            raise ValueError(
                "all footprint-head inputs must have the same batch size")
        if roi_features.ndim != 4 or roi_features.size(
                1) != self.roi_in_channels:
            raise ValueError(
                f"roi_features must have shape [N, {self.roi_in_channels}, H, W]"
            )
        if roof_logits.ndim != 4 or roof_logits.size(1) != self.roof_channels:
            raise ValueError(
                f"roof_logits must have shape [N, {self.roof_channels}, H, W]")
        if offsets.ndim != 2 or offsets.size(1) != 2:
            raise ValueError("offsets must have shape [N, 2]")
        if decoded_offsets.ndim != 2 or decoded_offsets.size(1) != 2:
            raise ValueError("decoded_offsets must have shape [N, 2]")
        if roi_boxes.ndim != 2 or roi_boxes.size(1) != 4:
            raise ValueError("roi_boxes must have shape [N, 4]")

    @force_fp32(apply_to=("mask_pred", ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = super().loss(mask_pred, mask_targets, labels)
        return {"loss_footprint_mask_geometry": loss["loss_mask"]}
