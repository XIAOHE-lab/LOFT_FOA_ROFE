"""RoI head for roof, offset, and footprint-mask prediction."""

import numpy as np
import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from ..utils import offset_roof_to_footprint
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class FootprintMaskRoIHead(StandardRoIHead):
    """Predict a roof mask, roof-to-footprint offset, and footprint mask."""

    def __init__(
        self,
        offset_roi_extractor,
        offset_head,
        footprint_mask_head,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.offset_roi_extractor = build_roi_extractor(offset_roi_extractor)
        self.offset_head = build_head(offset_head)
        self.footprint_mask_head = build_head(footprint_mask_head)

    def init_weights(self):
        super().init_weights()

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        gt_offsets=None,
        gt_footprint_masks=None,
        **kwargs,
    ):
        if gt_offsets is None or gt_footprint_masks is None:
            raise ValueError("gt_offsets and gt_footprint_masks are required")

        sampling_results = self._sample_proposals(x, proposal_list, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
        losses = {}

        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes,
                                                gt_labels, img_metas)
        losses.update(bbox_results["loss_bbox"])

        roof_results = self._mask_forward_train(x, sampling_results,
                                                bbox_results["bbox_feats"],
                                                gt_masks, img_metas)
        losses.update(roof_results["loss_mask"])

        offset_results = self._offset_forward_train(x, sampling_results,
                                                    gt_offsets)
        losses.update(offset_results["loss_offset"])

        # OffsetHeadExpandFeature concatenates predictions branch-first. The
        # first num_positive rows are the canonical (unrotated) predictions.
        num_positive = sum(
            result.pos_bboxes.size(0) for result in sampling_results)
        canonical_offsets = offset_results["offset_pred"][:num_positive]
        footprint_boxes = None
        if getattr(self.footprint_mask_head, "requires_geometry_inputs",
                   False):
            footprint_boxes = torch.cat(
                [result.pos_bboxes for result in sampling_results])
        footprint_pred = self._footprint_mask_forward(
            canonical_offsets,
            roof_results["mask_pred"],
            roof_results["mask_feats"],
            footprint_boxes,
        )
        footprint_targets = self.footprint_mask_head.get_targets(
            sampling_results, gt_footprint_masks, self.train_cfg)
        positive_labels = torch.cat(
            [result.pos_gt_labels for result in sampling_results])
        losses.update(
            self.footprint_mask_head.loss(footprint_pred, footprint_targets,
                                          positive_labels))
        return losses

    def _sample_proposals(self, x, proposal_list, gt_bboxes, gt_labels,
                          gt_bboxes_ignore):
        num_images = len(gt_bboxes)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None] * num_images

        sampling_results = []
        for index in range(num_images):
            assign_result = self.bbox_assigner.assign(
                proposal_list[index],
                gt_bboxes[index],
                gt_bboxes_ignore[index],
                gt_labels[index],
            )
            sampling_results.append(
                self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[index],
                    gt_bboxes[index],
                    gt_labels[index],
                    feats=[level[index][None] for level in x],
                ))
        return sampling_results

    def _offset_forward(self, x, rois):
        offset_features = self.offset_roi_extractor(
            x[:self.offset_roi_extractor.num_inputs], rois)
        return self.offset_head(offset_features)

    def _offset_forward_train(self, x, sampling_results, gt_offsets):
        positive_rois = bbox2roi(
            [result.pos_bboxes for result in sampling_results])
        offset_pred = self._offset_forward(x, positive_rois)
        offset_targets = self.offset_head.get_targets(sampling_results,
                                                      gt_offsets,
                                                      self.train_cfg)
        return dict(
            offset_pred=offset_pred,
            loss_offset=self.offset_head.loss(offset_pred, offset_targets),
        )

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    rescale=False,
                    **kwargs):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(boxes, labels, self.bbox_head.num_classes)
            for boxes, labels in zip(det_bboxes, det_labels)
        ]
        counts = [len(boxes) for boxes in det_bboxes]

        if sum(counts) == 0:
            empty_masks = [[[] for _ in range(self.mask_head.num_classes)]
                           for _ in counts]
            return dict(
                bbox=bbox_results,
                roof_mask=empty_masks,
                offset=[np.empty((0, 2), dtype=np.float32) for _ in counts],
                offset_footprint=empty_masks,
                footprint_mask=empty_masks,
            )

        roof_masks, roof_pred = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)
        test_boxes = self._boxes_at_test_scale(det_bboxes, img_metas, rescale)
        offset_rois = bbox2roi(test_boxes)
        expanded_offset_pred = self._offset_forward(x, offset_rois)
        offsets = self._decode_offsets(expanded_offset_pred, test_boxes,
                                       img_metas, counts, rescale)

        canonical_offset_pred = expanded_offset_pred[:sum(counts)]
        footprint_roi_features = None
        if getattr(self.footprint_mask_head, "requires_roi_features", False):
            footprint_roi_features = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], offset_rois)
            if self.with_shared_head:
                footprint_roi_features = self.shared_head(
                    footprint_roi_features)
        footprint_pred = self._footprint_mask_forward(canonical_offset_pred,
                                                      roof_pred,
                                                      footprint_roi_features,
                                                      torch.cat(test_boxes))
        footprint_masks = self._mask_predictions_to_results(
            footprint_pred,
            test_boxes,
            det_labels,
            img_metas,
            counts,
            rescale,
        )
        offset_footprints = offset_roof_to_footprint(offsets, roof_masks)

        return dict(
            bbox=bbox_results,
            roof_mask=roof_masks,
            offset=offsets,
            offset_footprint=offset_footprints,
            footprint_mask=footprint_masks,
        )

    def _footprint_mask_forward(self,
                                offsets,
                                roof_logits,
                                roi_features=None,
                                roi_boxes=None):
        """Call either the original or visual-feature footprint head."""
        if getattr(self.footprint_mask_head, "requires_geometry_inputs",
                   False):
            if roi_features is None or roi_boxes is None:
                raise RuntimeError(
                    "the configured footprint head requires RoI features and boxes"
                )
            decoded_offsets = self.offset_head.offset_coder.decode(
                roi_boxes, offsets)
            return self.footprint_mask_head(roi_features, roof_logits, offsets,
                                            decoded_offsets, roi_boxes)
        if getattr(self.footprint_mask_head, "requires_roi_features", False):
            if roi_features is None:
                raise RuntimeError(
                    "the configured footprint head requires RoI features")
            return self.footprint_mask_head(roi_features, roof_logits, offsets)
        return self.footprint_mask_head(offsets, roof_logits)

    @staticmethod
    def _boxes_at_test_scale(det_bboxes, img_metas, rescale):
        boxes = []
        for det_bbox, meta in zip(det_bboxes, img_metas):
            box = det_bbox[:, :4]
            if rescale:
                scale_factor = box.new_tensor(meta["scale_factor"])
                box = box * scale_factor
            boxes.append(box)
        return boxes

    def _decode_offsets(self, expanded_pred, boxes, img_metas, counts,
                        rescale):
        total = sum(counts)
        branch_count = self.offset_head.expand_feature_num
        branches = expanded_pred.split(total, dim=0)
        if len(branches) != branch_count:
            raise RuntimeError("unexpected expanded offset prediction shape")

        results = []
        start = 0
        for count, image_boxes, meta in zip(counts, boxes, img_metas):
            stop = start + count
            if count == 0:
                results.append(np.empty((0, 2), dtype=np.float32))
                start = stop
                continue
            image_pred = torch.cat([branch[start:stop] for branch in branches],
                                   dim=0)
            results.append(
                self.offset_head.get_offsets(
                    image_pred,
                    image_boxes,
                    meta["scale_factor"],
                    rescale,
                    img_shape=meta["img_shape"][:2],
                ))
            start = stop
        return results

    def _mask_predictions_to_results(self, mask_pred, boxes, det_labels,
                                     img_metas, counts, rescale):
        predictions = mask_pred.split(counts, dim=0)
        results = []
        for pred, boxes, labels, meta in zip(predictions, boxes, det_labels,
                                             img_metas):
            if len(boxes) == 0:
                results.append(
                    [[] for _ in range(self.footprint_mask_head.num_classes)])
                continue
            results.append(
                self.footprint_mask_head.get_seg_masks(
                    pred,
                    boxes[:, :4],
                    labels,
                    self.test_cfg,
                    meta["ori_shape"],
                    meta["scale_factor"],
                    rescale,
                ))
        return results
