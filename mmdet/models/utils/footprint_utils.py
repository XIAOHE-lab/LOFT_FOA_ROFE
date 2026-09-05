"""Utilities for converting roof masks to footprint masks."""

import math

import numpy as np


def offset_roof_to_footprint(offsets, roof_masks):
    """Translate each roof mask by the negative roof offset."""
    footprint_masks = []
    for image_offsets, image_roof_masks in zip(offsets, roof_masks):
        image_results = [[] for _ in image_roof_masks]
        for class_index, class_masks in enumerate(image_roof_masks):
            for offset, roof_mask in zip(image_offsets, class_masks):
                height, width = roof_mask.shape
                border = max(
                    math.ceil(abs(float(offset[0]))),
                    math.ceil(abs(float(offset[1]))),
                )
                canvas = np.zeros(
                    (height + 2 * border, width + 2 * border), dtype=bool
                )
                canvas[border : border + height, border : border + width] = roof_mask
                canvas = np.roll(
                    canvas,
                    shift=(-int(offset[1]), -int(offset[0])),
                    axis=(0, 1),
                )
                image_results[class_index].append(
                    canvas[border : border + height, border : border + width]
                )
        footprint_masks.append(image_results)
    return footprint_masks

