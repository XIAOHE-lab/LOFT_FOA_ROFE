from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class ROFMaskRCNN(TwoStageDetector):
    """Detect roofs and predict offsets and footprint masks.

    Task-specific training and inference live in ``FootprintMaskRoIHead``.
    This class intentionally adds no angle, height, or semi-supervised logic.
    """

    def show_result(self, img, result, *args, **kwargs):
        """Visualize detections and roof masks from an inference result."""
        if isinstance(result, dict):
            result = result["bbox"], result["roof_mask"]
        elif isinstance(result, tuple):
            result = result[:2]
        return super().show_result(img, result, *args, **kwargs)
