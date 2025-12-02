from .loader import DetectionResultLoader
from .analyzer import DetectionAnalyzer


class Dataset:
    """
    Unified interface for loading and analyzing object detection results.

    Combines ground truth labels, multi-version detection outputs, and analysis
    functionality. Provides convenient access to frame-by-frame data with
    integrated IoU-based matching and error classification.

    Attributes:
        loader (DetectionResultLoader): Handles loading GT and detection files.
        analyzer (DetectionAnalyzer): Performs frame-level analysis and classification.
        iou_th (float): IoU threshold for TP/FP classification.

    Example:
        >>> dataset = Dataset(
        ...     gt_dir='./labels/front',
        ...     det_dirs=['./detections/yolov8n/front', './detections/rtdetr/front'],
        ...     iou_th=0.5
        ... )
        >>> for frame_idx, gt, dets in dataset.loader.iter_frame():
        ...     analysis = dataset.analyzer.analyze_frame(gt, dets)
        ...     print(analysis['tp_count'], analysis['fp_count'], analysis['fn_count'])
    """

    def __init__(self, gt_dir, det_dirs, iou_th):
        """
        Initialize dataset with ground truth and detection directories.

        Args:
            gt_dir (str): Path to directory containing ground truth annotation files.
            det_dirs (list): List of paths to detection result directories for each model version.
            iou_th (float): IoU threshold for true positive classification. Default: 0.5.

        """
        self.loader = DetectionResultLoader(gt_dir, det_dirs)
        self.analyzer = DetectionAnalyzer(iou_th)
        self.iou_th = iou_th
