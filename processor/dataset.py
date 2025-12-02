from .loader import DetectionResultLoader
from .analyzer import DetectionAnalyzer


class Dataset:
    def __init__(self, gt_dir, det_dirs, iou_th):
        self.loader = DetectionResultLoader(gt_dir, det_dirs)
        self.analyzer = DetectionAnalyzer(iou_th)
        self.iou_th = iou_th
