from utils import utils


class Accuracy:
    def __init__(self, iou_th: float = 0.5):
        self.val = 0.0
        self.iou_th = iou_th
        self.num_frames = 0

    def update(self, gt, det):
        det_results = utils.classify(gt, det, self.iou_th)

        correct_detections = sum(len(boxes)
                                 for boxes in det_results['TP'].values())
        total_instances = sum(len(boxes) for boxes in det_results['TP'].values()) + sum(len(
            boxes) for boxes in det_results['FP'].values()) + sum(len(boxes) for boxes in det_results['FN'].values())
        self.val += correct_detections / total_instances if total_instances > 0 else 1.0
        self.num_frames += 1

    def compute(self):
        if self.num_frames == 0:
            return 1.0
        return self.val / self.num_frames
