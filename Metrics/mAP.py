from utils import utils
import numpy as np


class mAP:
    class AveragePrecision:
        def __init__(self, iou_th, class_id):
            self.iou_th = iou_th
            self.subject_class = class_id
            self.num_gt = 0
            self.det_list = list()

        def update(self, gt, det):
            subject_gt = gt.get(self.subject_class, [])
            subject_det = det.get(self.subject_class, [])
            self.num_gt += len(subject_gt)
            det_result = utils.classify(
                {self.subject_class: subject_gt}, {self.subject_class: subject_det}, self.iou_th)

            for box in det_result['TP'][self.subject_class]:
                self.det_list.append((box[4], True))
            for box in det_result['FP'][self.subject_class]:
                self.det_list.append((box[4], False))

        def compute(self):
            self.det_list.sort(key=lambda x: x[0], reverse=True)

            num_tp = 0
            num_fp = 0
            precisions = list()
            recalls = list()

            recall_levels = np.linspace(0.0, 1.0, 11)

            for conf, is_tp in self.det_list:
                if is_tp:
                    num_tp += 1
                else:
                    num_fp += 1

                precision = num_tp / \
                    (num_tp + num_fp) if (num_tp + num_fp) > 0 else 1.0
                recall = num_tp / self.num_gt if self.num_gt > 0 else 1.0

                precisions.append(precision)
                recalls.append(recall)
            precisions = np.array(precisions)
            recalls = np.array(recalls)
            ap = 0.0
            for rl in recall_levels:
                p_at_rl = precisions[recalls >= rl]
                if p_at_rl.size > 0:
                    ap += np.max(p_at_rl)
            ap /= len(recall_levels)
            return ap

    def __init__(self, iou_th=0.5):
        self.iou_th = iou_th
        self.subject_classes = utils.class_Map.values()
        self.ap_calculators = {class_id: self.AveragePrecision(
            iou_th, class_id) for class_id in self.subject_classes}

    def update(self, gt, det):
        for class_id in self.subject_classes:
            self.ap_calculators[class_id].update(gt, det)

    def compute(self):
        ap_values = dict()
        for class_id, calculator in self.ap_calculators.items():
            ap_values[class_id] = calculator.compute()

        mAP_value = np.mean(list(ap_values.values()))

        return mAP_value
