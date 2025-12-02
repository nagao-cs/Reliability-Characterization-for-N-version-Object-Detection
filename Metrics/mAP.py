from utils import utils
import numpy as np
from typing import Dict, List, Tuple, Any


class mAP:
    """
    Calculates the Mean Average Precision (mAP) metric for object detection.

    Attributes:
        iou_th (float): The Intersection over Union (IoU) threshold used for 
                        classifying detections as True Positives.
        subject_classes (List[int]): A list of class IDs for which AP will be calculated. 
                                     (Assumed to be retrieved from `utils.class_Map.values()`).
        ap_calculators (Dict[int, mAP.AveragePrecision]): A dictionary mapping class IDs 
                                                          to their respective AP calculation objects.
    """

    class AveragePrecision:
        """
        Calculates the Average Precision (AP) for a single object class.

        Attributes:
            iou_th (float): IoU threshold for TP/FP classification.
            subject_class (int): The ID of the object class being analyzed.
            num_gt (int): The total count of ground truth instances for this class across all frames.
            det_list (List[Tuple[float, bool]]): A list of detection results, stored as 
                                                 (confidence score, is_true_positive).
        """

        def __init__(self, iou_th: float, class_id: int):
            """
            Initializes the AP calculator for a specific class ID and IoU threshold.

            Args:
                iou_th (float): IoU threshold.
                class_id (int): The class ID to calculate AP for.
            """
            self.iou_th = iou_th
            self.subject_class = class_id
            self.num_gt = 0
            self.det_list = list()

        def update(self, gt: Dict[int, List], det: Dict[int, List]):
            """
            Processes the detection and ground truth results for a single frame, 
            classifies the detections, and accumulates the results.

            Only detections/GT belonging to `self.subject_class` are considered.

            Args:
                gt (Dict[int, List]): Ground truth boxes for the current frame 
                                      (Format: {class_id: [boxes]}).
                det (Dict[int, List]): Detection boxes for the current frame 
                                       (Format: {class_id: [boxes]}). Boxes are assumed 
                                       to be `(x, y, w, h, confidence_score, ...)`.
            """
            subject_gt = gt.get(self.subject_class, [])
            subject_det = det.get(self.subject_class, [])

            self.num_gt += len(subject_gt)

            det_result = utils.classify(
                {self.subject_class: subject_gt}, {self.subject_class: subject_det}, self.iou_th)

            for box in det_result['TP'][self.subject_class]:
                self.det_list.append((box[4], True))
            for box in det_result['FP'][self.subject_class]:
                self.det_list.append((box[4], False))

        def compute(self) -> float:
            """
            Computes the Average Precision (AP) for the `subject_class` based on 
            the accumulated detection list.

            This implementation uses the 11-point interpolation method (sampling 
            Recall at [0.0, 0.1, ..., 1.0]).

            Returns:
                float: The calculated Average Precision (AP) value for the class.
            """
            # 1. Sort all detections by confidence score in descending order
            self.det_list.sort(key=lambda x: x[0], reverse=True)

            num_tp = 0
            num_fp = 0
            precisions = list()
            recalls = list()

            recall_levels = np.linspace(0.0, 1.0, 11)

            # 2. Compute Precision and Recall at every detection point
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

            # 3. 11-point Interpolation: Find the maximum precision for each recall level
            for rl in recall_levels:
                # Get all precision values where Recall >= current recall level (rl)
                p_at_rl = precisions[recalls >= rl]

                if p_at_rl.size > 0:
                    # Take the maximum precision at this recall level
                    ap += np.max(p_at_rl)

            # 4. Average the 11 precision values
            ap /= len(recall_levels)
            return ap

    def __init__(self, iou_th: float = 0.5):
        """
        Initializes the mAP calculator.

        Args:
            iou_th (float, optional): The IoU threshold for AP calculation. Defaults to 0.5.
        """
        self.iou_th = iou_th
        # Assumes utils.class_Map exists and returns class IDs (e.g., [1, 2, 3...])
        self.subject_classes = utils.class_Map.values()
        self.ap_calculators = {class_id: self.AveragePrecision(
            iou_th, class_id) for class_id in self.subject_classes}

    def update(self, gt: Dict[int, List], det: Dict[int, List]):
        """
        Processes the detection and ground truth results for a single frame 
        by updating all per-class AP calculators.

        Args:
            gt (Dict[int, List]): Ground truth boxes for the current frame.
            det (Dict[int, List]): Detection boxes for the current frame.
        """
        for class_id in self.subject_classes:
            self.ap_calculators[class_id].update(gt, det)

    def compute(self) -> float:
        """
        Computes the final Mean Average Precision (mAP) value by averaging the 
        AP values across all subject classes.

        Returns:
            float: The calculated mAP value.
        """
        ap_values = dict()
        for class_id, calculator in self.ap_calculators.items():
            ap_values[class_id] = calculator.compute()

        # mAP is the mean of all AP values
        mAP_value = np.mean(list(ap_values.values()))

        return mAP_value
