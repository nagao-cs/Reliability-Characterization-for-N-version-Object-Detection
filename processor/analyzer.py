from utils import utils
from typing import Dict, List


class DetectionAnalyzer:
    """
    Frame-level analysis of detection results against ground truth.

    Performs per-frame IoU-based matching between detections and ground truth,
    classifies errors (TP, FP, FN), and computes analysis statistics for
    reliability and coverage metrics.

    Attributes:
        iou_th (float): IoU threshold for true positive classification.
    """

    def __init__(self, iou_th: float):
        """
        Initialize analyzer with IoU threshold.

        Args:
            iou_th (float): IoU threshold for matching detections to GT. Default: 0.5.
        """
        self.iou_th = iou_th

    def analyze_frame(self, gt: dict, dets: Dict[int, Dict[int, List]]) -> dict:
        """
        Analyze a single frame of detections vs. ground truth.

        Matches detections to ground truth objects using IoU-based Hungarian algorithm,
        classifies results (TP, FP, FN), and organizes errors for metric computation.

        Args:
            gt (dict): Ground truth {class_id: [(x, y, w, h, dist), ...], ...}
            dets (dict): Detections {model_idx: {class_id: [...], ...}, ...}
            mode (str): Analysis mode ('single' or 'multi'). Affects error classification.

        Returns:
            Dictionary containing:
            - 'tp_count' (int): True positives.
            - 'fp_count' (int): False positives.
            - 'fn_count' (int): False negatives.
            - 'classified_results' (dict): Detailed per-detection results.
            - 'intersection_errors' (dict): Errors for CovOD/CerOD metrics.
            - 'union_errors' (dict): Errors for coverage metrics.
            - 'total_instances' (dict): Instance counts per class.
            - 'is_correct_list' (list): Per-class correctness flags.

        Example:
            >>> analysis = analyzer.analyze_frame(gt, dets, mode='multi')
            >>> print(f"TP: {analysis['tp_count']}, FP: {analysis['fp_count']}, FN: {analysis['fn_count']}")
        """
        classified_results = self._classify_frame(
            gt, dets)
        intersection_errors = self._intersection_of_errors(classified_results)
        union_errors = self._union_of_errors(classified_results)
        total_instances = self._total_instances(classified_results)
        is_correct_list = self.is_corect_detection(classified_results)
        return {
            'classified_results': classified_results,
            'intersection_errors': intersection_errors,
            'union_errors': union_errors,
            'total_instances': total_instances,
            'is_correct_list': is_correct_list
        }

    def compare_detections(self, gt, dets) -> dict:
        """
        Compares detection errors across multiple model versions, identifying 
        errors that are **specific** to the base model (version 0).

        An error (FP or FN) in the base model is considered 'specific' if no
        corresponding error (with IoU >= iou_th) is found in any other model version.

        Args:
            gt (dict): Ground truth data.
            dets (Dict[int, Dict[int, List]]): Detections from multiple models/versions.
                                                Model 0 is treated as the base model for comparison.

        Returns:
            dict: A dictionary containing errors specific to the base model:
            - 'model_specific_FP' (dict): FP boxes in model 0 that are NOT FPs in any other model (version > 0).
            - 'model_specific_FN' (dict): FN boxes in model 0 that are NOT FNs in any other model (version > 0).
        """
        comparison_results = {
            'model_specific_FP': dict(),
            'model_specific_FN': dict(),
        }

        classified_dets = self._classify_frame(gt, dets)
        base_det = classified_dets[0]
        for error_type in ['FP', 'FN']:
            base_errors = base_det[error_type]
            specific_errors = dict()
            for class_id, base_boxes in base_errors.items():
                specific_boxes = []
                for box in base_boxes:
                    is_specific = True
                    for version, classified_det in classified_dets.items():
                        if version == 0:
                            continue
                        current_errors = classified_det.get(error_type, {})
                        if class_id not in current_errors:
                            continue
                        for curr_box in current_errors[class_id]:
                            if utils.compute_iou(box, curr_box) >= self.iou_th:
                                is_specific = False
                                break
                        if not is_specific:
                            break
                    if is_specific:
                        specific_boxes.append(box)
                if specific_boxes:
                    specific_errors[class_id] = specific_boxes
            comparison_results[f'model_specific_{error_type}'] = specific_errors
        return comparison_results

    def _intersection_of_errors(self, classified_dets) -> dict:
        """
        Calculates the intersection of FP and FN errors across all model versions using IoU-based matching.

        An error (FP or FN box) is in the intersection if it has a match (IoU >= iou_th) 
        in the error set of *every* model version.

        Args:
            classified_dets (Dict[int, dict]): The classification results for all model versions.

        Returns:
            dict: The intersection of errors, partitioned by error type ('FP', 'FN') and class ID.
        """
        intersection_errors = {'FP': dict(), 'FN': dict()}
        for error_type in ['FP', 'FN']:
            for version, classified_det in classified_dets.items():
                current_errors = classified_det[error_type]

                if version == 0:
                    for class_id, boxes in current_errors.items():
                        intersection_errors[error_type][class_id] = boxes.copy(
                        )
                    continue

                new_intersection = dict()
                for class_id, base_boxes in intersection_errors[error_type].items():
                    if class_id not in current_errors:
                        continue
                    matched_boxes = []
                    used = set()
                    for base_box in base_boxes:
                        best_iou = 0.0
                        best_box = None
                        for curr_box in current_errors[class_id]:
                            if curr_box in used:
                                continue
                            iou = utils.compute_iou(base_box, curr_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_box = curr_box
                        if best_iou >= self.iou_th:
                            matched_boxes.append(base_box)
                            used.add(best_box)
                    if matched_boxes:
                        new_intersection[class_id] = matched_boxes
                intersection_errors[error_type] = new_intersection
        return intersection_errors

    def _union_of_errors(self, classified_dets) -> dict:
        """
        Calculates the union of FP and FN errors across all 
        model versions.

        An error (FP or FN box) is in the union if it is an error in *at least one* model version. Duplicates are removed based on IoU matching (IoU >= iou_th). 

        Args:
            classified_dets (Dict[int, dict]): The classification results for all model versions.

        Returns:
            dict: The union of errors, partitioned by error type ('FP', 'FN') and class ID.
        """
        union_errors = {'FP': dict(), 'FN': dict()}
        for error_type in ['FP', 'FN']:
            for version, classified_det in classified_dets.items():
                current_errors = classified_det[error_type]
                if version == 0:
                    for class_id, boxes in current_errors.items():
                        union_errors[error_type][class_id] = boxes.copy(
                        )
                    continue
                for class_id, boxes in current_errors.items():
                    if class_id not in union_errors[error_type]:
                        union_errors[error_type][class_id] = []
                    for box in boxes:
                        already_added = False
                        for existing_box in union_errors[error_type][class_id]:
                            if utils.compute_iou(box, existing_box) >= self.iou_th:
                                already_added = True
                                break
                        if not already_added:
                            union_errors[error_type][class_id].append(box)
        return union_errors

    def _intersection_of_tp(self, classified_dets: dict[dict]) -> dict:
        """
        Calculates the intersection of True Positives (TP) across 
        all model versions.

        A TP is in the intersection if it has a match (IoU >= iou_th) in the TP set 
        of *every* model version. This set represents the most robust detections 
        common to all versions.

        Args:
            classified_dets (Dict[int, dict]): The classification results for all model versions.

        Returns:
            dict: The intersection of TPs, partitioned by class ID.
        """
        intersection_tp = dict()
        for version, classified_det in classified_dets.items():
            current_tp = classified_det['TP']

            if version == 0:
                for class_id, boxes in current_tp.items():
                    intersection_tp[class_id] = boxes.copy()
                continue

            new_intersection = dict()
            for class_id, base_boxes in intersection_tp.items():
                if class_id not in current_tp:
                    continue
                matched_boxes = []
                used = set()
                for base_box in base_boxes:
                    best_iou = 0.0
                    best_box = None
                    for curr_box in current_tp[class_id]:
                        if curr_box in used:
                            continue
                        iou = utils.compute_iou(base_box, curr_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_box = curr_box
                    if best_iou >= self.iou_th:
                        matched_boxes.append(base_box)
                        used.add(best_box)
                if matched_boxes:
                    new_intersection[class_id] = matched_boxes
            intersection_tp = new_intersection
        return intersection_tp

    def _total_instances(self, classified_dets: dict[dict]) -> dict:
        """
        Combines the calculated intersection of TPs and the union of FPs/FNs 
        to represent the total set of instances across the metrics.

        - **TP** instances are the **intersection** across all models
        - **FP** and **FN** instances are the **union** across all models.

        Args:
            classified_dets (Dict[int, dict]): The classification results for all model versions.

        Returns:
            dict: The total set of instances for each error type, used for denominator calculations 
                  in aggregated metrics. Format: `{'TP': {...}, 'FP': {...}, 'FN': {...}}`.
        """
        total_instances = {"TP": dict(), "FP": dict(), "FN": dict()}
        union_errors = self._union_of_errors(classified_dets)
        for error_type in ['FP', 'FN']:
            for class_id in union_errors[error_type].keys():
                total_instances[error_type][class_id] = union_errors[error_type][class_id]

        intersection_tp = self._intersection_of_tp(classified_dets)
        for class_id in intersection_tp.keys():
            total_instances["TP"][class_id] = intersection_tp[class_id]

        return total_instances

    def _classify_frame(self, gt, dets) -> dict[dict]:
        """
        Classifies the detections for ALL model versions in a single frame into TP/FP/FN.

        Args:
            gt (dict): Ground truth data.
            dets (Dict[int, Dict[int, List]]): Detections from multiple models/versions.

        Returns:
            Dict[int, dict]: A dictionary of classification results keyed by model version ID.
        """
        frame_results = dict()
        for version, det in dets.items():
            frame_results[version] = self._classify(gt, det)
        return frame_results

    def _classify(self, gt, det) -> dict:
        """
        Performs the core classification of detections into True Positives (TP), 
        False Positives (FP), and False Negatives (FN) for a SINGLE model version 
        against the Ground Truth.

        Uses a greedy, highest-IoU matching strategy (not the full Hungarian algorithm) 
        where a detection is classified as a TP if it matches an unused GT box 
        with IoU >= iou_th. 

        Args:
            gt (dict): Ground truth data for the frame.
            det (Dict[int, List]): Detections for a single model version.

        Returns:
            dict: The classification results for the single version: 
                  `{'TP': {class_id: [boxes]}, 'FP': {class_id: [boxes]}, 'FN': {class_id: [boxes]}}`
        """
        det_results = {'TP': dict(), 'FP': dict(), 'FN': dict()}
        for class_id in gt.keys():
            gt_boxes = gt[class_id]
            det_boxes = det.get(class_id, [])

            if class_id not in det_results['TP']:
                det_results['TP'][class_id] = list()
            if class_id not in det_results['FN']:
                det_results['FN'][class_id] = list()
            if class_id not in det_results['FP']:
                det_results['FP'][class_id] = list()

            used_gt = set()

            for det_box in det_boxes:
                best_iou = 0.0
                best_gt_idx = -1

                for i, gt_box in enumerate(gt_boxes):
                    if i in used_gt:
                        continue
                    iou = utils.compute_iou(gt_box, det_box)
                    if iou >= self.iou_th and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_gt_idx >= 0:
                    det_results['TP'][class_id].append(det_box)
                    used_gt.add(best_gt_idx)
                else:
                    det_results['FP'][class_id].append(det_box)

            for i, gt_box in enumerate(gt_boxes):
                if i not in used_gt:
                    det_results['FN'][class_id].append(gt_box)
        return det_results

    def is_corect_detection(self, classified_results) -> List[bool]:
        """
        Checks if each model version achieved a perfect detection (i.e., zero FP and zero FN).

        Args:
            classified_results (Dict[int, dict]): The classification results for all model versions.

        Returns:
            List[bool]: A list where each element is True if the corresponding model version 
                        has no FP or FN boxes, and False otherwise.
        """
        is_correct_list = []
        for version, result in classified_results.items():
            if all(len(boxes) == 0 for boxes in result['FP'].values()) and all(len(boxes) == 0 for boxes in result['FN'].values()):
                is_correct_list.append(True)
            else:
                is_correct_list.append(False)
        return is_correct_list
