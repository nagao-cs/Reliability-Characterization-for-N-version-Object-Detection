from .AbsIntegrator import AbstractIntegrator
from typing import Dict, List, Any, Tuple
from math import ceil


class AffirmativeIntegrator(AbstractIntegrator):
    """
    Integrates object detection results from multiple versions (models) using 
    an Affirmative method.

    The final integrated box is the average of all detections in the accepted group.

    Attributes:
        iou_th (float): The Intersection over Union (IoU) threshold used for 
                        grouping detections from different versions.
    """

    def __init__(self, iou_th: float = 0.5):
        """
        Initializes the integrator with the grouping IoU threshold.

        Args:
            iou_th (float, optional): IoU threshold for grouping detections. Defaults to 0.5.
        """
        self.iou_th = iou_th

    def integrate(self, detections: Dict[int, Dict[int, List[Tuple]]]) -> Dict[int, List[Tuple]]:
        """
        Integrates detection results from multiple versions using the Affirmative Voting scheme.

        1. **Flatten & Group**: Collects all detections across all versions for each class.
        2. **Clustering**: Groups detections that overlap above `self.iou_th`. A basic 
           connected-components approach is used where a new detection is added to an 
           existing group if it overlaps with *any* member of that group.
        3. **Majority Vote**: If a group contains detections from at least `majority_threshold` 
           (currently 1) unique versions, it is considered a valid detection.
        4. **Averaging**: The final bounding box and confidence are determined by averaging 
           the values of all detections within the valid group.

        Args:
            detections (Dict[int, Dict[int, List[tuple]]]): Detection results from multiple versions.
                        Format: `{version_id: {class_id: [(x_center, y_center, width, height, confidence), ...], ...}, ...}`

        Returns:
            Dict[int, List[tuple]]: The integrated and averaged detection results.
                                    Format: `{class_id: [(avg_x, avg_y, avg_w, avg_h, avg_conf), ...], ...}`
        """
        integrated: Dict[int, List[Tuple]] = dict()
        # Note: The implementation currently uses majority_threshold = 1,
        # which means a detection from any single version is sufficient to form a group.
        majority_threshold = 1
        IOU_THRESHOLD = self.iou_th

        # Find all unique class IDs present across all versions
        all_class_ids = set()
        for v_dets in detections.values():
            all_class_ids.update(v_dets.keys())

        for class_id in all_class_ids:
            # 1. Flatten all detections for the current class
            all_dets = []
            for version_id, v_dets in detections.items():
                boxes = v_dets.get(class_id, [])
                for b in boxes:
                    x, y, w, h, conf = b
                    all_dets.append({
                        'version_id': version_id,
                        'x_center': float(x),
                        'y_center': float(y),
                        'width': float(w),
                        'height': float(h),
                        'confidence': float(conf)
                    })

            if not all_dets:
                continue

            is_processed = [False] * len(all_dets)
            groups = []

            # 2. Grouping (Clustering based on IoU)
            for i in range(len(all_dets)):
                if is_processed[i]:
                    continue

                base = all_dets[i]
                group = [base]
                is_processed[i] = True

                for j in range(i + 1, len(all_dets)):
                    if is_processed[j]:
                        continue

                    cand = all_dets[j]

                    matched = False
                    for member in group:
                        if self._iou(member, cand) >= IOU_THRESHOLD:
                            matched = True
                            break

                    if matched:
                        group.append(cand)
                        is_processed[j] = True

                groups.append(group)

            # 3. & 4. Majority Vote and Averaging
            integrated[class_id] = []
            for group in groups:
                # Count unique versions to check majority rule
                unique_versions = set(d['version_id'] for d in group)

                if len(unique_versions) >= majority_threshold:
                    # Average the parameters of all detections in the group
                    n = len(group)
                    avg_x = sum(d['x_center'] for d in group) / n
                    avg_y = sum(d['y_center'] for d in group) / n
                    avg_w = sum(d['width'] for d in group) / n
                    avg_h = sum(d['height'] for d in group) / n
                    avg_conf = sum(d['confidence'] for d in group) / n

                    integrated[class_id].append(
                        (avg_x, avg_y, avg_w, avg_h, avg_conf))

            # Remove class entry if no detections were integrated
            if not integrated[class_id]:
                del integrated[class_id]

        return integrated

    def _iou(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes, 
        given in center-based coordinates (x_center, y_center, width, height).

        Args:
            a (Dict[str, Any]): First bounding box dictionary.
            b (Dict[str, Any]): Second bounding box dictionary.

        Returns:
            float: The IoU value between the two boxes. 
        """
        # Convert center-based to min-max coordinates for box A
        x1_min = a['x_center'] - a['width'] / 2
        y1_min = a['y_center'] - a['height'] / 2
        x1_max = a['x_center'] + a['width'] / 2
        y1_max = a['y_center'] + a['height'] / 2

        # Convert center-based to min-max coordinates for box B
        x2_min = b['x_center'] - b['width'] / 2
        y2_min = b['y_center'] - b['height'] / 2
        x2_max = b['x_center'] + b['width'] / 2
        y2_max = b['y_center'] + b['height'] / 2

        # Calculate intersection coordinates
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        # Calculate intersection area
        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        # Calculate individual areas
        area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
        area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

        # Calculate union area
        union = area1 + area2 - inter_area

        if union <= 0.0:
            return 0.0

        # IoU = Intersection / Union
        return inter_area / union

    def __call__(self, dets: Dict[int, Dict[int, List[Tuple]]]) -> Dict[int, List[Tuple]]:
        """
        Args:
            dets (Dict[int, Dict[int, List[tuple]]]): Detection results from multiple versions.

        Returns:
            Dict[int, List[tuple]]: The integrated detection results.
        """
        return self.integrate(dets)
