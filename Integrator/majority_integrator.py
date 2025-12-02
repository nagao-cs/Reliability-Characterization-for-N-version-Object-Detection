from .AbsIntegrator import AbstractIntegrator
from typing import Dict, List, Any, Tuple
from math import ceil


class MajorityIntegrator(AbstractIntegrator):
    """
    Integrates object detection results from multiple versions (models) using 
    a strict Majority Voting scheme.

    Detections are grouped based on an IoU threshold. A resulting integrated 
    bounding box is generated only if the group contains detections from a **strict majority** of the unique versions (i.e., more than half).

    The final integrated box is the average of all detections in the accepted group.

    Attributes:
        iou_th (float): The Intersection over Union (IoU) threshold used for 
                        grouping overlapping detections from different versions.
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
        Integrates detection results from multiple versions using the Majority Voting scheme.

        1. **Consensus Threshold**: The required number of unique versions (votes) is 
           calculated as `ceil(total_versions / 2)`.
        2. **Flatten & Group**: Detections are collected and grouped if they overlap 
           above `self.iou_th` with any member of an existing group (connected components clustering).
        3. **Majority Vote**: An integrated box is only generated if the group's 
           **unique version count** meets the `majority_threshold`.
        4. **Averaging**: The final bounding box and confidence are determined by 
           averaging the values of all detections in the valid group.

        Args:
            detections (Dict[int, Dict[int, List[tuple]]]): Detection results from multiple versions.
                        Format: `{version_id: {class_id: [(x_center, y_center, width, height, confidence), ...], ...}, ...}`

        Returns:
            Dict[int, List[tuple]]: The integrated and averaged detection results.
                                    Format: `{class_id: [(avg_x, avg_y, avg_w, avg_h, avg_conf), ...], ...}`
        """
        integrated: Dict[int, List[Tuple]] = dict()
        total_versions = len(detections)
        # Calculate the strict majority threshold: more than half the versions must agree.
        majority_threshold = ceil(total_versions / 2)
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
                    # Input assumed to be: (x_center, y_center, width, height, confidence)
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

                # Iteratively add other unprocessed detections to the group
                for j in range(i + 1, len(all_dets)):
                    if is_processed[j]:
                        continue

                    cand = all_dets[j]

                    if cand['version_id'] == base['version_id']:
                        continue

                    matched = False
                    for member in group:
                        # Check IoU against any existing member in the group
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
                    # Average the parameters of all detections in the accepted group
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
