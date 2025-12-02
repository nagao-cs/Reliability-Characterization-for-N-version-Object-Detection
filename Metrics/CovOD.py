from typing import Dict, List, Any


class CovOD:
    """
    Calculates the Confidence-aware Object Detection (CovOD) metric.

    The metric is calculated on a per-frame basis by finding the ratio of **common errors** (FP Intersection + FN Intersection) to the total number of instances (TP Intersection + FP Union + FN Union), 
    and then averaging this ratio across all frames.

    The final result represents the system's reliability against common, persistent failures.

    Attributes:
        val (float): The accumulated sum of the per-frame common error ratios.
        num_frames (int): The total number of frames processed.
    """

    def __init__(self):
        """
        Initializes the CovOD counters.
        """
        self.val = 0.0
        self.num_frames = 0

    def update(self, intersection_errors: Dict[str, dict], total_instances: Dict[str, dict]) -> None:
        """
        Updates the accumulated common error ratio (`self.val`) for a single frame.

        The per-frame common error ratio is calculated as:
        $$
        \frac{\text{Intersection Errors (FP} \cap \text{FN)}}{\text{Total Instances (TP}_{\cap} + \text{FP}_{\cup} + \text{FN}_{\cup})}
        $$

        Args:
            intersection_errors (Dict[str, dict]): Dictionary containing the **Intersection** of FP and FN boxes 
                                                   across all model versions (i.e., errors observed in ALL versions).
                                                   Format: `{'FP': {class_id: [boxes]}, 'FN': {class_id: [boxes]}}`.
            total_instances (Dict[str, dict]): Dictionary containing the total set of instances considered 
                                               for the denominator. This is typically: TP as Intersection ($\cap$), 
                                               and FP/FN as Union ($\cup$).
                                               Format: `{'TP': {Intersection boxes}, 'FP': {Union boxes}, 'FN': {Union boxes}}`.
        """
        intersection_fp = sum(
            len(boxes) for boxes in intersection_errors['FP'].values())
        intersection_fn = sum(
            len(boxes) for boxes in intersection_errors['FN'].values())

        total_instance_count = sum(len(boxes) for boxes in total_instances['TP'].values()) + sum(
            len(boxes) for boxes in total_instances['FN'].values()) + sum(
            len(boxes) for boxes in total_instances['FP'].values())

        if total_instance_count == 0:
            self.val += 0.0
        else:
            self.val += (intersection_fp + intersection_fn) / \
                total_instance_count

        self.num_frames += 1

    def compute(self) -> float:
        """
        Computes the final CovOD metric from the accumulated values.

        The final metric is the complement of the average common error rate:
        $$\text{CovOD} = 1 - \frac{\sum (\text{Per-Frame Common Error Ratio})}{\text{Total Number of Frames}}$$

        Returns:
            float: The final CovOD metric (system reliability against common failures). 
                   Returns 1.0 if no frames have been processed.
        """
        if self.num_frames == 0:
            return 1.0
        return 1 - self.val / self.num_frames
