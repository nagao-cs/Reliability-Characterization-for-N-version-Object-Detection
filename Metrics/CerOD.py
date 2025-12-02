from typing import Dict, List, Any


class CerOD:
    """
    Calculates the Coverage Metric for Object Detection (CovOD) based on the 
    union of errors across multiple model versions.

    The metric is calculated on a per-frame basis by finding the ratio of combined 
    errors (FP Union + FN Union) to the total number of instances (TP Intersection + FP Union + FN Union), 
    and then averaging this ratio across all frames.

    Attributes:
        val (float): The accumulated sum of the per-frame error ratios.
        num_frames (int): The total number of frames processed.
    """

    def __init__(self):
        """
        Initializes the CerOD counters.
        """
        self.val = 0.0
        self.num_frames = 0

    def update(self, union_errors: Dict[str, dict], total_instances: Dict[str, dict]) -> None:
        """
        Updates the accumulated error ratio (`self.val`) for a single frame.

        The per-frame error ratio is calculated as:
        $$
        \frac{\text{Union Errors (FP + FN)}}{\text{Total Instances (TP}_{\text{int}} + \text{FP}_{\text{uni}} + \text{FN}_{\text{uni}})}
        $$

        Args:
            union_errors (Dict[str, dict]): Dictionary containing the **Union** of FP and FN boxes 
                                            across all model versions (i.e., errors observed in ANY version).
                                            Format: `{'FP': {class_id: [boxes]}, 'FN': {class_id: [boxes]}}`.
            total_instances (Dict[str, dict]): Dictionary containing the total set of instances considered 
                                               for the denominator, which is typically: TP as Intersection, 
                                               and FP/FN as Union.
                                               Format: `{'TP': {Intersection boxes}, 'FP': {Union boxes}, 'FN': {Union boxes}}`.
        """
        # Sum the counts of False Positives in the Union set
        union_fp = sum(
            len(boxes) for boxes in union_errors['FP'].values())
        # Sum the counts of False Negatives in the Union set
        union_fn = sum(
            len(boxes) for boxes in union_errors['FN'].values())

        # Total denominator: TP (Intersection) + FN (Union) + FP (Union)
        total_instance_count = sum(len(boxes) for boxes in total_instances['TP'].values()) + sum(
            len(boxes) for boxes in total_instances['FN'].values()) + sum(
            len(boxes) for boxes in total_instances['FP'].values())

        if total_instance_count == 0:
            # If no instances exist in the frame, the error ratio is 0
            self.val += 0.0
        else:
            # Accumulate the per-frame error ratio
            self.val += (union_fp + union_fn) / \
                total_instance_count

        self.num_frames += 1

    def compute(self) -> float:
        """
        Computes the final CerOD (Coverage) metric from the accumulated values.

        The final metric is the complement of the average error rate:
        $$\text{Coverage/Reliability} = 1 - \frac{\sum (\text{Per-Frame Error Ratio})}{\text{Total Number of Frames}}$$

        Returns:
            float: The final Coverage metric. Returns 1.0 if no frames have been processed.
        """
        if self.num_frames == 0:
            return 1.0
        # The result is 1 - Average Error Ratio
        return 1 - self.val / self.num_frames
