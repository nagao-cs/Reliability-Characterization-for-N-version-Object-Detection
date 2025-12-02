from typing import List


class Cer:
    """
    This value represents the proportion of frames where ALL models achieved 
    perfect detection simultaneously.

    Attributes:
        val (float): The accumulated count of error frames.
        num_frames (int): The total number of frames processed.
    """

    def __init__(self):
        """
        Initializes the Certainty counters.
        """
        self.val = 0.0
        self.num_frames = 0

    def update(self, is_correct_list: List[bool]):
        """
        Updates the frame counters based on the detection results of a single frame.

        If `not all(is_correct_list)` is True, meaning at least one model version 
        failed (FP or FN), the frame is counted as an error frame.

        Args:
            is_correct_list (List[bool]): A list of boolean flags, where each element 
                                          indicates if a model version achieved **perfect 
                                          detection (zero FP and zero FN)** for this frame.
                                          (e.g., [True, True, False] results in an error count).
        """
        num_version = len(is_correct_list)

        if not all(is_correct_list):
            self.val += 1.0

        self.num_frames += 1

    def compute(self) -> float:
        """
        Computes the final Certainty metric from the accumulated frame counts.

        Returns:
            float: The Certainty value, calculated as $1 - \text{CER}$. Returns 1.0 if no frames have been processed.
        """
        if self.num_frames == 0:
            return 1.0
        # Certainty = 1 - Certainty Error Rate (CER)
        return 1 - self.val / self.num_frames
