class Cov:
    """
    Coverage metric: percentage of ground truth objects correctly detected.

    Measures recall on a per-class basis. A detection is counted as correct
    if IoU with some ground truth box exceeds the threshold.

    Formula:
        Cov = 1 - total_error_image / num_images
    """

    def __init__(self):
        self.val = 0.0
        self.num_frames = 0

    def update(self, is_correct_list):
        """
        Update coverage metric with results from one frame.

        Args:
            is_correct_list (list of bool): Per-model-detection correctness flags.
        """
        if not any(is_correct_list):
            self.val += 1.0

        self.num_frames += 1

    def compute(self):
        """
        Compute coverage.

        Returns:
            Coverage value in range [0, 1].
        """
        if self.num_frames == 0:
            return 1.0
        return 1 - self.val / self.num_frames
