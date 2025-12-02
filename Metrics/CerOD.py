class CerOD:
    def __init__(self):
        self.val = 0.0
        self.num_frames = 0

    def update(self, union_errors: dict, total_instances: dict) -> None:
        union_fp = sum(
            len(boxes) for boxes in union_errors['FP'].values())
        union_fn = sum(
            len(boxes) for boxes in union_errors['FN'].values())
        total_instance_count = sum(len(boxes) for boxes in total_instances['TP'].values()) + sum(
            len(boxes) for boxes in total_instances['FN'].values()) + sum(
            len(boxes) for boxes in total_instances['FP'].values())
        if total_instance_count == 0:
            self.val += 0.0
        else:
            self.val += (union_fp + union_fn) / \
                total_instance_count
        self.num_frames += 1

    def compute(self):
        if self.num_frames == 0:
            return 1.0
        return 1 - self.val / self.num_frames
