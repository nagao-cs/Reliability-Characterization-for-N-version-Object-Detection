class Cer:
    def __init__(self):
        self.val = 0.0
        self.num_frames = 0

    def update(self, is_correct_list):
        num_version = len(is_correct_list)

        if not all(is_correct_list):
            self.val += 1.0

        self.num_frames += 1

    def compute(self):
        if self.num_frames == 0:
            return 1.0
        return 1 - self.val / self.num_frames
