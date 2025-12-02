from .AbsIntegrator import AbstractIntegrator
from typing import Dict, List, Any
from math import ceil


class MajorityIntegrator(AbstractIntegrator):
    def __init__(self, iou_th: float = 0.5):
        self.iou_th = iou_th

    def integrate(self, detections: Dict[int, Dict[int, List[tuple]]]) -> Dict[int, List[tuple]]:
        """
        多数決（Affirmative）で検出結果を統合する。

        Args:
            detections: {version: {class_id: [(x_center,y_center,width,height,confidence), ...], ...}, ...}

        Returns:
            integrated: {class_id: [(x_center,y_center,width,height,confidence), ...], ...}
        """
        integrated = dict()
        majority_threshold = ceil(len(detections) / 2)
        IOU_THRESHOLD = self.iou_th

        # クラスごとに統合
        all_class_ids = set()
        for v_dets in detections.values():
            all_class_ids.update(v_dets.keys())

        for class_id in all_class_ids:
            # 全バージョンから当クラスの検出を取り出してフラット化
            all_dets = []
            for version_id, v_dets in detections.items():
                boxes = v_dets.get(class_id, [])
                for b in boxes:
                    # 入力がタプルの想定: (x_center,y_center,width,height,confidence)
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

            # グルーピング（同一バージョンはグループ内に複数持たないようにする）
            for i in range(len(all_dets)):
                if is_processed[i]:
                    continue
                base = all_dets[i]
                group = [base]
                is_processed[i] = True

                # 他の未処理検出をグループに追加（任意のメンバーとIOU閾値以上なら追加）
                for j in range(i + 1, len(all_dets)):
                    if is_processed[j]:
                        continue
                    cand = all_dets[j]
                    if cand['version_id'] == base['version_id']:
                        continue
                    # クラスは同じ前提なのでチェック不要
                    matched = False
                    for member in group:
                        if self._iou(member, cand) >= IOU_THRESHOLD:
                            matched = True
                            break
                    if matched:
                        group.append(cand)
                        is_processed[j] = True

                groups.append(group)

            # 各グループについて、多数決（不同バージョン数）を満たしたら平均化して出力
            integrated[class_id] = []
            for group in groups:
                unique_versions = set(d['version_id'] for d in group)
                if len(unique_versions) >= majority_threshold:
                    # 平均を取る
                    n = len(group)
                    avg_x = sum(d['x_center'] for d in group) / n
                    avg_y = sum(d['y_center'] for d in group) / n
                    avg_w = sum(d['width'] for d in group) / n
                    avg_h = sum(d['height'] for d in group) / n
                    avg_conf = sum(d['confidence'] for d in group) / n
                    integrated[class_id].append(
                        (avg_x, avg_y, avg_w, avg_h, avg_conf))

            # 空なら削除
            if not integrated[class_id]:
                del integrated[class_id]

        return integrated

    def _iou(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        x1_min = a['x_center'] - a['width'] / 2
        y1_min = a['y_center'] - a['height'] / 2
        x1_max = a['x_center'] + a['width'] / 2
        y1_max = a['y_center'] + a['height'] / 2

        x2_min = b['x_center'] - b['width'] / 2
        y2_min = b['y_center'] - b['height'] / 2
        x2_max = b['x_center'] + b['width'] / 2
        y2_max = b['y_center'] + b['height'] / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
        area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
        union = area1 + area2 - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def __call__(self, dets):
        return self.integrate(dets)
