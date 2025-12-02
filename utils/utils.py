from typing import List, Tuple
import numpy as np
import cv2

SIZE_THRESHOLD = 500  # バウンディングボックスの最小サイズ
CONF_THRESHOLD = 0  # 信頼度の閾値
IM_WIDTH = 800
IM_HEIGHT = 600
class_Map = {
    0: 0,  # pedestrian
    1: 2,  # bicycle
    2: 2,  # motorcycle
    3: 2,  # car
    5: 2,  # bus
    7: 2,  # truck
    9: 9,  # traffic light
    11: 11,  # stop sign
}


def classify(gt, det, iou_th):
    det_results = {'TP': dict(), 'FP': dict(), 'FN': dict()}
    # クラスごとに処理
    for class_id in gt.keys():
        gt_boxes = gt[class_id]
        det_boxes = det.get(class_id, [])

        if class_id not in det_results['TP']:
            det_results['TP'][class_id] = list()
        if class_id not in det_results['FN']:
            det_results['FN'][class_id] = list()
        if class_id not in det_results['FP']:
            det_results['FP'][class_id] = list()

        # GTとDetectionのマッチング
        used_gt = set()  # マッチ済みのGTを記録

        for det_box in det_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            # 未使用のGTと最もIoUが高いものを探す
            for i, gt_box in enumerate(gt_boxes):
                if i in used_gt:
                    continue
                iou = compute_iou(gt_box, det_box)
                if iou >= iou_th and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            # マッチング結果に基づいて分類
            if best_gt_idx >= 0:
                det_results['TP'][class_id].append(det_box)
                used_gt.add(best_gt_idx)
            else:
                det_results['FP'][class_id].append(det_box)

        # 未使用のGTをFNとして追加
        for i, gt_box in enumerate(gt_boxes):
            if i not in used_gt:
                det_results['FN'][class_id].append(gt_box)
    return det_results


def compute_iou(box1, box2):
    ax_center, ay_center, a_width, a_height, _ = box1
    bx_center, by_center, b_width, b_height, _ = box2
    axmin = ax_center - a_width / 2
    axmax = ax_center + a_width / 2
    aymin = ay_center - a_height / 2
    aymax = ay_center + a_height / 2
    bxmin = bx_center - b_width / 2
    bxmax = bx_center + b_width / 2
    bymin = by_center - b_height / 2
    bymax = by_center + b_height / 2
    area_a = (axmax - axmin) * (aymax - aymin)
    area_b = (bxmax - bxmin) * (bymax - bymin)

    abxmin = max(axmin, bxmin)
    abxmax = min(axmax, bxmax)
    abymin = max(aymin, bymin)
    abymax = min(aymax, bymax)
    intersection = max(0, abxmax - abxmin) * max(0, abymax - abymin)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0


def calibrate_affine(pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> np.ndarray:
    """
    対応点ペアから 2x3 アフィン行列を推定する。
    pairs: [ ((x_other_norm,y_other_norm),(x_center_norm,y_center_norm)), ... ]
    戻り: 2x3 行列 M  (OpenCV形式)。呼び出し側で None チェックを。
    """
    if len(pairs) < 3:
        return None
    src = np.array([p[0] for p in pairs], dtype=np.float32)
    dst = np.array([p[1] for p in pairs], dtype=np.float32)
    # スケーリング：normalize to pixels for安定化
    src_px = src * np.array([IM_WIDTH, IM_HEIGHT], dtype=np.float32)
    dst_px = dst * np.array([IM_WIDTH, IM_HEIGHT], dtype=np.float32)
    M, inliers = cv2.estimateAffine2D(
        src_px, dst_px, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M  # M is 2x3 or None


def apply_affine_to_bbox(bbox: Tuple[float, float, float, float, float],
                         M: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    アフィン行列 M(2x3, pixel単位) を bbox 中心に適用して center に移す。
    """
    if M is None:
        return bbox
    x, y, w, h, conf = bbox
    pt = np.array([[x * IM_WIDTH, y * IM_HEIGHT, 1.0]],
                  dtype=np.float32).T  # 3x1
    res = M.dot(pt).flatten()
    x2 = np.clip(res[0] / IM_WIDTH, 0.0, 1.0)
    y2 = np.clip(res[1] / IM_HEIGHT, 0.0, 1.0)
    return (x2, y2, w, h, conf)
