SIZE_THRESHOLD = 500
CONF_THRESHOLD = 0
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
    for class_id in gt.keys():
        gt_boxes = gt[class_id]
        det_boxes = det.get(class_id, [])

        if class_id not in det_results['TP']:
            det_results['TP'][class_id] = list()
        if class_id not in det_results['FN']:
            det_results['FN'][class_id] = list()
        if class_id not in det_results['FP']:
            det_results['FP'][class_id] = list()

        used_gt = set()

        for det_box in det_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                if i in used_gt:
                    continue
                iou = compute_iou(gt_box, det_box)
                if iou >= iou_th and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_gt_idx >= 0:
                det_results['TP'][class_id].append(det_box)
                used_gt.add(best_gt_idx)
            else:
                det_results['FP'][class_id].append(det_box)

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
