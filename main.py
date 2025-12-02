import os
from pathlib import Path
from Integrator.affirmative_integrator import AffirmativeIntegrator
from Integrator.majority_integrator import MajorityIntegrator
from Integrator.unanimous_integrator import UnanimousIntegrator
from Metrics.Cov import Cov
from Metrics.Cer import Cer
from Metrics.CovOD import CovOD
from Metrics.CerOD import CerOD
from Metrics.accuracy import Accuracy
from Metrics.mAP import mAP

from processor import dataset

import pandas as pd
from itertools import combinations
from tqdm import tqdm


def results_table(results_by_version):
    """複数バージョンの結果をテーブルで表示"""
    df = pd.DataFrame(results_by_version).T
    df.index.name = 'Num Versions'

    print(df.to_string())
    print()

    return df


def evaluate_with_model_versions(datasets, models, iou_th, num_versions):
    """指定されたバージョン数で評価を実行"""
    selected_models = models[:num_versions]
    det_dirs = [
        f"./dataset/detectionresult/labels/{model}/front/" for model in selected_models
    ]
    datasets_subset = dataset.Dataset(
        gt_dir=datasets.loader.gt_dir,
        det_dirs=det_dirs,
        iou_th=iou_th,
    )

    return get_result(datasets_subset)


def evaluate_with_camera_versions(datasets, cameras, iou_th, num_versions):
    """指定されたバージョン数で評価を実行"""
    selected_cameras = cameras[:num_versions]
    det_dirs = [
        f"./dataset/detectionresult/labels/yolov8n/{camera}/" for camera in selected_cameras
    ]
    datasets_subset = dataset.Dataset(
        gt_dir=datasets.loader.gt_dir,
        det_dirs=det_dirs,
        iou_th=iou_th,
    )

    return get_result(datasets_subset)


def get_result(datasets):
    affirmative_integrator = AffirmativeIntegrator(iou_th=datasets.iou_th)
    majority_integrator = MajorityIntegrator(iou_th=datasets.iou_th)
    unanimous_integrator = UnanimousIntegrator(iou_th=datasets.iou_th)
    cov = Cov()
    cer = Cer()
    covod = CovOD()
    cerod = CerOD()
    acc_affirmative = Accuracy(iou_th=datasets.iou_th)
    acc_majority = Accuracy(iou_th=datasets.iou_th)
    acc_unanimous = Accuracy(iou_th=datasets.iou_th)
    mAP_affirmative = mAP(iou_th=datasets.iou_th)
    mAP_majority = mAP(iou_th=datasets.iou_th)
    mAP_unanimous = mAP(iou_th=datasets.iou_th)

    for frame_idx, gt, dets in datasets.loader.iter_frame():
        affirmative = affirmative_integrator(dets)
        majority = majority_integrator(dets)
        unanimous = unanimous_integrator(dets)
        analysis_results = datasets.analyzer.analyze_frame(gt, dets)
        classified_results = analysis_results['classified_results']
        intersection_errors = analysis_results['intersection_errors']
        union_errors = analysis_results['union_errors']
        total_instances = analysis_results['total_instances']
        is_correct_list = analysis_results['is_correct_list']
        cov.update(is_correct_list)
        cer.update(is_correct_list)
        covod.update(intersection_errors, total_instances)
        cerod.update(union_errors, total_instances)
        acc_affirmative.update(gt, affirmative)
        acc_majority.update(gt, majority)
        acc_unanimous.update(gt, unanimous)
        mAP_affirmative.update(gt, affirmative)
        mAP_majority.update(gt, majority)
        mAP_unanimous.update(gt, unanimous)

    return {
        'cov': cov.compute(),
        'cer': cer.compute(),
        'covod': covod.compute(),
        'cerod': cerod.compute(),
        'acc_affirmative': acc_affirmative.compute(),
        'acc_majority': acc_majority.compute(),
        'acc_unanimous': acc_unanimous.compute(),
        'mAP_affirmative': mAP_affirmative.compute(),
        'mAP_majority': mAP_majority.compute(),
        'mAP_unanimous': mAP_unanimous.compute()
    }


def explore_model_combinations(gt_dir, models, num_version, iou_th):
    models_combination = combinations(models, num_version)
    result = dict()
    for comb in tqdm(models_combination):
        datasets_subset = dataset.Dataset(
            gt_dir=gt_dir,
            det_dirs=[
                f"./dataset/detectionresult/labels/{model}/front/" for model in comb],
            iou_th=iou_th
        )
        res = get_result(datasets_subset)
        result[comb] = res

    df = pd.DataFrame(result).T
    df.index.name = 'Model Combination'

    return df


def explore_camera_combinations(gt_dir, cameras, num_version, iou_th):
    camera_combination = combinations(cameras, num_version)
    result = dict()
    for comb in tqdm(camera_combination):
        datasets_subset = dataset.Dataset(
            gt_dir=gt_dir,
            det_dirs=[
                f"./dataset/detectionresult/labels/yolov8n/{camera}/" for camera in comb],
            iou_th=iou_th
        )
        res = get_result(datasets_subset)
        result[comb] = res

    df = pd.DataFrame(result).T
    df.index.name = 'Model Combination'

    return df


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(
        description="Evaluate object detection results")
    argparser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        choices=["yolov8n", "yolov11n",
                 "yolov5n", "rtdetr", "yolov8l", "ssd"],
    )
    argparser.add_argument(
        "--iou_th",
        type=float,
        default=0.5,
        help="IOU threshold for evaluation",
    )

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    args = argparser.parse_args()
    models = args.models
    print(args)

    os.makedirs("./result", exist_ok=True)

    gt_dir = "./dataset/labels/Town03/front/"

    max_versions = len(models)
    datasets = dataset.Dataset(
        gt_dir=gt_dir,
        det_dirs=[
            f"./dataset/detectionresult/labels/{model}/front/" for model in models],
        iou_th=args.iou_th
    )
    result_by_version = {}
    for num_ver in range(1, max_versions + 1):
        #     # print(f"\nEvaluating with {num_ver} version(s)...")
        result_by_version[num_ver] = evaluate_with_model_versions(
            datasets, models, args.iou_th, num_ver
        )
    df = results_table(result_by_version)
    df.to_csv(f"./result/models_evaluation_results.csv",
              index_label='Num Versions')

    cameras = ["front", "left_1", "right_1", "left_2", "right_2"]
    max_versions = len(cameras)
    datasets = dataset.Dataset(
        gt_dir=gt_dir,
        det_dirs=[
            f"./dataset/detectionresult/labels/yolov8n/{camera}/" for camera in cameras],
        iou_th=args.iou_th
    )
    result_by_version = {}
    for num_ver in range(1, max_versions + 1):
        # print(f"\nEvaluating with {num_ver} version(s)...")
        result_by_version[num_ver] = evaluate_with_camera_versions(
            datasets, cameras, args.iou_th, num_ver
        )
    df = results_table(result_by_version)
    df.to_csv(f"./result/cameras_evaluation_results.csv",
              index_label='Num Versions')

    num_version = 3
    df_model_comb = explore_model_combinations(
        gt_dir, models, num_version, args.iou_th)
    print(df_model_comb)
    df_model_comb.to_csv(
        f"./result/model_combinations_evaluation_results.csv", index_label='Model Combination')

    # cameras = ["front", "left_1", "right_1", "left_2", "right_2"]
    # df_camera_comb = explore_camera_combinations(
    #     gt_dir, cameras, num_version, args.iou_th)
    # print(df_camera_comb)
    # df_camera_comb.to_csv(
    #     f"./result/camera_combinations_evaluation_results.csv", index_label='Camera Combination')
