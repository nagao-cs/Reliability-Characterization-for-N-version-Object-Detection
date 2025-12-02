import os
from pathlib import Path
from Integrator.affirmative_integrator import AffirmativeIntegrator
from Integrator.majority_integrator import MajorityIntegrator
from Integrator.unanimous_integrator import UnanimousIntegrator
from Metrics.Cov import Cov
from Metrics.Cer import Cer
from Metrics.CovOD import CovOD
from Metrics.CerOD import CerOD
from Metrics.mAP import mAP

from processor import dataset

import pandas as pd
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Any


def results_table(results_by_version: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    Formats and prints evaluation results accumulated across multiple versions 
    into a readable Pandas DataFrame.

    Args:
        results_by_version (Dict[int, Dict[str, float]]): A dictionary where keys 
                                                          are the number of versions used 
                                                          and values are dictionaries of computed metrics.

    Returns:
        pd.DataFrame: The DataFrame containing the structured results.
    """
    df = pd.DataFrame(results_by_version).T
    df.index.name = 'Num Versions'

    print(df.to_string())
    print()

    return df


def evaluate_with_model_versions(datasets: 'dataset.Dataset', models: List[str], iou_th: float, num_versions: int) -> Dict[str, float]:
    """
    Executes a multi-version evaluation using a specified number of unique detection models.

    A subset of models is selected based on `num_versions`. A new Dataset object is 
    created to restrict the evaluation to only those model versions.

    Args:
        datasets (dataset.Dataset): The full Dataset object containing configuration 
                                    (used to extract gt_dir and analysis settings).
        models (List[str]): List of all available model names (e.g., ["yolov8n", "yolov11n", ...]).
        iou_th (float): IoU threshold for evaluation.
        num_versions (int): The number of models to include in the current evaluation subset.

    Returns:
        Dict[str, float]: The computed evaluation metrics for the selected model subset.
    """
    selected_models = models[:num_versions]
    det_dirs = [
        f"./dataset/detectionresult/labels/{model}/front/" for model in selected_models
    ]
    # Create a new Dataset instance pointing to the subset of detection directories
    datasets_subset = dataset.Dataset(
        gt_dir=datasets.loader.gt_dir,
        det_dirs=det_dirs,
        iou_th=iou_th,
    )

    return get_result(datasets_subset)


def evaluate_with_camera_versions(datasets: 'dataset.Dataset', cameras: List[str], iou_th: float, num_versions: int) -> Dict[str, float]:
    """
    Executes a multi-version evaluation using a specified number of different camera views (sensors), 
    while keeping the underlying detection model constant (assumed 'yolov8n').

    Args:
        datasets (dataset.Dataset): The full Dataset object containing configuration.
        cameras (List[str]): List of all available camera names (e.g., ["front", "left_1", ...]).
        iou_th (float): IoU threshold for evaluation.
        num_versions (int): The number of camera views (versions) to include in the current evaluation subset.

    Returns:
        Dict[str, float]: The computed evaluation metrics for the selected camera subset.
    """
    selected_cameras = cameras[:num_versions]
    # The model name is hardcoded as 'yolov8n'
    det_dirs = [
        f"./dataset/detectionresult/labels/yolov8n/{camera}/" for camera in selected_cameras
    ]
    # Create a new Dataset instance pointing to the subset of detection directories
    datasets_subset = dataset.Dataset(
        gt_dir=datasets.loader.gt_dir,
        det_dirs=det_dirs,
        iou_th=iou_th,
    )

    return get_result(datasets_subset)


def get_result(datasets: 'dataset.Dataset') -> Dict[str, float]:
    """
    Performs the core frame-by-frame analysis, ensemble integration, and metric calculation 
    for a given multi-version dataset configuration.

    It initializes all metrics and integrators, iterates over all frames, and updates 
    all metric accumulators based on the analysis results and integrated detections.

    Args:
        datasets (dataset.Dataset): A Dataset object initialized with GT and multiple 
                                    detection directories.

    Returns:
        Dict[str, float]: A dictionary containing the final computed values for all 
                          metrics (Cov, Cer, CovOD, CerOD, and mAP for all three integrators).
    """
    affirmative_integrator = AffirmativeIntegrator(iou_th=datasets.iou_th)
    majority_integrator = MajorityIntegrator(iou_th=datasets.iou_th)
    unanimous_integrator = UnanimousIntegrator(iou_th=datasets.iou_th)
    cov = Cov()
    cer = Cer()
    covod = CovOD()
    cerod = CerOD()
    mAP_affirmative = mAP(iou_th=datasets.iou_th)
    mAP_majority = mAP(iou_th=datasets.iou_th)
    mAP_unanimous = mAP(iou_th=datasets.iou_th)

    # Loop through each frame in the dataset
    for frame_idx, gt, dets in datasets.loader.iter_frame():
        # 1. Ensemble Integration
        affirmative = affirmative_integrator(dets)
        majority = majority_integrator(dets)
        unanimous = unanimous_integrator(dets)

        # 2. Multi-Version Analysis
        analysis_results = datasets.analyzer.analyze_frame(gt, dets)

        # Extract analysis results for metrics
        intersection_errors = analysis_results['intersection_errors']
        union_errors = analysis_results['union_errors']
        total_instances = analysis_results['total_instances']
        is_correct_list = analysis_results['is_correct_list']

        # 3. Update Metrics
        cov.update(is_correct_list)
        cer.update(is_correct_list)
        covod.update(intersection_errors, total_instances)
        cerod.update(union_errors, total_instances)

        # 4. Update mAP for Integrated Results
        mAP_affirmative.update(gt, affirmative)
        mAP_majority.update(gt, majority)
        mAP_unanimous.update(gt, unanimous)

    return {
        'cov': cov.compute(),
        'cer': cer.compute(),
        'covod': covod.compute(),
        'cerod': cerod.compute(),
        'mAP_affirmative': mAP_affirmative.compute(),
        'mAP_majority': mAP_majority.compute(),
        'mAP_unanimous': mAP_unanimous.compute()
    }


def explore_model_combinations(gt_dir: str, models: List[str], num_version: int, iou_th: float) -> pd.DataFrame:
    """
    Evaluates all possible combinations of a specified size from the available model list.

    Args:
        gt_dir (str): Path to the ground truth directory.
        models (List[str]): List of all available model names.
        num_version (int): The fixed size of the model combinations to test.
        iou_th (float): IoU threshold for evaluation.

    Returns:
        pd.DataFrame: A DataFrame where the index is the model combination tuple, 
                      and columns are the resulting metric values.
    """
    models_combination = combinations(models, num_version)
    result = dict()

    for comb in tqdm(models_combination):
        # Create a specific Dataset instance for the current combination
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


def explore_camera_combinations(gt_dir: str, cameras: List[str], num_version: int, iou_th: float) -> pd.DataFrame:
    """
    Evaluates all possible combinations of a specified size from the available camera list, 
    assuming a fixed detection model ('yolov8n').

    Args:
        gt_dir (str): Path to the ground truth directory.
        cameras (List[str]): List of all available camera names.
        num_version (int): The fixed size of the camera combinations to test.
        iou_th (float): IoU threshold for evaluation.

    Returns:
        pd.DataFrame: A DataFrame where the index is the camera combination tuple, 
                      and columns are the resulting metric values.
    """
    camera_combination = combinations(cameras, num_version)
    result = dict()

    for comb in tqdm(camera_combination):
        # Create a specific Dataset instance for the current camera combination
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
