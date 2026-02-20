#!/usr/bin/env python3
"""
YOLO CPU Inference — Evaluate all YOLO model variants on CPU.

Loads each YOLO variant (full .pt, ONNX FP32, ONNX INT8, TFLite FP32)
and runs inference on test sets. Records mAP50, mAP50-95, precision, recall, F1.

Usage:
    python run_cpu_inference_yolo.py --models_dir ./models/yolo --datasets_dir ./datasets --results_dir ./results
"""

import argparse
import logging
import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO CPU Inference Benchmark')
    parser.add_argument('--models_dir', type=str, default='./models/yolo',
                        help='Directory containing YOLO model subdirs (full/, onnx/, quantized/, tflite/)')
    parser.add_argument('--datasets_dir', type=str, default='./datasets',
                        help='Directory containing test datasets with data.yaml')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results CSV')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    return parser.parse_args()


def discover_models(models_dir: str) -> list:
    """Find all YOLO model files across format subdirs."""
    models_root = Path(models_dir)
    discovered = []

    format_dirs = {
        'full':      ('*.pt',    'pytorch'),
        'onnx':      ('*.onnx',  'onnx'),
        'quantized': ('*.onnx',  'onnx_int8'),
        'tflite':    ('*.tflite','tflite'),
    }

    for subdir, (glob_pat, fmt_name) in format_dirs.items():
        fmt_dir = models_root / subdir
        if not fmt_dir.exists():
            continue
        for model_path in sorted(fmt_dir.glob(glob_pat)):
            stem = model_path.stem.replace('_best', '')
            parts = stem.split('_')
            if len(parts) >= 2:
                dataset = parts[0] if parts[0] in ['NEU-DET', 'DAGM', 'KSDD2'] else '_'.join(parts[:-1])
                model_variant = parts[-1]
            else:
                dataset, model_variant = 'unknown', stem

            # Fix NEU-DET naming (contains hyphen)
            if 'NEU-DET' in stem:
                dataset = 'NEU-DET'
                model_variant = stem.replace('NEU-DET_', '').replace('_best', '').replace('_int8', '')

            discovered.append({
                'path': str(model_path),
                'dataset': dataset,
                'model': model_variant,
                'format': fmt_name,
                'stem': stem,
            })

    logger.info(f'Discovered {len(discovered)} YOLO models in {models_dir}')
    for m in discovered:
        logger.info(f'  [{m["format"]}] {m["dataset"]}/{m["model"]} -> {m["path"]}')
    return discovered


def resolve_data_yaml(data_yaml: str, datasets_dir: str, dataset: str) -> str:
    """Create a temporary data.yaml with absolute path so Ultralytics finds the images.

    Ultralytics resolves the 'path' field relative to CWD, not the YAML location.
    This function reads the original data.yaml, replaces 'path' with the absolute
    dataset directory, and writes it to a temp file.
    """
    dataset_dir = str(Path(datasets_dir).resolve() / dataset)

    # Read original yaml content
    with open(data_yaml, 'r') as f:
        lines = f.readlines()

    # Replace or add the path line with the absolute dataset directory
    new_lines = []
    path_found = False
    for line in lines:
        if line.strip().startswith('path:'):
            new_lines.append(f'path: {dataset_dir}\n')
            path_found = True
        else:
            new_lines.append(line)
    if not path_found:
        new_lines.insert(0, f'path: {dataset_dir}\n')

    # Write to a temp file in the same directory (so relative refs still work)
    tmp_yaml = os.path.join(os.path.dirname(data_yaml), f'_tmp_data_{dataset}.yaml')
    with open(tmp_yaml, 'w') as f:
        f.writelines(new_lines)

    logger.info(f'  Resolved data.yaml path -> {dataset_dir}')
    return tmp_yaml


def run_yolo_inference(model_path: str, data_yaml: str, fmt: str,
                       conf: float, iou: float, imgsz: int) -> dict:
    """Run YOLO inference with appropriate backend. Returns metrics dict."""
    try:
        from ultralytics import YOLO

        if fmt in ('pytorch', 'onnx', 'onnx_int8'):
            # All can be loaded via ultralytics (it wraps ONNX runtime for .onnx)
            model = YOLO(model_path)
            metrics = model.val(
                data=data_yaml,
                split='test',
                device='cpu',
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            box = metrics.box
            mp = float(box.mp)
            mr = float(box.mr)
            return {
                'map50':     round(float(box.map50), 4),
                'map50_95':  round(float(box.map), 4),
                'precision': round(mp, 4),
                'recall':    round(mr, 4),
                'f1':        round(2 * mp * mr / (mp + mr + 1e-9), 4),
                'status': 'ok',
            }
        elif fmt == 'tflite':
            # For TFLite we can still use ultralytics if model ends in .tflite
            model = YOLO(model_path)
            metrics = model.val(data=data_yaml, split='test', device='cpu',
                                imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            box = metrics.box
            mp = float(box.mp)
            mr = float(box.mr)
            return {
                'map50':     round(float(box.map50), 4),
                'map50_95':  round(float(box.map), 4),
                'precision': round(mp, 4),
                'recall':    round(mr, 4),
                'f1':        round(2 * mp * mr / (mp + mr + 1e-9), 4),
                'status': 'ok',
            }
        else:
            return {'status': 'unsupported_format', 'error': f'Unknown format: {fmt}',
                    'map50': None, 'map50_95': None, 'precision': None, 'recall': None, 'f1': None}
    except Exception as e:
        logger.warning(f'Inference failed for {model_path}: {e}')
        return {'status': 'error', 'error': str(e),
                'map50': None, 'map50_95': None, 'precision': None, 'recall': None, 'f1': None}


def main():
    args = parse_args()
    logger.info('YOLO CPU Inference — Starting')
    logger.info(f'Models dir: {args.models_dir}')
    logger.info(f'Datasets dir: {args.datasets_dir}')
    logger.info(f'Results dir: {args.results_dir}')

    os.makedirs(args.results_dir + '/tables', exist_ok=True)

    # Discover models
    models = discover_models(args.models_dir)
    if not models:
        logger.error('No YOLO models found. Check --models_dir path.')
        sys.exit(1)

    # Pre-resolve data.yaml for each dataset (absolute paths for Ultralytics)
    resolved_yamls = {}
    tmp_yamls = []
    for ds_name in ['NEU-DET', 'DAGM', 'KSDD2']:
        orig_yaml = os.path.join(args.datasets_dir, ds_name, 'data.yaml')
        if os.path.exists(orig_yaml):
            tmp = resolve_data_yaml(orig_yaml, args.datasets_dir, ds_name)
            resolved_yamls[ds_name] = tmp
            tmp_yamls.append(tmp)

    results = []
    try:
        for entry in tqdm(models, desc='YOLO CPU Inference'):
            dataset = entry['dataset']
            data_yaml = resolved_yamls.get(dataset)

            if not data_yaml or not os.path.exists(data_yaml):
                logger.warning(f'data.yaml not found for {dataset}, skipping')
                continue

            logger.info(f'Running: [{entry["format"]}] {entry["dataset"]}/{entry["model"]}')
            t0 = time.time()
            metrics = run_yolo_inference(
                model_path=entry['path'],
                data_yaml=data_yaml,
                fmt=entry['format'],
                conf=args.conf_threshold,
                iou=args.iou_threshold,
                imgsz=args.imgsz,
            )
            elapsed = time.time() - t0

            row = {
                'dataset':     entry['dataset'],
                'model':       entry['model'],
                'format':      entry['format'],
                'model_path':  entry['path'],
                'eval_time_s': round(elapsed, 1),
            }
            row.update(metrics)
            results.append(row)
            logger.info(f'  mAP50={metrics.get("map50")} P={metrics.get("precision")} R={metrics.get("recall")}')
    finally:
        # Clean up temporary data.yaml files
        for tmp in tmp_yamls:
            try:
                os.remove(tmp)
            except OSError:
                pass

    # Save results
    df = pd.DataFrame(results)
    out_path = os.path.join(args.results_dir, 'tables', 'yolo_cpu_inference.csv')
    df.to_csv(out_path, index=False)
    logger.info(f'Results saved to: {out_path}')

    # Print summary
    print('\n' + '='*70)
    print('YOLO CPU Inference Summary')
    print('='*70)
    if not df.empty:
        for fmt in df['format'].unique():
            sub = df[df['format'] == fmt]
            print(f'\nFormat: {fmt}')
            print(sub[['dataset', 'model', 'map50', 'map50_95', 'precision', 'recall']].to_string(index=False))
    print(f'\nFull results: {out_path}')


if __name__ == '__main__':
    main()
