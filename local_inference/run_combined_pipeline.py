#!/usr/bin/env python3
"""
Combined Pipeline — YOLO detection -> crop -> CNN classification, end-to-end.

Runs the full two-stage pipeline: YOLO detects defect regions, crops them,
then CNN classifies each crop. Records end-to-end latency and throughput.

Usage:
    python run_combined_pipeline.py --dataset NEU-DET --format onnx_fp32
    python run_combined_pipeline.py --dataset DAGM --yolo_variant yolov8s --cnn_variant resnet50
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Combined YOLO+CNN Pipeline Benchmark')
    parser.add_argument('--yolo_dir',     type=str, default='./models/yolo')
    parser.add_argument('--cnn_dir',      type=str, default='./models/cnn')
    parser.add_argument('--datasets_dir', type=str, default='./datasets')
    parser.add_argument('--results_dir',  type=str, default='./results')
    parser.add_argument('--dataset',      type=str, default='NEU-DET',
                        choices=['NEU-DET', 'DAGM', 'KSDD2'])
    parser.add_argument('--yolo_variant', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s'])
    parser.add_argument('--cnn_variant',  type=str, default='efficientnet_b0',
                        choices=['resnet50', 'efficientnet_b0', 'mobilenet_v3_large'])
    parser.add_argument('--format',       type=str, default='onnx_fp32',
                        choices=['pytorch', 'onnx_fp32', 'onnx_int8'])
    parser.add_argument('--conf',         type=float, default=0.25)
    parser.add_argument('--iou',          type=float, default=0.45)
    parser.add_argument('--num_runs',     type=int, default=50)
    parser.add_argument('--all',          action='store_true',
                        help='Benchmark all dataset/model/format combinations')
    return parser.parse_args()


NUM_CLASSES = {'NEU-DET': 5, 'DAGM': 2, 'KSDD2': 2}
DATASETS      = ['NEU-DET', 'DAGM', 'KSDD2']
YOLO_VARIANTS = ['yolov8n', 'yolov8s']
CNN_VARIANTS  = ['resnet50', 'efficientnet_b0', 'mobilenet_v3_large']
FORMATS       = ['pytorch', 'onnx_fp32', 'onnx_int8']


def resolve_paths(yolo_dir: str, cnn_dir: str,
                  dataset: str, yolo_variant: str, cnn_variant: str, fmt: str):
    """Return (yolo_path, cnn_path) for the given combination, or None if missing."""
    yolo_root = Path(yolo_dir)
    cnn_root  = Path(cnn_dir)
    run_y = f'{dataset}_{yolo_variant}'
    run_c = f'{dataset}_{cnn_variant}'

    if fmt == 'pytorch':
        yp = yolo_root / 'full'      / f'{run_y}_best.pt'
        cp = cnn_root  / 'full'      / f'{run_c}_best.pth'
    elif fmt == 'onnx_fp32':
        yp = yolo_root / 'onnx'      / f'{run_y}.onnx'
        cp = cnn_root  / 'onnx'      / f'{run_c}.onnx'
    elif fmt == 'onnx_int8':
        yp = yolo_root / 'quantized' / f'{run_y}_int8.onnx'
        cp = cnn_root  / 'quantized' / f'{run_c}_int8.onnx'
    else:
        return None, None

    return (str(yp) if yp.exists() else None,
            str(cp) if cp.exists() else None)


def load_yolo(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def load_cnn_onnx(model_path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    return sess, inp_name


def load_cnn_pytorch(model_path: str, model_name: str, num_classes: int):
    import torch
    import torchvision.models as tvm
    import torch.nn as nn

    if model_name == 'resnet50':
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        m = tvm.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f'Unknown CNN: {model_name}')
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval()
    return m


def preprocess_crop_for_cnn(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize crop to 224x224, normalize to ImageNet stats, return NCHW float32."""
    crop_rgb = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    img = crop_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)


def crop_boxes(img: np.ndarray, boxes, pad: int = 8) -> list:
    h, w = img.shape[:2]
    crops = []
    if boxes is None:
        return crops
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 > x1 and y2 > y1:
            crops.append(img[y1:y2, x1:x2])
    return crops


def get_test_images(datasets_dir: str, dataset: str, max_n: int = 300) -> list:
    img_dir = Path(datasets_dir) / dataset / 'images' / 'test'
    if not img_dir.exists():
        return []
    imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    return [str(p) for p in imgs[:max_n]]


def benchmark_one(yolo_model, cnn_backend, cnn_inp_name, fmt: str,
                  image_paths: list, conf: float, iou: float, num_runs: int) -> dict:
    """Run pipeline on images, return timing dict."""
    det_times, cls_times, e2e_times, det_counts = [], [], [], []

    for img_path in image_paths[:num_runs]:
        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.perf_counter()

        # Stage 1: Detect
        t_det = time.perf_counter()
        results = yolo_model(img, conf=conf, iou=iou, verbose=False, device='cpu')
        det_ms = (time.perf_counter() - t_det) * 1000

        boxes  = results[0].boxes if results else None
        crops  = crop_boxes(img, boxes)
        det_counts.append(len(crops))

        # Stage 2: Classify each crop
        t_cls = time.perf_counter()
        for crop in crops:
            if crop.size == 0:
                continue
            inp = preprocess_crop_for_cnn(crop)
            if fmt == 'pytorch':
                import torch
                with torch.no_grad():
                    cnn_backend(torch.tensor(inp))
            else:
                cnn_backend.run(None, {cnn_inp_name: inp})
        cls_ms = (time.perf_counter() - t_cls) * 1000

        e2e_ms = (time.perf_counter() - t0) * 1000
        det_times.append(det_ms)
        cls_times.append(cls_ms)
        e2e_times.append(e2e_ms)

    if not e2e_times:
        return {'status': 'no_images'}

    det_arr = np.array(det_times)
    cls_arr = np.array(cls_times)
    e2e_arr = np.array(e2e_times)

    return {
        'num_images':       len(e2e_times),
        'mean_det_ms':      round(float(det_arr.mean()), 2),
        'mean_cls_ms':      round(float(cls_arr.mean()), 2),
        'mean_e2e_ms':      round(float(e2e_arr.mean()), 2),
        'std_e2e_ms':       round(float(e2e_arr.std()),  2),
        'p95_e2e_ms':       round(float(np.percentile(e2e_arr, 95)), 2),
        'fps_e2e':          round(1000 / e2e_arr.mean(), 2),
        'avg_detections':   round(float(np.mean(det_counts)), 2),
        'det_pct':          round(float(det_arr.mean() / (e2e_arr.mean() + 1e-9) * 100), 1),
        'cls_pct':          round(float(cls_arr.mean() / (e2e_arr.mean() + 1e-9) * 100), 1),
        'status':           'ok',
    }


def run_single(args, dataset, yolo_variant, cnn_variant, fmt, image_paths) -> dict | None:
    yolo_path, cnn_path = resolve_paths(
        args.yolo_dir, args.cnn_dir, dataset, yolo_variant, cnn_variant, fmt)

    if not yolo_path:
        logger.warning(f'YOLO model missing: {dataset}/{yolo_variant} ({fmt})')
        return None
    if not cnn_path:
        logger.warning(f'CNN model missing: {dataset}/{cnn_variant} ({fmt})')
        return None

    logger.info(f'[{fmt}] {dataset} | YOLO={yolo_variant} CNN={cnn_variant}')
    try:
        yolo_model = load_yolo(yolo_path)
        num_cls = NUM_CLASSES.get(dataset, 2)

        if fmt == 'pytorch':
            import torch
            cnn_backend  = load_cnn_pytorch(cnn_path, cnn_variant, num_cls)
            cnn_inp_name = None
        else:
            cnn_backend, cnn_inp_name = load_cnn_onnx(cnn_path)

        metrics = benchmark_one(
            yolo_model, cnn_backend, cnn_inp_name, fmt,
            image_paths, args.conf, args.iou, args.num_runs)

        return {
            'dataset': dataset, 'yolo_model': yolo_variant,
            'cnn_model': cnn_variant, 'format': fmt,
            **metrics,
        }
    except Exception as e:
        logger.error(f'  Error: {e}')
        return {'dataset': dataset, 'yolo_model': yolo_variant,
                'cnn_model': cnn_variant, 'format': fmt, 'status': 'error', 'error': str(e)}


def main():
    args = parse_args()
    logger.info('Combined Pipeline Benchmark — Starting')
    os.makedirs(f'{args.results_dir}/tables', exist_ok=True)
    out_path = f'{args.results_dir}/tables/combined_pipeline_results.csv'

    combos = []
    if args.all:
        for ds in DATASETS:
            for yv in YOLO_VARIANTS:
                for cv in CNN_VARIANTS:
                    for fmt in FORMATS:
                        combos.append((ds, yv, cv, fmt))
    else:
        combos = [(args.dataset, args.yolo_variant, args.cnn_variant, args.format)]

    results = []
    for ds, yv, cv, fmt in tqdm(combos, desc='Combinations'):
        imgs = get_test_images(args.datasets_dir, ds, max_n=300)
        if not imgs:
            logger.warning(f'No test images for {ds}')
            continue
        row = run_single(args, ds, yv, cv, fmt, imgs)
        if row:
            results.append(row)

    if not results:
        logger.error('No results produced.')
        sys.exit(1)

    df = pd.DataFrame(results)
    # Append to existing if present
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=['dataset', 'yolo_model', 'cnn_model', 'format'], keep='last')
    df.to_csv(out_path, index=False)
    logger.info(f'Results saved: {out_path}')

    print('\n' + '='*70)
    print('Combined Pipeline Results')
    print('='*70)
    cols = ['dataset', 'yolo_model', 'cnn_model', 'format',
            'mean_e2e_ms', 'fps_e2e', 'det_pct', 'cls_pct']
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
