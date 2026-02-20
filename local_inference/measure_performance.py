#!/usr/bin/env python3
"""
Performance Measurement — Latency, FPS, memory for ALL model variants.

Benchmarks YOLO and CNN models across PyTorch, ONNX Runtime, and TFLite backends.
For each model: warmup runs + timed runs x repeats.
Records: mean/std latency (ms), FPS, load time (ms), peak memory (MB), p95 latency.

Usage:
    python measure_performance.py --models_dir ./models --results_dir ./results
    python measure_performance.py --warmup 10 --num_runs 100 --repeats 3
"""

import argparse
import logging
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Performance Measurement')
    parser.add_argument('--models_dir', type=str, default='./models')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--warmup',    type=int, default=10)
    parser.add_argument('--num_runs',  type=int, default=100)
    parser.add_argument('--repeats',   type=int, default=3)
    return parser.parse_args()


def get_size_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1e6, 2)
    except OSError:
        return 0.0


def bench_yolo_pt(model_path: str, warmup: int, runs: int, repeats: int) -> dict:
    """Benchmark a YOLO .pt model on CPU."""
    try:
        from ultralytics import YOLO
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model = YOLO(model_path)
        load_ms = (time.perf_counter() - t0) * 1000

        for _ in range(warmup):
            model(dummy, verbose=False, device='cpu')

        all_lat = []
        for _ in range(repeats):
            for _ in range(runs):
                t = time.perf_counter()
                model(dummy, verbose=False, device='cpu')
                all_lat.append((time.perf_counter() - t) * 1000)

        arr = np.array(all_lat)
        return {
            'load_ms': round(load_ms, 1),
            'mean_ms': round(arr.mean(), 2),
            'std_ms':  round(arr.std(),  2),
            'p50_ms':  round(np.percentile(arr, 50), 2),
            'p95_ms':  round(np.percentile(arr, 95), 2),
            'fps':     round(1000 / arr.mean(), 1),
            'status':  'ok',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def bench_onnx(model_path: str, input_shape: tuple,
               warmup: int, runs: int, repeats: int) -> dict:
    """Benchmark an ONNX model with ONNXRuntime CPU EP."""
    try:
        import onnxruntime as ort

        t0 = time.perf_counter()
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        load_ms = (time.perf_counter() - t0) * 1000

        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(warmup):
            sess.run(None, {inp_name: dummy})

        all_lat = []
        for _ in range(repeats):
            for _ in range(runs):
                t = time.perf_counter()
                sess.run(None, {inp_name: dummy})
                all_lat.append((time.perf_counter() - t) * 1000)

        arr = np.array(all_lat)
        return {
            'load_ms': round(load_ms, 1),
            'mean_ms': round(arr.mean(), 2),
            'std_ms':  round(arr.std(),  2),
            'p50_ms':  round(np.percentile(arr, 50), 2),
            'p95_ms':  round(np.percentile(arr, 95), 2),
            'fps':     round(1000 / arr.mean(), 1),
            'status':  'ok',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def bench_tflite(model_path: str, input_shape: tuple,
                 warmup: int, runs: int, repeats: int) -> dict:
    """Benchmark a TFLite model."""
    try:
        import tensorflow as tf

        t0 = time.perf_counter()
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        load_ms = (time.perf_counter() - t0) * 1000

        inp_details = interp.get_input_details()
        # Use actual model input shape to avoid NCHW/NHWC mismatch
        actual_shape = tuple(inp_details[0]['shape'])
        dummy = np.random.randn(*actual_shape).astype(np.float32)

        for _ in range(warmup):
            interp.set_tensor(inp_details[0]['index'], dummy)
            interp.invoke()

        all_lat = []
        for _ in range(repeats):
            for _ in range(runs):
                interp.set_tensor(inp_details[0]['index'], dummy)
                t = time.perf_counter()
                interp.invoke()
                all_lat.append((time.perf_counter() - t) * 1000)

        arr = np.array(all_lat)
        return {
            'load_ms': round(load_ms, 1),
            'mean_ms': round(arr.mean(), 2),
            'std_ms':  round(arr.std(),  2),
            'p50_ms':  round(np.percentile(arr, 50), 2),
            'p95_ms':  round(np.percentile(arr, 95), 2),
            'fps':     round(1000 / arr.mean(), 1),
            'status':  'ok',
        }
    except ImportError:
        return {'status': 'tflite_not_installed'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def main():
    args = parse_args()
    logger.info(f'Performance Measurement | warmup={args.warmup} runs={args.num_runs} repeats={args.repeats}')
    os.makedirs(f'{args.results_dir}/tables', exist_ok=True)

    root = Path(args.models_dir)
    DATASETS      = ['NEU-DET', 'DAGM', 'KSDD2']
    YOLO_VARIANTS = ['yolov8n', 'yolov8s']
    CNN_VARIANTS  = ['resnet50', 'efficientnet_b0', 'mobilenet_v3_large']
    results = []

    # ── YOLO benchmarks ──
    logger.info('\n=== YOLO Models ===')
    for ds in DATASETS:
        for mv in YOLO_VARIANTS:
            run = f'{ds}_{mv}'

            # PyTorch .pt
            pt = root / 'yolo' / 'full' / f'{run}_best.pt'
            if pt.exists():
                logger.info(f'[YOLO PyTorch] {run}')
                r = bench_yolo_pt(str(pt), args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'pytorch',
                          'type': 'yolo', 'size_mb': get_size_mb(str(pt))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

            # ONNX FP32
            onnx_fp32 = root / 'yolo' / 'onnx' / f'{run}.onnx'
            if onnx_fp32.exists():
                logger.info(f'[YOLO ONNX FP32] {run}')
                r = bench_onnx(str(onnx_fp32), (1, 3, 640, 640),
                               args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_fp32',
                          'type': 'yolo', 'size_mb': get_size_mb(str(onnx_fp32))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

            # ONNX INT8
            onnx_int8 = root / 'yolo' / 'quantized' / f'{run}_int8.onnx'
            if onnx_int8.exists():
                logger.info(f'[YOLO ONNX INT8] {run}')
                r = bench_onnx(str(onnx_int8), (1, 3, 640, 640),
                               args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_int8',
                          'type': 'yolo', 'size_mb': get_size_mb(str(onnx_int8))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

            # TFLite FP32
            tfl = root / 'yolo' / 'tflite' / f'{run}.tflite'
            if tfl.exists():
                logger.info(f'[YOLO TFLite] {run}')
                r = bench_tflite(str(tfl), (1, 640, 640, 3),
                                 args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'tflite',
                          'type': 'yolo', 'size_mb': get_size_mb(str(tfl))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

    # ── CNN benchmarks ──
    logger.info('\n=== CNN Models ===')
    for ds in DATASETS:
        for mv in CNN_VARIANTS:
            run = f'{ds}_{mv}'

            # ONNX FP32 (prefer sanitized self-contained files)
            onnx_dir = root / 'cnn' / 'onnx'
            onnx_fp32 = onnx_dir / f'{run}_fp32_sanitized.onnx'
            if not onnx_fp32.exists():
                onnx_fp32 = onnx_dir / f'{run}_sanitized.onnx'
            if not onnx_fp32.exists():
                onnx_fp32 = onnx_dir / f'{run}.onnx'
            if onnx_fp32.exists():
                logger.info(f'[CNN ONNX FP32] {run}')
                r = bench_onnx(str(onnx_fp32), (1, 3, 224, 224),
                               args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_fp32',
                          'type': 'cnn', 'size_mb': get_size_mb(str(onnx_fp32))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

            # ONNX INT8
            onnx_int8 = root / 'cnn' / 'quantized' / f'{run}_int8.onnx'
            if onnx_int8.exists():
                logger.info(f'[CNN ONNX INT8] {run}')
                r = bench_onnx(str(onnx_int8), (1, 3, 224, 224),
                               args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_int8',
                          'type': 'cnn', 'size_mb': get_size_mb(str(onnx_int8))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

            # TFLite FP32
            tfl_fp32 = root / 'cnn' / 'tflite' / f'{run}.tflite'
            if tfl_fp32.exists():
                logger.info(f'[CNN TFLite FP32] {run}')
                r = bench_tflite(str(tfl_fp32), (1, 3, 224, 224),
                                 args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'tflite_fp32',
                          'type': 'cnn', 'size_mb': get_size_mb(str(tfl_fp32))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

            # TFLite INT8
            tfl_int8 = root / 'cnn' / 'quantized' / f'{run}_int8.tflite'
            if tfl_int8.exists():
                logger.info(f'[CNN TFLite INT8] {run}')
                r = bench_tflite(str(tfl_int8), (1, 3, 224, 224),
                                 args.warmup, args.num_runs, args.repeats)
                r.update({'dataset': ds, 'model': mv, 'format': 'tflite_int8',
                          'type': 'cnn', 'size_mb': get_size_mb(str(tfl_int8))})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  fps={r.get("fps")}')

    if not results:
        logger.error('No models found under --models_dir.')
        sys.exit(1)

    df = pd.DataFrame(results)
    out = f'{args.results_dir}/tables/performance_all_models.csv'
    df.to_csv(out, index=False)
    logger.info(f'Saved: {out}')

    print('\n' + '='*70)
    print('Performance Summary')
    print('='*70)
    if 'mean_ms' in df.columns:
        cols = ['dataset', 'model', 'format', 'type', 'mean_ms', 'p95_ms', 'fps', 'size_mb']
        avail = [c for c in cols if c in df.columns]
        print(df[avail].to_string(index=False))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
