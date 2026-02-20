#!/usr/bin/env python3
"""
Energy Estimation — TDP-based energy proxy for all model variants.

Methodology:
  - CPU TDP = 15W (Intel i5-1335U PBP)
  - Sample psutil.cpu_percent() during inference in a separate thread
  - Energy_per_frame (J) = TDP * (avg_cpu_util / 100) * inference_time_seconds
  - Also computes Wh per frame and Wh per 1000 frames
  - 50 inference passes per model while measuring

Usage:
    python energy_estimation.py --models_dir ./models --results_dir ./results
    python energy_estimation.py --num_runs 100 --tdp 28.0
"""

import argparse
import logging
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CPU_TDP_WATTS = 15.0  # Intel i5-1335U sustained PBP


def parse_args():
    parser = argparse.ArgumentParser(description='Energy Estimation')
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Root directory containing model variants')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--num_runs', type=int, default=50,
                        help='Number of inference passes per model')
    parser.add_argument('--tdp', type=float, default=CPU_TDP_WATTS,
                        help='CPU TDP in Watts (default 15W for i5-1335U)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Warmup runs before measuring')
    return parser.parse_args()


# ── CPU utilisation sampler ──────────────────────────────────────────────────

class CPUSampler:
    """Samples CPU utilisation at ~100 ms intervals in a background thread."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        self._thread.join(timeout=2)
        return list(self._samples)

    def _run(self):
        while not self._stop.is_set():
            util = psutil.cpu_percent(interval=None)
            self._samples.append(util)
            time.sleep(self.interval)


# ── Per-backend energy measurement ───────────────────────────────────────────

def measure_energy_yolo_pt(model_path: str, warmup: int, runs: int, tdp: float) -> dict:
    try:
        from ultralytics import YOLO
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        model = YOLO(model_path)
        for _ in range(warmup):
            model(dummy, verbose=False, device='cpu')

        sampler = CPUSampler(interval=0.05)
        sampler.start()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy, verbose=False, device='cpu')
        elapsed = time.perf_counter() - t0
        samples = sampler.stop()

        avg_util = float(np.mean(samples)) if samples else 50.0
        energy_j_total = tdp * (avg_util / 100.0) * elapsed
        energy_j_per_frame = energy_j_total / runs
        return {
            'elapsed_s':       round(elapsed, 3),
            'avg_cpu_pct':     round(avg_util, 1),
            'energy_j_total':  round(energy_j_total, 4),
            'energy_j_frame':  round(energy_j_per_frame, 6),
            'wh_per_frame':    round(energy_j_per_frame / 3600, 8),
            'wh_per_1000':     round(energy_j_per_frame * 1000 / 3600, 5),
            'mean_ms':         round(elapsed / runs * 1000, 2),
            'fps':             round(runs / elapsed, 1),
            'status':          'ok',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def measure_energy_onnx(model_path: str, input_shape: tuple,
                        warmup: int, runs: int, tdp: float) -> dict:
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(warmup):
            sess.run(None, {inp_name: dummy})

        sampler = CPUSampler(interval=0.05)
        sampler.start()
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(None, {inp_name: dummy})
        elapsed = time.perf_counter() - t0
        samples = sampler.stop()

        avg_util = float(np.mean(samples)) if samples else 50.0
        energy_j_total = tdp * (avg_util / 100.0) * elapsed
        energy_j_per_frame = energy_j_total / runs
        return {
            'elapsed_s':       round(elapsed, 3),
            'avg_cpu_pct':     round(avg_util, 1),
            'energy_j_total':  round(energy_j_total, 4),
            'energy_j_frame':  round(energy_j_per_frame, 6),
            'wh_per_frame':    round(energy_j_per_frame / 3600, 8),
            'wh_per_1000':     round(energy_j_per_frame * 1000 / 3600, 5),
            'mean_ms':         round(elapsed / runs * 1000, 2),
            'fps':             round(runs / elapsed, 1),
            'status':          'ok',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def measure_energy_tflite(model_path: str, input_shape: tuple,
                           warmup: int, runs: int, tdp: float) -> dict:
    try:
        import tensorflow as tf
    except ImportError:
        return {'status': 'tflite_not_installed'}
    try:
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp_details = interp.get_input_details()
        # Use actual model input shape to avoid NCHW/NHWC mismatch
        actual_shape = tuple(inp_details[0]['shape'])
        dummy = np.random.randn(*actual_shape).astype(np.float32)

        for _ in range(warmup):
            interp.set_tensor(inp_details[0]['index'], dummy)
            interp.invoke()

        sampler = CPUSampler(interval=0.05)
        sampler.start()
        t0 = time.perf_counter()
        for _ in range(runs):
            interp.set_tensor(inp_details[0]['index'], dummy)
            interp.invoke()
        elapsed = time.perf_counter() - t0
        samples = sampler.stop()

        avg_util = float(np.mean(samples)) if samples else 50.0
        energy_j_total = tdp * (avg_util / 100.0) * elapsed
        energy_j_per_frame = energy_j_total / runs
        return {
            'elapsed_s':       round(elapsed, 3),
            'avg_cpu_pct':     round(avg_util, 1),
            'energy_j_total':  round(energy_j_total, 4),
            'energy_j_frame':  round(energy_j_per_frame, 6),
            'wh_per_frame':    round(energy_j_per_frame / 3600, 8),
            'wh_per_1000':     round(energy_j_per_frame * 1000 / 3600, 5),
            'mean_ms':         round(elapsed / runs * 1000, 2),
            'fps':             round(runs / elapsed, 1),
            'status':          'ok',
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tdp = args.tdp
    logger.info('Energy Estimation — Starting')
    logger.info(f'CPU TDP: {tdp}W | runs={args.num_runs} warmup={args.warmup}')
    os.makedirs(f'{args.results_dir}/tables', exist_ok=True)

    root = Path(args.models_dir)
    DATASETS      = ['NEU-DET', 'DAGM', 'KSDD2']
    YOLO_VARIANTS = ['yolov8n', 'yolov8s']
    CNN_VARIANTS  = ['resnet50', 'efficientnet_b0', 'mobilenet_v3_large']
    results = []

    # ── YOLO energy ──
    logger.info('\n=== YOLO Energy ===')
    for ds in DATASETS:
        for mv in YOLO_VARIANTS:
            run = f'{ds}_{mv}'

            # PyTorch .pt
            pt = root / 'yolo' / 'full' / f'{run}_best.pt'
            if pt.exists():
                logger.info(f'[YOLO PT] {run}')
                r = measure_energy_yolo_pt(str(pt), args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'pytorch',
                          'type': 'yolo', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

            # ONNX FP32
            onnx_fp32 = root / 'yolo' / 'onnx' / f'{run}.onnx'
            if onnx_fp32.exists():
                logger.info(f'[YOLO ONNX FP32] {run}')
                r = measure_energy_onnx(str(onnx_fp32), (1, 3, 640, 640),
                                        args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_fp32',
                          'type': 'yolo', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

            # ONNX INT8
            onnx_int8 = root / 'yolo' / 'quantized' / f'{run}_int8.onnx'
            if onnx_int8.exists():
                logger.info(f'[YOLO ONNX INT8] {run}')
                r = measure_energy_onnx(str(onnx_int8), (1, 3, 640, 640),
                                        args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_int8',
                          'type': 'yolo', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

            # TFLite
            tfl = root / 'yolo' / 'tflite' / f'{run}.tflite'
            if tfl.exists():
                logger.info(f'[YOLO TFLite] {run}')
                r = measure_energy_tflite(str(tfl), (1, 640, 640, 3),
                                          args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'tflite',
                          'type': 'yolo', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

    # ── CNN energy ──
    logger.info('\n=== CNN Energy ===')
    for ds in DATASETS:
        for mv in CNN_VARIANTS:
            run = f'{ds}_{mv}'

            onnx_dir = root / 'cnn' / 'onnx'
            onnx_fp32 = onnx_dir / f'{run}_fp32_sanitized.onnx'
            if not onnx_fp32.exists():
                onnx_fp32 = onnx_dir / f'{run}_sanitized.onnx'
            if not onnx_fp32.exists():
                onnx_fp32 = onnx_dir / f'{run}.onnx'
            if onnx_fp32.exists():
                logger.info(f'[CNN ONNX FP32] {run}')
                r = measure_energy_onnx(str(onnx_fp32), (1, 3, 224, 224),
                                        args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_fp32',
                          'type': 'cnn', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

            onnx_int8 = root / 'cnn' / 'quantized' / f'{run}_int8.onnx'
            if onnx_int8.exists():
                logger.info(f'[CNN ONNX INT8] {run}')
                r = measure_energy_onnx(str(onnx_int8), (1, 3, 224, 224),
                                        args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'onnx_int8',
                          'type': 'cnn', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

            tfl_fp32 = root / 'cnn' / 'tflite' / f'{run}.tflite'
            if tfl_fp32.exists():
                logger.info(f'[CNN TFLite FP32] {run}')
                r = measure_energy_tflite(str(tfl_fp32), (1, 3, 224, 224),
                                          args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'tflite_fp32',
                          'type': 'cnn', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

            tfl_int8 = root / 'cnn' / 'quantized' / f'{run}_int8.tflite'
            if tfl_int8.exists():
                logger.info(f'[CNN TFLite INT8] {run}')
                r = measure_energy_tflite(str(tfl_int8), (1, 3, 224, 224),
                                          args.warmup, args.num_runs, tdp)
                r.update({'dataset': ds, 'model': mv, 'format': 'tflite_int8',
                          'type': 'cnn', 'tdp_w': tdp})
                results.append(r)
                logger.info(f'  {r.get("mean_ms")}ms  {r.get("wh_per_1000")} Wh/1000')

    if not results:
        logger.error('No models found under --models_dir.')
        sys.exit(1)

    df = pd.DataFrame(results)
    out = f'{args.results_dir}/tables/energy_estimation.csv'
    df.to_csv(out, index=False)
    logger.info(f'Saved: {out}')

    print('\n' + '='*70)
    print('Energy Estimation Summary')
    print('='*70)
    cols = ['dataset', 'model', 'format', 'type', 'mean_ms', 'fps',
            'avg_cpu_pct', 'wh_per_1000']
    avail = [c for c in cols if c in df.columns]
    ok = df[df.get('status', pd.Series(['ok']*len(df))) == 'ok'] if 'status' in df.columns else df
    if not ok.empty:
        print(ok[avail].to_string(index=False))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
