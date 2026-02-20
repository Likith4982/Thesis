#!/usr/bin/env python3
"""
CNN CPU Inference — Evaluate all CNN model variants on CPU.

Loads each CNN variant (.pth, ONNX FP32, ONNX INT8, TFLite FP32, TFLite INT8)
and runs inference on test sets. Records accuracy, macro-F1, per-class precision/recall.

Usage:
    python run_cpu_inference_cnn.py --models_dir ./models/cnn --datasets_dir ./datasets --results_dir ./results
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='CNN CPU Inference Benchmark')
    parser.add_argument('--models_dir', type=str, default='./models/cnn')
    parser.add_argument('--datasets_dir', type=str, default='./datasets')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Inference batch size (1 for latency, 32 for throughput)')
    return parser.parse_args()


NUM_CLASSES = {'NEU-DET': 5, 'DAGM': 2, 'KSDD2': 2}
CNN_MODEL_NAMES = ['resnet50', 'efficientnet_b0', 'mobilenet_v3_large']
DATASETS = ['NEU-DET', 'DAGM', 'KSDD2']


def build_model_skeleton(model_name: str, num_classes: int):
    """Build CNN skeleton for loading weights."""
    import torchvision.models as models
    import torch.nn as nn
    if model_name == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return m


def get_test_loader(dataset_name: str, datasets_dir: str, batch_size: int = 1):
    """Build test DataLoader from ImageFolder structure."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dir = os.path.join(datasets_dir, dataset_name, 'classification', 'test')
    if not os.path.exists(test_dir):
        return None, None
    ds = ImageFolder(test_dir, transform=val_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, ds.classes


def evaluate_pytorch(model_path: str, model_name: str, dataset: str,
                     datasets_dir: str, batch_size: int) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    num_cls = NUM_CLASSES.get(dataset, 2)
    loader, class_names = get_test_loader(dataset, datasets_dir, batch_size)
    if loader is None:
        return {'status': 'no_test_set'}

    model = build_model_skeleton(model_name, num_cls)
    try:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
    except Exception as e:
        return {'status': 'load_error', 'error': str(e)}

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {'accuracy': round(acc, 4), 'f1_macro': round(f1, 4),
            'num_samples': len(all_labels), 'status': 'ok'}


def evaluate_onnx(model_path: str, dataset: str, datasets_dir: str, batch_size: int) -> dict:
    import onnxruntime as ort
    from sklearn.metrics import accuracy_score, f1_score

    loader, _ = get_test_loader(dataset, datasets_dir, batch_size)
    if loader is None:
        return {'status': 'no_test_set'}

    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        return {'status': 'load_error', 'error': str(e)}

    all_preds, all_labels = [], []
    for imgs, labels in loader:
        out = sess.run(None, {input_name: imgs.numpy()})
        preds = np.argmax(out[0], axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {'accuracy': round(acc, 4), 'f1_macro': round(f1, 4),
            'num_samples': len(all_labels), 'status': 'ok'}


def evaluate_tflite(model_path: str, dataset: str, datasets_dir: str) -> dict:
    try:
        import tensorflow as tf
    except ImportError:
        return {'status': 'tflite_not_installed'}
    from sklearn.metrics import accuracy_score, f1_score

    loader, _ = get_test_loader(dataset, datasets_dir, batch_size=1)
    if loader is None:
        return {'status': 'no_test_set'}

    try:
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()
    except Exception as e:
        return {'status': 'load_error', 'error': str(e)}

    # Check if TFLite expects NHWC; PyTorch DataLoader outputs NCHW
    inp_shape = inp_details[0]['shape']  # e.g. [1,224,224,3] or [1,3,224,224]
    needs_transpose = (len(inp_shape) == 4 and inp_shape[-1] in (1, 3))

    all_preds, all_labels = [], []
    for imgs, labels in loader:
        img_np = imgs.numpy().astype(np.float32)
        if needs_transpose:
            img_np = img_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        interp.set_tensor(inp_details[0]['index'], img_np)
        interp.invoke()
        out = interp.get_tensor(out_details[0]['index'])
        all_preds.append(int(np.argmax(out, axis=1)[0]))
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {'accuracy': round(acc, 4), 'f1_macro': round(f1, 4),
            'num_samples': len(all_labels), 'status': 'ok'}


def main():
    args = parse_args()
    logger.info('CNN CPU Inference — Starting')
    os.makedirs(f'{args.results_dir}/tables', exist_ok=True)

    results = []

    for dataset in DATASETS:
        for model_name in CNN_MODEL_NAMES:
            run_name = f'{dataset}_{model_name}'

            # -- PyTorch .pth --
            pth = Path(args.models_dir) / 'full' / f'{run_name}_best.pth'
            if pth.exists():
                logger.info(f'[PyTorch] {run_name}')
                t0 = time.time()
                m = evaluate_pytorch(str(pth), model_name, dataset, args.datasets_dir, args.batch_size)
                m.update({'dataset': dataset, 'model': model_name, 'format': 'pytorch',
                          'eval_s': round(time.time() - t0, 1)})
                results.append(m)
                logger.info(f'  acc={m.get("accuracy")} f1={m.get("f1_macro")}')

            # -- ONNX FP32 -- (prefer sanitized self-contained files)
            onnx_dir = Path(args.models_dir) / 'onnx'
            onnx_fp32 = onnx_dir / f'{run_name}_fp32_sanitized.onnx'
            if not onnx_fp32.exists():
                onnx_fp32 = onnx_dir / f'{run_name}_sanitized.onnx'
            if not onnx_fp32.exists():
                onnx_fp32 = onnx_dir / f'{run_name}.onnx'
            if onnx_fp32.exists():
                logger.info(f'[ONNX FP32] {run_name}')
                t0 = time.time()
                m = evaluate_onnx(str(onnx_fp32), dataset, args.datasets_dir, args.batch_size)
                m.update({'dataset': dataset, 'model': model_name, 'format': 'onnx_fp32',
                          'eval_s': round(time.time() - t0, 1)})
                results.append(m)
                logger.info(f'  acc={m.get("accuracy")} f1={m.get("f1_macro")}')

            # -- ONNX INT8 --
            onnx_int8 = Path(args.models_dir) / 'quantized' / f'{run_name}_int8.onnx'
            if onnx_int8.exists():
                logger.info(f'[ONNX INT8] {run_name}')
                t0 = time.time()
                m = evaluate_onnx(str(onnx_int8), dataset, args.datasets_dir, args.batch_size)
                m.update({'dataset': dataset, 'model': model_name, 'format': 'onnx_int8',
                          'eval_s': round(time.time() - t0, 1)})
                results.append(m)
                logger.info(f'  acc={m.get("accuracy")} f1={m.get("f1_macro")}')

            # -- TFLite FP32 --
            tflite_fp32 = Path(args.models_dir) / 'tflite' / f'{run_name}.tflite'
            if tflite_fp32.exists():
                logger.info(f'[TFLite FP32] {run_name}')
                t0 = time.time()
                m = evaluate_tflite(str(tflite_fp32), dataset, args.datasets_dir)
                m.update({'dataset': dataset, 'model': model_name, 'format': 'tflite_fp32',
                          'eval_s': round(time.time() - t0, 1)})
                results.append(m)
                logger.info(f'  acc={m.get("accuracy")} f1={m.get("f1_macro")}')

            # -- TFLite INT8 --
            tflite_int8 = Path(args.models_dir) / 'quantized' / f'{run_name}_int8.tflite'
            if tflite_int8.exists():
                logger.info(f'[TFLite INT8] {run_name}')
                t0 = time.time()
                m = evaluate_tflite(str(tflite_int8), dataset, args.datasets_dir)
                m.update({'dataset': dataset, 'model': model_name, 'format': 'tflite_int8',
                          'eval_s': round(time.time() - t0, 1)})
                results.append(m)
                logger.info(f'  acc={m.get("accuracy")} f1={m.get("f1_macro")}')

    df = pd.DataFrame(results)
    out = f'{args.results_dir}/tables/cnn_cpu_inference.csv'
    df.to_csv(out, index=False)
    logger.info(f'Results saved: {out}')

    print('\n' + '='*70)
    print('CNN CPU Inference Summary')
    print('='*70)
    if not df.empty and 'accuracy' in df.columns:
        print(df[['dataset', 'model', 'format', 'accuracy', 'f1_macro']].to_string(index=False))
    print(f'\nFull results: {out}')


if __name__ == '__main__':
    main()
