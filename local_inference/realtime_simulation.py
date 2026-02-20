#!/usr/bin/env python3
"""
Real-Time Defect Inspection Simulation.

Reads images sequentially simulating a conveyor belt camera. Displays predictions
with bounding boxes, class labels, confidence scores, and live FPS counter.
Supports YOLO, CNN, and combined modes. Optional webcam input.

Usage:
    python realtime_simulation.py --mode yolo --model_path ./models/yolo/full/NEU-DET_yolov8n_best.pt \
        --dataset_dir ./datasets/NEU-DET/images/test
    python realtime_simulation.py --mode cnn --model_path ./models/cnn/onnx/NEU-DET_resnet50.onnx \
        --dataset_dir ./datasets/NEU-DET/classification/test --model_name resnet50
    python realtime_simulation.py --mode combined \
        --yolo_model ./models/yolo/onnx/NEU-DET_yolov8n.onnx \
        --cnn_model  ./models/cnn/onnx/NEU-DET_resnet50.onnx \
        --dataset_dir ./datasets/NEU-DET/images/test
    python realtime_simulation.py --mode yolo --model_path ... --webcam
"""

import argparse
import logging
import os
import sys
import time
import collections
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Display constants
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.55
THICKNESS     = 2
WIN_NAME      = 'Defect Inspection — q:quit  s:screenshot  p:pause  +/-:speed'
BOX_COLOUR    = (0, 200, 0)
LABEL_COLOUR  = (255, 255, 255)
TEXT_BG       = (0, 0, 0)

NUM_CLASSES   = {'NEU-DET': 5, 'DAGM': 2, 'KSDD2': 2}
CLASS_LABELS  = {
    'NEU-DET': ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale'],
    'DAGM':    ['defect', 'no_defect'],
    'KSDD2':   ['defect', 'no_defect'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Real-Time Defect Inspection Simulation')
    parser.add_argument('--mode', type=str, choices=['yolo', 'cnn', 'combined'],
                        required=True, help='Inference mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model (for yolo or cnn mode)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='CNN backbone name (resnet50/efficientnet_b0/mobilenet_v3_large) for .pth loading')
    parser.add_argument('--yolo_model', type=str, default=None,
                        help='Path to YOLO model (for combined mode)')
    parser.add_argument('--cnn_model', type=str, default=None,
                        help='Path to CNN model (for combined mode)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to test images directory')
    parser.add_argument('--dataset', type=str, default='NEU-DET',
                        choices=['NEU-DET', 'DAGM', 'KSDD2'],
                        help='Dataset name for class labels')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam input instead of dataset images')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='Webcam device ID')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Min delay between frames in seconds (0 = as fast as possible)')
    parser.add_argument('--save_video', action='store_true',
                        help='Save output to results/realtime_demo.mp4')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='YOLO IoU threshold')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory for screenshots and video output')
    return parser.parse_args()


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_yolo_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def load_cnn_onnx(path: str):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return sess, sess.get_inputs()[0].name


def load_cnn_pytorch(path: str, model_name: str, num_classes: int):
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
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_yolo(model, frame: np.ndarray, conf: float, iou: float) -> list:
    """Returns list of (x1,y1,x2,y2, conf, cls_id)."""
    res = model(frame, conf=conf, iou=iou, verbose=False, device='cpu')
    dets = []
    if res and res[0].boxes is not None:
        for box in res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            score = float(box.conf[0])
            cls   = int(box.cls[0])
            dets.append((x1, y1, x2, y2, score, cls))
    return dets


def preprocess_cnn(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    return ((img - mean) / std).transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)


def infer_cnn(model_obj, inp_name, frame_bgr: np.ndarray, is_pytorch: bool = False):
    """Returns (cls_id, confidence)."""
    tensor = preprocess_cnn(frame_bgr)
    if is_pytorch:
        import torch
        with torch.no_grad():
            logits = model_obj(torch.tensor(tensor)).numpy()[0]
    else:
        logits = model_obj.run(None, {inp_name: tensor})[0][0]
    probs   = np.exp(logits) / np.exp(logits).sum()
    cls_id  = int(np.argmax(probs))
    conf    = float(probs[cls_id])
    return cls_id, conf


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_yolo_boxes(frame: np.ndarray, dets: list, labels: list) -> np.ndarray:
    for (x1, y1, x2, y2, score, cls) in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, THICKNESS)
        label = f'{labels[cls] if cls < len(labels) else cls} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), TEXT_BG, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3), FONT, FONT_SCALE, LABEL_COLOUR, 1)
    return frame


def draw_cnn_label(frame: np.ndarray, cls_id: int, conf: float, labels: list) -> np.ndarray:
    label = f'{labels[cls_id] if cls_id < len(labels) else cls_id}  {conf:.2f}'
    colour = (0, 0, 200) if (cls_id == 0 and 'defect' in labels[0].lower()) else (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 24), FONT, 0.8, colour, 2)
    return frame


def draw_hud(frame: np.ndarray, fps: float, lat_ms: float,
             paused: bool, frame_idx: int, total: int) -> np.ndarray:
    h, w = frame.shape[:2]
    hud = f'FPS:{fps:5.1f}  Lat:{lat_ms:6.1f}ms  [{frame_idx}/{total}]'
    if paused:
        hud += '  [PAUSED]'
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, h - 8), FONT, 0.5, (200, 200, 200), 1)
    return frame


# ── Image source ──────────────────────────────────────────────────────────────

def get_image_list(dataset_dir: str) -> list:
    d = Path(dataset_dir)
    imgs = sorted(d.glob('*.jpg')) + sorted(d.glob('*.png')) + sorted(d.glob('*.bmp'))
    return [str(p) for p in imgs]


# ── Main display loop ─────────────────────────────────────────────────────────

def run_loop(args, frames_src, yolo_model, cnn_model, cnn_inp_name,
             labels: list, is_pytorch: bool, writer=None):
    fps_buf   = collections.deque(maxlen=30)
    stats     = {'frames': 0, 'defects': 0, 'lat_ms': []}
    paused    = False
    delay     = args.delay
    screenshot_n = 0

    os.makedirs(args.results_dir, exist_ok=True)

    total = len(frames_src) if not args.webcam else 0
    idx   = 0

    while True:
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            if 'frame_disp' in dir():
                sc_path = os.path.join(args.results_dir, f'screenshot_{screenshot_n:04d}.png')
                cv2.imwrite(sc_path, frame_disp)
                screenshot_n += 1
                logger.info(f'Screenshot: {sc_path}')
        elif key == ord('+') or key == ord('='):
            delay = max(0.0, delay - 0.02)
        elif key == ord('-'):
            delay = min(2.0, delay + 0.02)

        if paused:
            time.sleep(0.05)
            continue

        # Acquire frame
        if args.webcam:
            ret, frame = frames_src.read()
            if not ret:
                break
        else:
            if idx >= len(frames_src):
                idx = 0  # loop
            frame = cv2.imread(frames_src[idx])
            idx += 1
            if frame is None:
                continue

        t0 = time.perf_counter()

        frame_disp = frame.copy()
        dets = []

        # Inference
        if args.mode == 'yolo':
            dets = infer_yolo(yolo_model, frame, args.conf, args.iou)
            draw_yolo_boxes(frame_disp, dets, labels)

        elif args.mode == 'cnn':
            cls_id, conf = infer_cnn(cnn_model, cnn_inp_name, frame, is_pytorch)
            draw_cnn_label(frame_disp, cls_id, conf, labels)

        elif args.mode == 'combined':
            dets = infer_yolo(yolo_model, frame, args.conf, args.iou)
            draw_yolo_boxes(frame_disp, dets, labels)
            for (x1, y1, x2, y2, _, _) in dets:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cls_id, conf = infer_cnn(cnn_model, cnn_inp_name, crop, is_pytorch)
                lbl = f'{labels[cls_id] if cls_id < len(labels) else cls_id} {conf:.2f}'
                cv2.putText(frame_disp, lbl, (x1, y1 - 20),
                            FONT, FONT_SCALE, (200, 200, 0), 1)

        lat_ms = (time.perf_counter() - t0) * 1000
        fps_buf.append(1000.0 / (lat_ms + 1e-6))
        fps    = float(np.mean(fps_buf))

        stats['frames'] += 1
        stats['defects'] += len(dets)
        stats['lat_ms'].append(lat_ms)

        draw_hud(frame_disp, fps, lat_ms, paused, idx, total)
        cv2.imshow(WIN_NAME, frame_disp)
        if writer is not None:
            writer.write(frame_disp)

        elapsed = time.perf_counter() - t0
        wait    = delay - elapsed
        if wait > 0:
            time.sleep(wait)

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info(f'Real-Time Simulation — Mode: {args.mode}')

    labels     = CLASS_LABELS.get(args.dataset, [str(i) for i in range(10)])
    yolo_model = cnn_model = cnn_inp_name = None
    is_pytorch = False

    # Load models
    if args.mode in ('yolo', 'combined'):
        path = args.model_path if args.mode == 'yolo' else args.yolo_model
        if not path:
            logger.error('--model_path / --yolo_model required')
            sys.exit(1)
        logger.info(f'Loading YOLO: {path}')
        yolo_model = load_yolo_model(path)

    if args.mode in ('cnn', 'combined'):
        path = args.model_path if args.mode == 'cnn' else args.cnn_model
        if not path:
            logger.error('--model_path / --cnn_model required')
            sys.exit(1)
        logger.info(f'Loading CNN: {path}')
        if path.endswith('.pth'):
            if not args.model_name:
                logger.error('--model_name required when loading a .pth CNN')
                sys.exit(1)
            num_cls    = NUM_CLASSES.get(args.dataset, 2)
            cnn_model  = load_cnn_pytorch(path, args.model_name, num_cls)
            is_pytorch = True
        else:
            cnn_model, cnn_inp_name = load_cnn_onnx(path)

    # Image source
    if args.webcam:
        frames_src = cv2.VideoCapture(args.cam_id)
        if not frames_src.isOpened():
            logger.error(f'Cannot open webcam {args.cam_id}')
            sys.exit(1)
        total_frames = 0
    else:
        if not args.dataset_dir:
            logger.error('--dataset_dir required when not using --webcam')
            sys.exit(1)
        frames_src = get_image_list(args.dataset_dir)
        total_frames = len(frames_src)
        if not frames_src:
            logger.error(f'No images found in {args.dataset_dir}')
            sys.exit(1)
        logger.info(f'Found {total_frames} images')

    # Optional video writer
    writer = None
    if args.save_video:
        os.makedirs(args.results_dir, exist_ok=True)
        vid_path = os.path.join(args.results_dir, 'realtime_demo.mp4')
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(vid_path, fourcc, 20.0, (640, 480))
        logger.info(f'Saving video to {vid_path}')

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 600)

    try:
        stats = run_loop(args, frames_src, yolo_model, cnn_model, cnn_inp_name,
                         labels, is_pytorch, writer)
    finally:
        if args.webcam:
            frames_src.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    # Summary
    n = stats['frames']
    if n > 0:
        lat = np.array(stats['lat_ms'])
        print('\n' + '='*60)
        print('Real-Time Simulation Summary')
        print('='*60)
        print(f'  Frames processed : {n}')
        print(f'  Total detections : {stats["defects"]}')
        print(f'  Mean latency     : {lat.mean():.1f} ms')
        print(f'  P95  latency     : {np.percentile(lat, 95):.1f} ms')
        print(f'  Mean FPS         : {1000/lat.mean():.1f}')
        print('='*60)


if __name__ == '__main__':
    main()
