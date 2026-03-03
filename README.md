# Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing

Master Thesis Project
Author: Likith
Target Hardware (local/edge): Intel Core i5-1335U | 16 GB RAM | Windows 11 | CPU-only

---

## Project Overview

This project implements a **dual-pipeline** defect inspection system for manufacturing quality control,
trained on three industrial benchmark datasets and deployed on a local CPU (edge device substitute).

### Pipeline A - Object Detection (YOLOv8)
Localize and classify defects using bounding boxes.

| Model | Params | Speed |
|-------|--------|-------|
| YOLOv8n (nano) | ~3 M | Fastest |
| YOLOv8s (small) | ~11 M | Balanced |

### Pipeline B - Image Classification (CNN Transfer Learning)
Classify cropped defect regions or full images.

| Model | Params | Notes |
|-------|--------|-------|
| ResNet-50 | ~25 M | Established baseline |
| EfficientNet-B0 | ~5.3 M | Efficient scaling |
| MobileNetV3-Large | ~5.4 M | Mobile-optimized |

### Combined Pipeline
YOLOv8 detects defect regions -> crops them -> CNN classifies the defect type.

---

### Datasets

| Dataset | Classes | Images | Annotations | Task |
|---------|---------|--------|-------------|------|
| NEU-DET | 6 defect types | 1,550 (1,085/232/233 train/val/test) | PASCAL VOC XML bboxes | Detection + Classification |
| DAGM 2007 | binary (defect / non_defective) | 6,900 (4,830/1,035/1,035 train/val/test) | Binary mask -> derived bbox | Detection + Classification |
| Kolektor KSDD2 | binary (defective / non_defective) | 5,715 (4,667/545/503 train/val/test) | Pixel mask -> derived bbox | Detection + Classification |

### Optimization Formats
| Format | Target |
|--------|--------|
| PyTorch `.pt` / `.pth` (FP32) | Training reference |
| ONNX FP32 | Primary CPU inference |
| ONNX INT8 (dynamic quantized) | Faster CPU inference (operator support issue noted) |
| TFLite FP32 / INT8 | Edge/mobile deployment |
| TensorRT FP16 | Colab GPU benchmarks only |
| Structured pruning (L1, 20%) | Model compression |

---

## Directory Structure

```
thesis_project/
├── colab_notebooks/                    # Run on Google Colab (T4 GPU)
│   ├── 01_data_download_and_preprocessing.ipynb   ← PHASE 2 (run first)
│   ├── 02_yolo_detection_training.ipynb           ← PHASE 3A
│   ├── 03_cnn_classification_training.ipynb       ← PHASE 3B
│   ├── 04_model_optimization.ipynb                ← PHASE 4
│   ├── 04b_tensorrt_benchmarks.ipynb              ← PHASE 4 (GPU benchmarks)
│   └── 05_generate_figures_and_tables.ipynb       ← PHASE 5
├── local_inference/                    # Run on local Windows CPU
│   ├── run_cpu_inference_yolo.py       ← Benchmark all YOLO formats
│   ├── run_cpu_inference_cnn.py        ← Benchmark all CNN formats
│   ├── run_combined_pipeline.py        ← YOLO + CNN end-to-end
│   ├── realtime_simulation.py          ← Simulate conveyor belt
│   ├── generate_local_figures.py       ← Figures 21–33
│   ├── measure_performance.py          ← Latency + throughput benchmarks
│   ├── energy_estimation.py            ← CPU energy proxy (psutil × TDP)
│   └── compare_results.py             ← Aggregate all results
├── datasets/                           # Placeholder — filled by Notebook 01
├── models/
│   ├── yolo/
│   │   ├── full/        ← Best .pt from YOLO training
│   │   ├── onnx/        ← ONNX FP32 exports
│   │   ├── tflite/      ← TFLite FP32 exports
│   │   ├── quantized/   ← ONNX INT8 + TFLite INT8
│   │   └── pruned/      ← Structured-pruned .pt
│   ├── cnn/
│   │   ├── full/        ← Best .pth from CNN training
│   │   ├── onnx/        ← ONNX FP32 exports
│   │   ├── tflite/      ← TFLite FP32 exports
│   │   ├── quantized/   ← ONNX INT8 + TFLite INT8
│   │   └── pruned/      ← Structured-pruned .pth
│   └── combined/        ← YOLO + CNN combined weights
├── results/
│   ├── figures/          ← Publication-ready plots (.png)
│   ├── tables/           ← Summary CSVs (local CPU results)
│   ├── Colab_results/    ← Training/eval results from Colab
│   └── logs/             ← Training histories
├── docs/
│   └── thesis_explanation_document.md
├── requirements_colab.txt
├── requirements_local.txt
└── README.md
```

---

## Execution Guide

### Phase 2 — Data Preparation (Colab)

Upload and run `01_data_download_and_preprocessing.ipynb`.

- Downloads NEU-DET, DAGM, KSDD2 from Kaggle
- Parses PASCAL VOC XML → YOLO format (NEU-DET)
- Derives bounding boxes from masks via `cv2.findContours()` (DAGM, KSDD2)
- Creates both detection (YOLO `.txt` labels) and classification (ImageFolder) formats
- Applies stratified 70/15/15 split
- Saves all outputs to `MyDrive/thesis_project/`

**Prerequisite:** Upload your `kaggle.json` API token before running.

### Phase 3A — YOLO Detection Training (Colab)

Run `02_yolo_detection_training.ipynb`.

- Trains 4 YOLO models: YOLOv8{n,s} × {NEU-DET, DAGM, KSDD2} (6 model-dataset combinations)
- Hyperparameters: epochs=100, patience=20, imgsz=640, batch=16, AdamW (lr0=0.001, lrf=0.01, weight_decay=0.0005), warmup=3 epochs
- Augmentation: mosaic=1.0, fliplr=0.5, flipud=0.0, scale, HSV
- Saves best weights to `models/yolo/full/`
- Estimated time: ~2–3 h on T4 GPU

### Phase 3B — CNN Classification Training (Colab)

Run `03_cnn_classification_training.ipynb`.

- Trains 9 CNN models: {ResNet-50, EfficientNet-B0, MobileNetV3-L} × {NEU-DET, DAGM, KSDD2}
- Two-stage transfer learning: freeze backbone (5 epochs, lr=1e-4) → fine-tune all (50 epochs, lr=1e-5)
- Optimizer: AdamW (weight_decay=1e-4); Scheduler: CosineAnnealingLR; Patience: 10
- KSDD2: WeightedRandomSampler + class-weighted CrossEntropyLoss
- Exports all 9 models to ONNX (opset_version=18)
- Saves best weights to `models/cnn/full/`
- Estimated time: ~2–3 h on T4 GPU

### Phase 4 — Model Optimization (Colab)

Run `04_model_optimization.ipynb` then `04b_tensorrt_benchmarks.ipynb`.

- Exports to ONNX FP32, TFLite FP32/INT8, and ONNX INT8 (dynamic quantization)
- Structured pruning: 20% of filters per conv layer (L1 magnitude), 1-epoch fine-tune
- TensorRT FP16 benchmarks (Colab T4 GPU only — upper-bound reference)
- Validates exported model accuracy ≥ 95% of original

### Phase 5 — Figures & Tables (Colab)

Run `05_generate_figures_and_tables.ipynb` to produce 49 figures and 17+ tables from Colab results.

### Phase 6 — Local CPU Benchmarking

Download `models/` and `results/` folders from Drive to your local machine.

```bash
# Install local dependencies (CPU-only PyTorch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_local.txt

# Benchmark YOLO models on CPU
python local_inference/run_cpu_inference_yolo.py \
    --models_dir models/yolo --datasets_dir datasets/ --results_dir results/

# Benchmark CNN models on CPU
python local_inference/run_cpu_inference_cnn.py \
    --models_dir models/cnn --datasets_dir datasets/ --results_dir results/

# Run combined pipeline benchmark (all combinations)
python local_inference/run_combined_pipeline.py --all \
    --yolo_dir models/yolo --cnn_dir models/cnn \
    --datasets_dir datasets/ --results_dir results/

# Measure latency + throughput
python local_inference/measure_performance.py \
    --models_dir models/ --results_dir results/

# Estimate energy consumption
python local_inference/energy_estimation.py \
    --models_dir models/ --results_dir results/

# Simulate real-time conveyor belt inspection
python local_inference/realtime_simulation.py --mode combined

# Aggregate and compare all results
python local_inference/compare_results.py --results_dir results/

# Generate local CPU figures
python local_inference/generate_local_figures.py \
    --results_dir results/ --output_dir results/figures/
```

---

## Metrics Recorded

| Metric | Pipeline | Where measured |
|--------|----------|----------------|
| mAP@50, mAP@50-95 | YOLO | Colab (test set) |
| Precision, Recall, F1 | Both | Colab (test set) |
| Top-1 Accuracy, macro-F1 | CNN | Colab (test set) |
| Latency (ms/image): mean, std, p50, p95 | Both | Local CPU |
| Throughput (FPS) | Both | Local CPU |
| Model size (MB) | Both | Local |
| Energy proxy (Wh/1000 frames) | Both | Local CPU (psutil × TDP) |
| Load time (ms) | Both | Local CPU |

---

## Hardware Reference

| Component | Spec |
|-----------|------|
| CPU | 13th Gen Intel Core i5-1335U, 1.30 GHz |
| TDP (sustained) | 15 W |
| RAM | 16 GB |
| OS | Windows 11 Home 64-bit |
| GPU | None (CPU-only inference) |
| Colab GPU | T4 (training + TensorRT benchmarks) |

---

## Latest Results Snapshot

Tables below are synchronized with the current CSV outputs in `results/tables/` and `results/Colab_results/results/tables/`.

### YOLO Detection Accuracy (Colab Test Set)

| Dataset | Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|---------|-------|--------|-----------|-----------|--------|-----|
| NEU-DET | YOLOv8n | 0.7235 | 0.4000 | 0.6336 | 0.7187 | 0.6735 |
| NEU-DET | YOLOv8s | 0.7191 | 0.4047 | 0.6812 | 0.6813 | 0.6813 |
| DAGM | YOLOv8n | 0.9910 | 0.6363 | 0.9641 | 1.0000 | 0.9817 |
| DAGM | YOLOv8s | 0.9931 | 0.6223 | 0.9674 | 0.9898 | 0.9785 |
| KSDD2 | YOLOv8n | 0.6673 | 0.3552 | 0.7466 | 0.6481 | 0.6939 |
| KSDD2 | YOLOv8s | 0.6422 | 0.3417 | 0.6447 | 0.7055 | 0.6737 |

### CNN Classification Accuracy (Colab Test Set)

| Dataset | Model | Test Accuracy | Test F1 (macro) | Train Time (min) |
|---------|-------|--------------|-----------------|------------------|
| NEU-DET | ResNet-50 | 99.88% | 0.9990 | 34.9 |
| NEU-DET | EfficientNet-B0 | 98.51% | 0.9855 | 24.5 |
| NEU-DET | MobileNetV3-L | 98.63% | 0.9857 | 13.8 |
| DAGM | ResNet-50 | 99.48% | 0.9882 | 47.8 |
| DAGM | EfficientNet-B0 | 86.29% | 0.7499 | 27.9 |
| DAGM | MobileNetV3-L | 96.34% | 0.9159 | 32.2 |
| KSDD2 | ResNet-50 | 97.21% | 0.9328 | 49.2 |
| KSDD2 | EfficientNet-B0 | 92.61% | 0.8436 | 46.1 |
| KSDD2 | MobileNetV3-L | 93.81% | 0.8673 | 65.9 |

### Local CPU Inference - YOLO (successful runs)

| Dataset | Model | Format | Latency (ms) | FPS | Size (MB) | Energy (Wh/1000) |
|---------|-------|--------|--------------|-----|-----------|------------------|
| NEU-DET | YOLOv8n | PyTorch | 55.67 | 18.0 | 6.26 | 0.339 |
| NEU-DET | YOLOv8n | ONNX FP32 | 37.91 | 26.4 | 12.31 | 0.242 |
| NEU-DET | YOLOv8n | TFLite FP32 | 79.57 | 12.6 | 12.28 | 0.092 |
| NEU-DET | YOLOv8s | PyTorch | 161.62 | 6.2 | 22.53 | 0.823 |
| NEU-DET | YOLOv8s | ONNX FP32 | 131.00 | 7.6 | 44.80 | 0.618 |
| NEU-DET | YOLOv8s | TFLite FP32 | 97.19 | 10.3 | 12.28 | 0.030 |
| DAGM | YOLOv8n | PyTorch | 69.36 | 14.4 | 6.25 | 0.146 |
| DAGM | YOLOv8n | ONNX FP32 | 43.64 | 22.9 | 12.31 | 0.102 |
| DAGM | YOLOv8n | TFLite FP32 | 78.04 | 12.8 | 12.28 | 0.033 |
| DAGM | YOLOv8s | PyTorch | 181.93 | 5.5 | 22.52 | 0.401 |
| DAGM | YOLOv8s | ONNX FP32 | 132.71 | 7.5 | 44.79 | 0.337 |
| DAGM | YOLOv8s | TFLite FP32 | 82.38 | 12.1 | 12.28 | 0.031 |
| KSDD2 | YOLOv8n | PyTorch | 80.36 | 12.4 | 6.25 | 0.161 |
| KSDD2 | YOLOv8n | ONNX FP32 | 52.19 | 19.2 | 12.31 | 0.125 |
| KSDD2 | YOLOv8n | TFLite FP32 | 84.08 | 11.9 | 12.28 | 0.030 |
| KSDD2 | YOLOv8s | PyTorch | 200.22 | 5.0 | 22.52 | 0.498 |
| KSDD2 | YOLOv8s | ONNX FP32 | 142.15 | 7.0 | 44.79 | 0.419 |
| KSDD2 | YOLOv8s | TFLite FP32 | 83.95 | 11.9 | 12.28 | 0.033 |

*All YOLO ONNX INT8 runs fail on this CPU with `ConvInteger(10) NOT_IMPLEMENTED`.*

### Local CPU Inference - CNN (successful runs)

| Dataset | Model | Format | Latency (ms) | FPS | Size (MB) | Energy (Wh/1000) |
|---------|-------|--------|--------------|-----|-----------|------------------|
| NEU-DET | ResNet-50 | ONNX FP32 | 33.79 | 29.6 | 94.17 | 0.104 |
| NEU-DET | ResNet-50 | TFLite FP32 | 83.58 | 12.0 | 94.00 | 0.032 |
| NEU-DET | ResNet-50 | TFLite INT8 | 28.71 | 34.8 | 23.92 | 0.010 |
| NEU-DET | EfficientNet-B0 | ONNX FP32 | 9.78 | 102.2 | 16.53 | 0.025 |
| NEU-DET | EfficientNet-B0 | TFLite FP32 | 12.81 | 78.1 | 16.03 | 0.005 |
| NEU-DET | EfficientNet-B0 | TFLite INT8 | 46.01 | 21.7 | 4.53 | 0.016 |
| NEU-DET | MobileNetV3-L | ONNX FP32 | 5.37 | 186.1 | 17.10 | 0.012 |
| NEU-DET | MobileNetV3-L | TFLite FP32 | 6.08 | 164.6 | 16.83 | 0.002 |
| NEU-DET | MobileNetV3-L | TFLite INT8 | 20.87 | 47.9 | 4.53 | 0.008 |
| DAGM | ResNet-50 | ONNX FP32 | 33.98 | 29.4 | 94.15 | 0.108 |
| DAGM | ResNet-50 | TFLite FP32 | 82.75 | 12.1 | 93.97 | 0.051 |
| DAGM | ResNet-50 | TFLite INT8 | 28.67 | 34.9 | 23.91 | 0.012 |
| DAGM | EfficientNet-B0 | ONNX FP32 | 9.84 | 101.6 | 16.52 | 0.026 |
| DAGM | EfficientNet-B0 | TFLite FP32 | 12.61 | 79.3 | 16.02 | 0.006 |
| DAGM | EfficientNet-B0 | TFLite INT8 | 45.55 | 22.0 | 4.52 | 0.019 |
| DAGM | MobileNetV3-L | ONNX FP32 | 5.34 | 187.3 | 17.08 | 0.011 |
| DAGM | MobileNetV3-L | TFLite FP32 | 6.11 | 163.6 | 16.82 | 0.002 |
| DAGM | MobileNetV3-L | TFLite INT8 | 20.89 | 47.9 | 4.52 | 0.009 |
| KSDD2 | ResNet-50 | ONNX FP32 | 33.83 | 29.6 | 94.15 | 0.126 |
| KSDD2 | ResNet-50 | TFLite FP32 | 83.19 | 12.0 | 93.97 | 0.029 |
| KSDD2 | ResNet-50 | TFLite INT8 | 28.77 | 34.8 | 23.91 | 0.010 |
| KSDD2 | EfficientNet-B0 | ONNX FP32 | 9.93 | 100.7 | 16.52 | 0.024 |
| KSDD2 | EfficientNet-B0 | TFLite FP32 | 12.52 | 79.9 | 16.02 | 0.004 |
| KSDD2 | EfficientNet-B0 | TFLite INT8 | 45.71 | 21.9 | 4.52 | 0.017 |
| KSDD2 | MobileNetV3-L | ONNX FP32 | 5.35 | 186.8 | 17.08 | 0.010 |
| KSDD2 | MobileNetV3-L | TFLite FP32 | 6.04 | 165.5 | 16.82 | 0.002 |
| KSDD2 | MobileNetV3-L | TFLite INT8 | 20.91 | 47.8 | 4.52 | 0.009 |

*All CNN ONNX INT8 runs fail on this CPU with `ConvInteger(10) NOT_IMPLEMENTED`.*

### Combined Pipeline (YOLOv8n + CNN, PyTorch, local CPU)

| Dataset | CNN Model | E2E Latency (ms) | FPS | Avg Detections | Det % | Cls % |
|---------|-----------|-----------------|-----|----------------|-------|-------|
| NEU-DET | MobileNetV3-L | 104.33 | 9.58 | 2.10 | 63.8% | 36.0% |
| NEU-DET | EfficientNet-B0 | 140.00 | 7.14 | 2.10 | 54.0% | 45.9% |
| NEU-DET | ResNet-50 | 157.14 | 6.36 | 2.10 | 34.9% | 64.9% |
| DAGM | MobileNetV3-L | 82.19 | 12.17 | 0.08 | 97.5% | 2.3% |
| DAGM | EfficientNet-B0 | 81.40 | 12.28 | 0.08 | 96.9% | 2.9% |
| DAGM | ResNet-50 | 82.95 | 12.06 | 0.08 | 93.0% | 6.8% |
| KSDD2 | MobileNetV3-L | 83.68 | 11.95 | 0.14 | 96.0% | 3.8% |
| KSDD2 | EfficientNet-B0 | 82.92 | 12.06 | 0.14 | 94.2% | 5.5% |
| KSDD2 | ResNet-50 | 88.36 | 11.32 | 0.14 | 88.1% | 11.7% |

### TensorRT GPU Reference Benchmarks (Colab T4, FP16)

| Model | Type | Mean Latency (ms) | FPS |
|-------|------|-------------------|-----|
| YOLOv8n | YOLO | 3.49-4.89 | 205-287 |
| YOLOv8s | YOLO | 4.55-5.05 | 198-220 |
| ResNet-50 | CNN | 3.63-3.74 | 267-275 |
| EfficientNet-B0 | CNN | 1.75-2.03 | 493-571 |
| MobileNetV3-L | CNN | 1.17-1.30 | 768-854 |

### Format Comparison Summary (mean across datasets)

| Pipeline | Format | Mean mAP50 / Accuracy | Mean Latency (ms) | Mean FPS | Mean Size (MB) | Mean Energy (Wh/1000) | Notes |
|----------|--------|----------------------|-------------------|----------|----------------|-----------------------|-------|
| YOLO | PyTorch | 0.668 | 124.86 | 10.25 | 14.39 | 0.3948 | Reference training/export format. |
| YOLO | ONNX FP32 | 0.668 | 89.93 | 15.10 | 28.55 | 0.3072 | Fastest successful YOLO CPU format with no mean mAP loss vs PyTorch. |
| YOLO | TFLite FP32 | 0.227 | 84.20 | 11.93 | 12.28 | 0.0414 | Large accuracy drop after conversion. |
| YOLO | ONNX INT8 | 0.636 | n/a | n/a | 7.47 | 3.7197 | Runtime error on this CPU (`ConvInteger(10)`). |
| CNN | ONNX FP32 | 0.959 | 16.36 | 105.92 | 42.59 | 0.0496 | Best overall local CPU throughput. |
| CNN | TFLite FP32 | 0.959 | 33.97 | 85.23 | 42.28 | 0.0148 | Lower energy than ONNX FP32. |
| CNN | TFLite INT8 | 0.954 | 31.79 | 34.86 | 10.99 | 0.0122 | Small accuracy drop with about 4x smaller models. |
| CNN | ONNX INT8 | 0.813 | n/a | n/a | 10.93 | 0.5904 | Runtime error on this CPU (`ConvInteger(10)`). |

### Best Models per Dataset (weighted ranking)

| Pipeline | Dataset | Best Model | Format | Key Metric | Latency (ms) | FPS | Energy (Wh/1000) |
|----------|---------|------------|--------|-----------|--------------|-----|------------------|
| YOLO | DAGM | YOLOv8n | PyTorch | mAP50 = 0.9950 | 69.36 | 14.4 | 0.146 |
| YOLO | NEU-DET | YOLOv8n | PyTorch | mAP50 = 0.7148 | 55.67 | 18.0 | 0.339 |
| YOLO | KSDD2 | YOLOv8n | PyTorch | mAP50 = 0.3274 | 80.36 | 12.4 | 0.161 |
| CNN | DAGM | MobileNetV3-L | ONNX FP32 | Acc = 96.34% | 5.34 | 187.3 | 0.011 |
| CNN | NEU-DET | MobileNetV3-L | ONNX FP32 | Acc = 98.63% | 5.37 | 186.1 | 0.012 |
| CNN | KSDD2 | MobileNetV3-L | ONNX FP32 | Acc = 93.81% | 5.35 | 186.8 | 0.010 |

### Key Findings
- MobileNetV3-L in ONNX FP32 is the strongest local CPU classifier: 186-187 FPS at 5.34-5.37 ms with 93.81-98.63% accuracy.
- YOLO ONNX FP32 is the fastest successful local detection format: mean mAP@50 stays at 0.668 while mean latency drops from 124.86 ms (PyTorch) to 89.93 ms.
- DAGM is the easiest detection dataset (Colab mAP@50 up to 0.9931); KSDD2 remains the hardest (0.6422-0.6673 on Colab test evaluation).
- The fastest PyTorch combined pipeline per dataset is YOLOv8n + MobileNetV3-L on NEU-DET (104.33 ms), YOLOv8n + EfficientNet-B0 on DAGM (81.40 ms), and YOLOv8n + EfficientNet-B0 on KSDD2 (82.92 ms).
- TFLite INT8 CNN exports keep mean accuracy close to FP32 (0.954 vs 0.959) while reducing mean model size from 42.28 MB to 10.99 MB.
- YOLO TFLite export causes a severe mean mAP drop (0.668 -> 0.227), indicating a conversion/post-processing problem rather than a training problem.
- ONNX INT8 fails on this CPU for all 15 models (6 YOLO and 9 CNN) because `ConvInteger(10)` is not implemented in the tested runtime/provider path.
- TensorRT FP16 on Colab T4 provides the upper-bound reference: 198-287 FPS for YOLO and 267-854 FPS for CNNs.

---

## Results Source Map

Results come from two complementary environments:

| Source | What it measures | Hardware | Key tables |
|--------|-----------------|----------|------------|
| **Colab** | Training/test accuracy, mAP, F1, GPU benchmarks | T4 GPU | `TABLE_YOLO_DetectionBenchmark_All.csv`, `TABLE_CNN_ClassificationBenchmark_All.csv`, `TABLE_TensorRT_Benchmarks.csv` |
| **Local** | CPU latency, FPS, energy, combined pipeline | i5-1335U | `master_results_yolo.csv`, `master_results_cnn.csv`, `master_results_combined.csv`, `best_models_summary.csv`, `performance_all_models.csv`, `energy_estimation.csv` |

- Colab tables/figures: `results/Colab_results/results/{tables,figures}/`
- Local tables/figures: `results/{tables,figures}/`

---

## Reproducibility Path

Exact command order to regenerate all results from trained models:

```bash
# 1. Install local dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_local.txt

# 2. Run YOLO CPU inference benchmarks
python local_inference/run_cpu_inference_yolo.py \
    --models_dir models/yolo --datasets_dir datasets/ --results_dir results/

# 3. Run CNN CPU inference benchmarks
python local_inference/run_cpu_inference_cnn.py \
    --models_dir models/cnn --datasets_dir datasets/ --results_dir results/

# 4. Run combined pipeline benchmarks
python local_inference/run_combined_pipeline.py --all \
    --yolo_dir models/yolo --cnn_dir models/cnn \
    --datasets_dir datasets/ --results_dir results/

# 5. Measure latency and throughput
python local_inference/measure_performance.py \
    --models_dir models/ --results_dir results/

# 6. Estimate energy consumption
python local_inference/energy_estimation.py \
    --models_dir models/ --results_dir results/

# 7. Aggregate all results into master tables
python local_inference/compare_results.py --results_dir results/

# 8. Generate local CPU figures (fig21–fig33)
python local_inference/generate_local_figures.py \
    --results_dir results/ --output_dir results/figures/
```

Steps must run in order (later scripts depend on earlier outputs).

---

## Repository Modes

### Full Research Workspace (~9 GB)
Contains everything: models, datasets, venv, training runs, all results. Required for reproducing inference benchmarks locally.

### Lightweight Git-Upload Version
For submission/sharing, exclude large binary assets via `.gitignore`:
- `venv/` (~2.9 GB) — recreate with `pip install -r requirements_local.txt`
- `datasets/` (~2.4 GB) — regenerate with Colab notebook 01
- `models/` (~3.6 GB) — download from Google Drive or retrain
- `runs/` (~90 MB) — training artifacts, not needed for inference

Keep: scripts, notebooks, docs, requirements, result CSVs/figures, README.

---

## Known Issues

1. **ONNX INT8 `ConvInteger` not implemented:** All 15 ONNX INT8 models (6 YOLO + 9 CNN) fail at runtime with `NOT_IMPLEMENTED: Could not find an implementation for ConvInteger(10)`. This is an ONNX Runtime CPU execution provider limitation. These models are excluded from latency benchmarks and marked `status=error` in `performance_all_models.csv`.

2. **YOLO TFLite accuracy degradation:** YOLO models exported to TFLite via `onnx-tf` show near-zero mAP on DAGM and KSDD2 (only NEU-DET retains partial accuracy; format-averaged mAP drops from 0.668 to 0.227). This points to a conversion/post-processing issue rather than a training issue.

3. **DAGM class subset:** Only classes 1-6 of 10 are used, due to Google Drive storage limits.

4. **KSDD2 class imbalance:** The non-defective:defective ratio is about 12.5:1, which makes detection materially harder. Classification mitigations include `WeightedRandomSampler` and class-weighted loss.

5. **Combined pipeline ONNX latency:** When YOLO runs through ONNX Runtime inside the combined pipeline wrapper, end-to-end latency is much higher than PyTorch (for example, NEU-DET with ResNet-50 is about 665 ms vs 157 ms).

---

## Requirements

- **Colab:** `requirements_colab.txt` - install via `!pip install -r requirements_colab.txt`
- **Local:** `requirements_local.txt` - CPU-only, install CPU PyTorch first

---

## Citation / Acknowledgements

- NEU Surface Defect Database: Song and Yan (2013)
- DAGM 2007: German Working Group on Pattern Recognition
- Kolektor KSDD2: Bozic et al. (2021)
- YOLOv8: Ultralytics (2023)
- Transfer learning backbones: torchvision / PyTorch Hub (ImageNet pretrained)
