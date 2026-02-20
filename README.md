# Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing

Master Thesis Project
Author: Likith
Target Hardware (local/edge): Intel Core i5-1335U · 16 GB RAM · Windows 11 · CPU-only

---

## Project Overview

This project implements a **dual-pipeline** defect inspection system for manufacturing quality control,
trained on three industrial benchmark datasets and deployed on a local CPU (edge device substitute).

### Pipeline A — Object Detection (YOLOv8)
Localize and classify defects using bounding boxes.

| Model | Params | Speed |
|-------|--------|-------|
| YOLOv8n (nano) | ~3 M | Fastest |
| YOLOv8s (small) | ~11 M | Balanced |

### Pipeline B — Image Classification (CNN Transfer Learning)
Classify cropped defect regions or full images.

| Model | Params | Notes |
|-------|--------|-------|
| ResNet-50 | ~25 M | Established baseline |
| EfficientNet-B0 | ~5.3 M | Efficient scaling |
| MobileNetV3-Large | ~5.4 M | Mobile-optimized |

### Combined Pipeline
YOLOv8 detects defect regions → crops them → CNN classifies the defect type.

---

### Datasets

| Dataset | Classes | Images | Annotations | Task |
|---------|---------|--------|-------------|------|
| NEU-DET | 6 defect types (detection) / 5 (classification) | ~1,800 | PASCAL VOC XML bboxes | Detection + Classification |
| DAGM 2007 | binary (defective / non-defective) | ~7,000 (classes 1–6) | Mask → derived bbox | Detection + Classification |
| Kolektor KSDD2 | binary (defective / non-defective) | ~3,335 | Pixel mask → derived bbox | Detection + Classification |

### Optimization Formats
| Format | Target |
|--------|--------|
| PyTorch `.pt` (FP32) | Training reference |
| ONNX FP32 | Primary CPU inference |
| ONNX INT8 (dynamic quantized) | Faster CPU inference |
| TFLite FP32 / INT8 | Edge/mobile deployment |
| TensorRT FP16 / INT8 | Colab GPU benchmarks only |
| Structured pruning | Model compression |

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
│   ├── generate_local_figures.py       ← Figures 21–35
│   ├── measure_performance.py          ← Latency + throughput benchmarks
│   ├── energy_estimation.py            ← CPU energy proxy (psutil × TDP)
│   └── compare_results.py             ← Aggregate all results
├── datasets/                           # Placeholder — filled by Notebook 01
├── models/
│   ├── yolo/
│   │   ├── full/        ← Best .pt from YOLO training
│   │   ├── onnx/        ← ONNX exports
│   │   ├── tflite/      ← TFLite exports
│   │   ├── tensorrt/    ← TensorRT engines (Colab only)
│   │   ├── quantized/   ← INT8 quantized
│   │   └── pruned/
│   ├── cnn/
│   │   ├── full/        ← Best .pth from CNN training
│   │   ├── onnx/        ← ONNX exports
│   │   ├── tflite/      ← TFLite exports
│   │   ├── quantized/   ← INT8 quantized
│   │   └── pruned/
│   └── combined/        ← YOLO + CNN combined weights
├── results/
│   ├── figures/          ← Publication-ready plots (.png)
│   ├── tables/           ← Summary CSVs
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

- Trains 6 YOLO models: YOLOv8{n,s} × {NEU-DET, DAGM, KSDD2}
- Hyperparameters: epochs=100, patience=15, imgsz=640, batch=32, AdamW
- Saves best weights to `models/yolo/full/`
- Estimated time: ~2–3 h on T4 GPU

### Phase 3B — CNN Classification Training (Colab)

Run `03_cnn_classification_training.ipynb`.

- Trains 9 CNN models: {ResNet50, EfficientNet-B0, MobileNetV3} × {NEU-DET, DAGM, KSDD2}
- Transfer learning from ImageNet, replace classification head
- KSDD2: WeightedRandomSampler + class-weighted CrossEntropyLoss
- Saves best weights to `models/cnn/full/`
- Estimated time: ~2–3 h on T4 GPU

### Phase 4 — Model Optimization (Colab)

Run `04_model_optimization.ipynb` then `04b_tensorrt_benchmarks.ipynb`.

- Exports to ONNX, TFLite, and quantized (INT8) formats
- TensorRT benchmarks (Colab T4 GPU only — upper-bound reference)
- Validates exported model accuracy ≥ 95% of original

### Phase 5 — Figures & Tables (Colab)

Run `05_generate_figures_and_tables.ipynb` to produce figures and tables from Colab results.

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
| Top-1 Accuracy | CNN | Colab (test set) |
| Latency (ms/image) | Both | Local CPU |
| Throughput (FPS) | Both | Local CPU |
| Model size (MB) | Both | Local |
| Energy proxy (mJ/image) | Both | Local CPU (psutil × TDP) |

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

## Current Results Snapshot

### Best YOLO Detection Models (Local CPU, i5-1335U)

| Dataset | Model | mAP@50 | Latency (ms) | FPS | Energy (Wh/1000) |
|---------|-------|--------|-------------|-----|-------------------|
| DAGM | YOLOv8n (PyTorch) | 0.995 | 69.4 | 14.4 | 0.146 |
| NEU-DET | YOLOv8n (PyTorch) | 0.715 | 55.7 | 18.0 | 0.339 |
| KSDD2 | YOLOv8n (PyTorch) | 0.327 | 80.4 | 12.4 | 0.161 |

### Best CNN Classification Models (Local CPU, i5-1335U)

| Dataset | Model | Accuracy | Latency (ms) | FPS | Energy (Wh/1000) |
|---------|-------|----------|-------------|-----|-------------------|
| DAGM | MobileNetV3-L (ONNX FP32) | 96.34% | 5.3 | 187 | 0.011 |
| NEU-DET | MobileNetV3-L (ONNX FP32) | 98.63% | 5.4 | 186 | 0.012 |
| KSDD2 | MobileNetV3-L (ONNX FP32) | 93.81% | 5.4 | 187 | 0.010 |

### Key Findings
- MobileNetV3-L in ONNX FP32 is the best CPU edge model: 186 FPS with 93–99% accuracy
- Combined pipeline (YOLO+CNN) achieves <100 ms on DAGM/KSDD2 with MobileNetV3-L
- TensorRT FP16 on T4 GPU is 10–50x faster than CPU (reference upper bound)

---

## Results Source Map

Results come from two complementary environments:

| Source | What it measures | Hardware | Key tables |
|--------|-----------------|----------|------------|
| **Colab** | Training/test accuracy, mAP, F1, GPU benchmarks | T4 GPU | `TABLE_YOLO_DetectionBenchmark_All.csv`, `TABLE_CNN_ClassificationBenchmark_All.csv`, `TABLE_TensorRT_Benchmarks.csv` |
| **Local** | CPU latency, FPS, energy, combined pipeline | i5-1335U | `master_results_yolo.csv`, `master_results_cnn.csv`, `master_results_combined.csv`, `best_models_summary.csv` |

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

1. **ONNX INT8 `ConvInteger` not implemented:** All 15 ONNX INT8 models fail at runtime with `NOT_IMPLEMENTED: Could not find an implementation for ConvInteger(10)`. This is an ONNX Runtime CPU execution provider limitation — the `ConvInteger` operator from dynamic quantization is not supported on the test CPU. These models are excluded from latency benchmarks.

2. **YOLO TFLite accuracy degradation:** YOLO models exported to TFLite via `onnx-tf` show near-zero mAP on DAGM and KSDD2 (only NEU-DET retains partial accuracy). This is a format conversion issue with YOLO post-processing layers, not a model quality issue.

3. **DAGM class subset:** Only classes 1–6 of 10 are used, due to Google Drive storage limits (~15 GB free).

4. **KSDD2 class imbalance:** 12.5:1 non-defective:defective ratio limits detection mAP. Mitigated with `WeightedRandomSampler` and class-weighted loss for classification.

---

## Requirements

- **Colab:** `requirements_colab.txt` — install via `!pip install -r requirements_colab.txt`
- **Local:** `requirements_local.txt` — CPU-only, install with CPU PyTorch first

---

## AI Usage Disclosure

AI tools (Claude Code by Anthropic) were used for code scaffolding, experiment orchestration, and writing support. All code was manually reviewed and all empirical results were produced by running actual models. See [`docs/ai_usage_statement.md`](docs/ai_usage_statement.md) for details.

---

## Citation / Acknowledgements

- NEU Surface Defect Database: Song & Yan (2013)
- DAGM 2007: German Working Group on Pattern Recognition
- Kolektor KSDD2: Božič et al. (2021)
- YOLOv8: Ultralytics (2023)
- Transfer learning backbones: torchvision / PyTorch Hub (ImageNet pretrained)
