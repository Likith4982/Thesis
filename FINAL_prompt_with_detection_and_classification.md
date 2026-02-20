# Master Thesis — Complete Prompt for Claude Code Extension
# BOTH Pipelines: YOLOv8 Detection + CNN Classification
# Phase 1 (project structure) already done — start from Phase 2

## Copy everything below the line and paste into Claude Code in VS Code

---

You are helping me complete my Master Thesis project end-to-end. The topic is **"Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing"**. Read this ENTIRE prompt carefully before writing any code or creating any files.

## PROJECT OVERVIEW

This project implements a **dual-pipeline** defect inspection system for manufacturing:

**Pipeline A — Detection (YOLOv8):** Localize and classify defects using bounding boxes. Uses YOLOv8 variants (YOLOv8n, YOLOv8s, YOLOv8m) for object detection on all 3 datasets.

**Pipeline B — Classification (Transfer Learning CNNs):** Classify entire images or cropped defect regions using transfer learning with ResNet50, EfficientNet-B0, and MobileNetV3-Small on all 3 datasets.

**Combined Pipeline:** YOLO detects defect regions → crops them → CNN classifier classifies the defect type. This two-stage approach is also evaluated.

Both pipelines are trained on Google Colab (T4 GPU), optimized (quantization + pruning), and deployed on my local CPU (substitute for edge device).

### Datasets:
1. **NEU Surface Defect Database** (NEU-DET version with bounding box annotations) — 6 defect classes
2. **DAGM 2007** — texture defects, binary (defective/non-defective), bounding boxes derived from masks
3. **Kolektor KSDD2** — metallic surface defects, binary classification, bounding boxes derived from segmentation masks

### Models:
- **Detection:** YOLOv8n (nano), YOLOv8s (small), YOLOv8m (medium)
- **Classification:** ResNet50, EfficientNet-B0, MobileNetV3-Small (all with ImageNet transfer learning)

## CRITICAL CONSTRAINTS

1. **Training on Google Colab (T4 GPU, Colab Pro).** Generate complete `.ipynb` notebooks that I upload to Colab and run manually. I am NOT connecting VS Code to Colab.
2. **All Colab notebooks must:**
   - Mount Google Drive at the start: `from google.colab import drive; drive.mount('/content/drive')`
   - Save ALL outputs (models, metrics, figures, logs) to `drive/MyDrive/thesis_project/`
   - Include **checkpoint-resume logic** — if Colab disconnects mid-training, re-running skips already-completed models
   - Include `!pip install` cells at top for dependencies not pre-installed on Colab
   - Print clear summary of what was saved and where at the end
3. **Inference on my local Windows machine (CPU only):**
   - Processor: 13th Gen Intel Core i5-1335U (1.30 GHz), TDP ~15W
   - RAM: 16 GB, Windows 11, 64-bit, No GPU
   - My CPU replaces Raspberry Pi / Jetson Nano as edge device
4. **T4 GPU is sufficient** — batch size 32 (reduce to 16 if OOM).
5. **Drive space may be limited** — download datasets efficiently. If DAGM is too large, use classes 1–6 and document the choice.
6. **Seeds: 42 everywhere** — `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` for reproducibility.

## PROJECT STRUCTURE (Phase 1 — Already Done)

```
thesis_project/
├── colab_notebooks/
│   ├── 01_data_download_and_preprocessing.ipynb
│   ├── 02_yolo_detection_training.ipynb
│   ├── 03_cnn_classification_training.ipynb
│   ├── 04_model_optimization.ipynb
│   ├── 04b_tensorrt_benchmarks.ipynb   (TensorRT runs on Colab T4 GPU only)
│   └── 05_generate_figures_and_tables.ipynb
├── local_inference/
│   ├── run_cpu_inference_yolo.py
│   ├── run_cpu_inference_cnn.py
│   ├── run_combined_pipeline.py
│   ├── measure_performance.py
│   ├── energy_estimation.py
│   ├── compare_results.py
│   ├── realtime_simulation.py
│   └── generate_local_figures.py
├── datasets/
├── models/
│   ├── yolo/
│   │   ├── full/
│   │   ├── onnx/
│   │   ├── tflite/
│   │   ├── quantized/
│   │   └── pruned/
│   ├── cnn/
│   │   ├── full/
│   │   ├── onnx/
│   │   ├── tflite/
│   │   ├── quantized/
│   │   └── pruned/
│   └── combined/
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── docs/
│   └── thesis_explanation_document.md
├── requirements_colab.txt
├── requirements_local.txt
└── README.md
```

## GOOGLE DRIVE OUTPUT STRUCTURE (created by Colab notebooks)

```
MyDrive/thesis_project/
├── datasets/
│   ├── NEU-DET/           (images + YOLO-format annotations)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── data.yaml
│   ├── DAGM/              (same structure)
│   └── KSDD2/             (same structure)
├── dataset_metadata/
│   ├── neu_classification_splits.csv
│   ├── neu_detection_splits.csv
│   ├── dagm_splits.csv
│   └── ksdd2_splits.csv
├── models/
│   ├── yolo/
│   │   ├── full/          (YOLOv8n/s/m trained weights for each dataset)
│   │   ├── onnx/          (ONNX exports)
│   │   ├── tflite/        (TFLite FP32 + INT8 exports)
│   │   ├── tensorrt/      (TensorRT engines — Colab T4 only)
│   │   ├── quantized/     (ONNX quantized)
│   │   └── pruned/
│   ├── cnn/
│   │   ├── full/          (9 .pth files)
│   │   ├── onnx/          (9 .onnx files + quantized ONNX)
│   │   ├── tflite/        (TFLite FP32 + INT8)
│   │   ├── quantized/     (PyTorch dynamic/static quantized)
│   │   └── pruned/
│   └── combined/
├── training_logs/
│   ├── yolo/
│   ├── cnn/
│   └── tensorboard/
├── results/
│   ├── figures/
│   ├── tables/
│   └── tensorrt_benchmarks/  (GPU-only benchmarks from Colab)
└── checkpoints/
```

---

# ============================================================
# PHASE 2: DATASETS — Download, Annotate, Convert, Augment
# ============================================================

**Create: `colab_notebooks/01_data_download_and_preprocessing.ipynb`**

This is the most critical notebook — all subsequent work depends on it.

## 2.1 Setup Cell
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Create directory structure
import os
BASE = '/content/drive/MyDrive/thesis_project'
# ... create all subdirs

# Install dependencies
!pip install ultralytics albumentations kaggle

# Set seeds
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```

## 2.2 NEU Surface Defect Database (NEU-DET)

**This dataset has TWO versions — we need the DET version with bounding box annotations.**

- NEU-DET: ~1800 images across 6 classes with XML/PASCAL VOC bounding box annotations
- Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratched
- Image size: 200×200, grayscale

**Download sources (try in order):**
1. Kaggle: `kaustubhdikshit/neu-surface-defect-database` or search for "NEU-DET"
2. Direct academic mirror
3. Manual download fallback instructions

**Processing steps:**
1. Download and extract
2. Parse PASCAL VOC XML annotations → extract bounding boxes
3. Convert annotations to **YOLO format** (class_id, x_center, y_center, width, height — all normalized 0-1)
4. Also create a **classification version** (crop each bounding box region, save as separate images per class; also keep full images labeled by dominant class)
5. Split 70/15/15 (stratified) into train/val/test
6. Create YOLO `data.yaml` file:
```yaml
path: /content/drive/MyDrive/thesis_project/datasets/NEU-DET
train: images/train
val: images/val
test: images/test
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratched']
```
7. Save split metadata as CSV

## 2.3 DAGM 2007 Dataset

- 10 texture classes, each with ~1000 defect-free and ~150 defective images
- Defective images have elliptical mask labels (not bounding boxes)
- Image size: 512×512, grayscale

**Download sources:**
1. Kaggle: `mhskjelvareid/dagm-2007-competition-dataset-optical-inspection`
2. Alternative Kaggle mirrors

**If dataset > 2GB, use only classes 1–6 (document this decision in a markdown cell).**

**Processing steps:**
1. Download and extract
2. For defective images: read the corresponding label files (CSV with ellipse parameters or mask images)
3. **Derive bounding boxes from masks/ellipses:**
   - If mask images exist: `cv2.findContours()` → `cv2.boundingRect()` → YOLO format
   - If ellipse parameters: convert ellipse to bounding rectangle
4. Create two versions:
   - **Detection version (YOLO format):** defective images with bounding box labels. Non-defective images included with empty label files (YOLO handles this as background).
   - **Classification version:** binary — defective (1) vs non-defective (0) per image. Since there are 10 classes of texture, treat each class separately OR merge into a single binary task. **Decision: Merge all classes into a single binary defect/no-defect classification task** — this is more realistic for manufacturing and gives us a cleaner dataset.
5. Split 70/15/15, create `data.yaml`, save metadata CSV

## 2.4 Kolektor Surface Defect Dataset (KSDD2)

- ~3335 images (246 defective, ~3089 non-defective) — heavily imbalanced
- Pixel-level segmentation masks for defective images
- Image size varies (approx. 230×630), grayscale

**Download sources:**
1. Official: https://www.vicos.si/resources/kolektorsdd2/
2. Kaggle mirrors (search "KSDD2" or "Kolektor")

**Processing steps:**
1. Download and extract
2. For defective images with masks:
   - Derive bounding boxes: `cv2.findContours()` on binary mask → `cv2.boundingRect()`
   - Convert to YOLO format (single class: "defect")
3. Non-defective images get empty label files
4. **For classification:** binary — defective (1) vs non-defective (0)
5. **Handle class imbalance:**
   - For detection: YOLO handles this internally, but document the ratio
   - For classification: compute class weights, apply oversampling on training set, and use weighted loss
6. Split 70/15/15 (stratified to preserve defect ratio), create `data.yaml`, save metadata CSV

## 2.5 Data Augmentation

**For YOLO detection (applied during YOLO training via Ultralytics built-in augmentation):**
- Ultralytics handles augmentation automatically (mosaic, mixup, hsv shifts, flips, scale)
- Configure via training `args`: `augment=True, mosaic=1.0, mixup=0.1, flipud=0.5, fliplr=0.5`

**For CNN classification (applied via PyTorch transforms on training set only):**
```python
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

- Convert grayscale to RGB by repeating channels: `img = img.convert('RGB')`

## 2.6 Verification & Summary
At the end of this notebook:
- Print dataset statistics table (per dataset: total images, per-class counts, train/val/test splits)
- Print number of bounding box annotations per class
- Print class imbalance ratios
- Save sample image grids (5 images per class with bounding box overlays for detection version) as figures
- Verify YOLO data.yaml files are valid by running `from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.val(data='data.yaml')` as a quick sanity check
- Save everything to Drive

## 2.7 Checkpoint Logic
- Before downloading each dataset, check if it already exists on Drive
- If yes, skip download and print "Dataset already exists, skipping..."

---

# ============================================================
# PHASE 3A: YOLO DETECTION TRAINING (Colab Notebook)
# ============================================================

**Create: `colab_notebooks/02_yolo_detection_training.ipynb`**

## 3A.1 Setup
- Mount Drive, install `ultralytics`
- Load dataset `data.yaml` files from Phase 2

## 3A.2 Train YOLOv8 Models

Train **3 YOLO sizes × 3 datasets = 9 detection models:**

| # | YOLO Variant | Dataset | Purpose |
|---|-------------|---------|---------|
| 1 | YOLOv8n (nano) | NEU-DET | Lightweight, edge-focused |
| 2 | YOLOv8n | DAGM | |
| 3 | YOLOv8n | KSDD2 | |
| 4 | YOLOv8s (small) | NEU-DET | Balanced |
| 5 | YOLOv8s | DAGM | |
| 6 | YOLOv8s | KSDD2 | |


### Training Configuration:
```python
from ultralytics import YOLO

for yolo_size in ['yolov8n', 'yolov8s', 'yolov8m']:
    for dataset_name, yaml_path in datasets.items():
        run_name = f'{yolo_size}_{dataset_name}'
        
        # Check if already trained (checkpoint-resume)
        save_dir = f'{BASE}/models/yolo/full/{run_name}'
        if os.path.exists(f'{save_dir}/weights/best.pt'):
            print(f'{run_name} already trained, skipping...')
            continue
        
        model = YOLO(f'{yolo_size}.pt')  # Load pretrained COCO weights
        results = model.train(
            data=yaml_path,
            epochs=40,
            patience=15,          # Early stopping
            batch=32,             # Reduce to 16 if OOM
            imgsz=640,            # YOLO standard input size
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,             # Final LR = lr0 * lrf
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            project=f'{BASE}/models/yolo/full',
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=10,       # Save checkpoint every 10 epochs
            plots=True,           # Generate training plots
            device=0,             # GPU
            seed=42,
            verbose=True,
        )
```

### For DAGM and KSDD2 (binary detection — single "defect" class):
- Since these have only 1 class (defect), YOLO essentially does anomaly localization
- Consider setting `imgsz=512` for DAGM (native resolution) or keeping 640

## 3A.3 YOLO Evaluation
After training each model:
```python
# Evaluate on test set
metrics = model.val(
    data=yaml_path,
    split='test',
    save_json=True,
    plots=True,
)

# Record metrics
results_dict = {
    'model': yolo_size,
    'dataset': dataset_name,
    'mAP50': metrics.box.map50,
    'mAP50-95': metrics.box.map,
    'precision': metrics.box.mp,
    'recall': metrics.box.mr,
    'f1': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-8),
}
```

## 3A.4 Export Models
```python
# Export to ONNX for CPU deployment
model.export(format='onnx', imgsz=640, dynamic=True, simplify=True)

# Export to TorchScript
model.export(format='torchscript', imgsz=640)

# Export to TensorFlow Lite (FP32 — for edge deployment)
model.export(format='tflite', imgsz=640)

# Export to TensorRT (runs on Colab T4 GPU only — for GPU benchmark comparison)
model.export(format='engine', imgsz=640, device=0)
```

**IMPORTANT — Deployment framework mapping (matches thesis proposal):**
- **ONNX Runtime** → primary CPU inference framework (your local i5)
- **TensorFlow Lite** → lightweight edge deployment framework (also runs on CPU, simulates edge device)
- **NVIDIA TensorRT** → GPU-accelerated inference (benchmark on Colab T4 only, report as "GPU baseline" comparison)

This exactly matches the proposal's stated frameworks: *"Optimization Frameworks: TensorFlow Lite, ONNX Runtime, and NVIDIA TensorRT."*

## 3A.5 Save Everything
- All trained weights (`best.pt`, `last.pt`) to Drive
- All exported models (ONNX, TFLite, TensorRT engine) to Drive
- Training curves (auto-generated by Ultralytics) to Drive
- Evaluation metrics as CSV to Drive
- Confusion matrices (auto-generated) to Drive

## 3A.6 TensorRT GPU Benchmarks (Colab only)
Since TensorRT requires an NVIDIA GPU (not available on local CPU), run GPU benchmarks here:
```python
# Load TensorRT engine and benchmark on Colab T4
for yolo_size in ['yolov8n', 'yolov8s', 'yolov8m']:
    for dataset_name in datasets:
        engine_path = f'{BASE}/models/yolo/tensorrt/{yolo_size}_{dataset_name}.engine'
        model_trt = YOLO(engine_path)
        
        # Benchmark: 100 inferences, record latency
        # Record: mean latency (ms), FPS, mAP on test set
        # Save results to results/tables/tensorrt_gpu_benchmarks.csv
```
These GPU results serve as the "high-end" comparison point against your CPU inference results — showing the speed gap between GPU (TensorRT) and CPU (ONNX Runtime / TFLite).

---

# ============================================================
# PHASE 3B: CNN CLASSIFICATION TRAINING (Colab Notebook)
# ============================================================

**Create: `colab_notebooks/03_cnn_classification_training.ipynb`**

## 3B.1 Setup
- Mount Drive, load classification split CSVs from Phase 2
- Create PyTorch Dataset classes and DataLoaders

## 3B.2 Dataset Class
```python
class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        label = self.df.iloc[idx]['label']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.df)
```

## 3B.3 Train 9 CNN Classification Models

| # | Backbone | Dataset | Num Classes |
|---|----------|---------|-------------|
| 1 | ResNet50 | NEU | 6 |
| 2 | ResNet50 | DAGM | 2 (defect/no-defect) |
| 3 | ResNet50 | KSDD2 | 2 (defect/no-defect) |
| 4 | EfficientNet-B0 | NEU | 6 |
| 5 | EfficientNet-B0 | DAGM | 2 |
| 6 | EfficientNet-B0 | KSDD2 | 2 |
| 7 | MobileNetV3-Small | NEU | 6 |
| 8 | MobileNetV3-Small | DAGM | 2 |
| 9 | MobileNetV3-Small | KSDD2 | 2 |

### Model Creation Function:
```python
import torchvision.models as models

def create_model(backbone_name, num_classes, pretrained=True):
    if backbone_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model
```

### Transfer Learning — Two-Stage Strategy:

**Stage 1: Feature Extraction (5 epochs)**
- Freeze ALL backbone layers
- Train only the new classification head
- LR: 1e-3, optimizer: AdamW

**Stage 2: Fine-Tuning (up to 45 epochs)**
- Unfreeze last layers:
  - ResNet50: unfreeze `layer4` and `layer3`
  - EfficientNet-B0: unfreeze last 3 `features` blocks
  - MobileNetV3-Small: unfreeze last 4 `features` blocks
- LR: 1e-5 for unfrozen backbone, 1e-4 for head (use parameter groups)
- Scheduler: CosineAnnealingLR (T_max = remaining epochs)
- Early stopping: patience=7, monitor val_f1 (macro)

### Training Loop Must:
- Handle **class imbalance**: compute class weights → `CrossEntropyLoss(weight=class_weights)`
- Log per epoch: train_loss, val_loss, train_acc, val_acc, val_precision, val_recall, val_f1 (macro)
- Save logs as CSV to Drive after every epoch
- Save best model (by val_f1) as `.pth`
- Use `tqdm` for progress bars
- Print estimated time remaining per model

### Post-Training (per model):
- Evaluate on test set: accuracy, precision, recall, F1 (macro + per-class), AUC (one-vs-rest)
- Save confusion matrix data as numpy array
- Save per-class probabilities (for ROC/PR curves later)
- Export to ONNX (opset 13, dynamic batch, input shape [1, 3, 224, 224])
- Extract penultimate layer features for test set (for t-SNE later) → save as numpy array

### Checkpoint-Resume:
- Before training each combo, check if `{backbone}_{dataset}_best.pth` exists on Drive
- If yes, skip

---

# ============================================================
# PHASE 4: MODEL OPTIMIZATION (Colab Notebook)
# ============================================================

**Create: `colab_notebooks/04_model_optimization.ipynb`**

Apply optimization to BOTH YOLO and CNN models.

## 4.1 YOLO Model Optimization

### 4.1.1 YOLO Quantization
For each of the 9 YOLO models:

**a) ONNX Runtime Quantization:**
```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

# Dynamic quantization
quantize_dynamic(
    f'{model_path}.onnx',
    f'{model_path}_onnx_dynamic_quant.onnx',
    weight_type=QuantType.QUInt8
)

# Static quantization (with calibration dataset)
# Create calibration data reader from validation images
# quantize_static(model_input, model_output, calibration_reader)
```

**b) TFLite Quantization:**
```python
import tensorflow as tf

# Load the TFLite FP32 model and create INT8 version
# Use representative dataset for full integer quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_types = [tf.int8]
tflite_quant_model = converter.convert()
```

**c) TensorRT Quantization (Colab only — INT8 calibration on T4):**
```python
# Ultralytics handles TensorRT INT8 via:
model.export(format='engine', imgsz=640, device=0, int8=True, data=yaml_path)
# This creates a calibrated INT8 TensorRT engine
# Benchmark on Colab T4, save results for comparison
```

### 4.1.2 YOLO Pruning
- Ultralytics doesn't natively support pruning, but we can:
  1. Load the PyTorch model from `best.pt`
  2. Apply `torch.nn.utils.prune.l1_unstructured` to Conv2d layers at 20%, 40%, 60%
  3. Fine-tune for 10 more epochs
  4. Re-export to ONNX and TFLite
  5. Evaluate mAP, precision, recall

### 4.1.3 Record All YOLO Metrics
- Model size (MB) for each variant
- mAP50, mAP50-95, precision, recall, F1
- Save to `results/tables/yolo_optimization_summary.csv`

## 4.2 CNN Model Optimization

### 4.2.1 CNN Quantization

**a) PyTorch Dynamic Quantization (all 9 models):**
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)
```

**b) PyTorch Static Quantization (all 9 models):**
- Insert QuantStub/DeQuantStub
- Prepare with `torch.quantization.prepare`
- Calibrate on 500 validation images
- Convert with `torch.quantization.convert`
- **NOTE:** If static quantization fails for EfficientNet or MobileNetV3 (common due to complex blocks), document the error and use only dynamic quantization for those. This is expected and not a problem.

**c) ONNX Runtime Quantization (all 9 models):**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(input_onnx, output_onnx, weight_type=QuantType.QUInt8)
```

### 4.2.2 CNN Pruning
For each of the 9 CNN models, at sparsity levels 20%, 40%, 60%:
1. Apply L1 unstructured pruning to all Conv2d and Linear layers
2. Fine-tune for 10 epochs (early stopping patience=3)
3. Make pruning permanent with `prune.remove()`
4. Evaluate on test set
5. Save model
6. Also apply dynamic quantization on top of the best pruned model (combined optimization)

### 4.2.3 CNN — TensorFlow Lite Conversion (REQUIRED — matches proposal)
For each of the 9 CNN models:
1. Export PyTorch → ONNX (already done in Phase 3B)
2. Convert ONNX → TensorFlow SavedModel using `onnx-tf`:
   ```python
   !pip install onnx-tf
   from onnx_tf.backend import prepare
   import onnx
   onnx_model = onnx.load(onnx_path)
   tf_rep = prepare(onnx_model)
   tf_rep.export_graph(saved_model_dir)
   ```
3. Convert SavedModel → TFLite FP32:
   ```python
   converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
   tflite_fp32 = converter.convert()
   ```
4. Convert SavedModel → TFLite INT8 (with calibration):
   ```python
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = representative_dataset_gen  # 100 val images
   converter.target_spec.supported_types = [tf.int8]
   tflite_int8 = converter.convert()
   ```
5. Save both FP32 and INT8 TFLite models
6. Evaluate TFLite models using `tf.lite.Interpreter` on test set
7. Record: model_size_mb, accuracy, F1 for each TFLite variant

**NOTE:** If `onnx-tf` conversion fails for any architecture (common with MobileNetV3 due to hard-swish), document the error, try the alternative path `torch → ONNX → tf2onnx → TFLite`, and if that also fails, skip TFLite for that specific model only.

### 4.2.4 CNN — ONNX Runtime Quantization (REQUIRED)
For each of the 9 ONNX models:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(input_onnx, output_onnx, weight_type=QuantType.QUInt8)
```

### 4.2.5 CNN — TensorRT Benchmarks (Colab T4 only)
For the 3 best CNN models (one per backbone, best dataset):
- Convert ONNX → TensorRT engine on Colab T4
- Run inference benchmarks (latency, FPS) on GPU
- Save as GPU baseline comparison
- These numbers will NOT be available on local CPU — they serve as the "what GPU gives you" reference point

## 4.3 Record All CNN Metrics
- For EVERY variant: model_size_mb, accuracy, precision, recall, f1, num_params, num_nonzero_params, framework (PyTorch/ONNX/TFLite)
- Save to `results/tables/cnn_optimization_summary.csv`

## 4.4 Master Optimization Summary
- Merge YOLO and CNN results into `results/tables/full_optimization_summary.csv`
- Include columns: pipeline (yolo/cnn), model_name, dataset, opt_type, framework (PyTorch/ONNX/TFLite/TensorRT), sparsity_pct, model_size_mb, primary_metric (mAP50 for YOLO, F1 for CNN), accuracy, gpu_latency_ms (TensorRT on Colab, if available), cpu_latency_placeholder (filled by local inference)

---

# ============================================================
# PHASE 5: LOCAL CPU INFERENCE
# ============================================================

**Create these scripts in `local_inference/`**

All scripts must:
- Accept `--models_dir`, `--datasets_dir`, `--results_dir` arguments
- Work on Windows 11, Python 3.10+, CPU only
- Handle both PyTorch `.pth`, ONNX `.onnx`, and TFLite `.tflite` models
- Use `tqdm` progress bars, proper logging, error handling

## 5.1 `run_cpu_inference_yolo.py`

Evaluates ALL YOLO model variants on CPU:
```
python run_cpu_inference_yolo.py --models_dir ./models/yolo --datasets_dir ./datasets --results_dir ./results
```
- Load each YOLO variant: full (.pt), ONNX (.onnx), ONNX quantized, TFLite FP32, TFLite INT8, pruned
- For each, run inference on the test set of each dataset
- Record: mAP50, mAP50-95, precision, recall, F1, per-class AP
- **Test with ONNX Runtime:** `import onnxruntime as ort; session = ort.InferenceSession(onnx_path)`
- **Test with TFLite Runtime:** `import tensorflow as tf; interpreter = tf.lite.Interpreter(model_path=tflite_path)`
- Save results to `results/tables/yolo_cpu_inference.csv`

## 5.2 `run_cpu_inference_cnn.py`

Evaluates ALL CNN model variants on CPU:
```
python run_cpu_inference_cnn.py --models_dir ./models/cnn --datasets_dir ./datasets --results_dir ./results
```
- Load each CNN variant: full (.pth), dynamic quantized, static quantized (if available), ONNX, ONNX quantized, TFLite FP32, TFLite INT8 (if available), pruned at each sparsity
- Run inference on test set
- Record: accuracy, precision, recall, F1, per-class metrics, confusion matrix
- Save results to `results/tables/cnn_cpu_inference.csv`

## 5.3 `run_combined_pipeline.py`

The **two-stage pipeline**: YOLO detects → CNN classifies:
```
python run_combined_pipeline.py --yolo_model ./models/yolo/full/yolov8s_neu/best.pt --cnn_model ./models/cnn/full/resnet50_neu_best.pth --dataset_dir ./datasets/NEU-DET/images/test --results_dir ./results
```
- For each test image:
  1. Run YOLO detection → get bounding boxes
  2. Crop each detected region
  3. Resize crop to 224×224, apply CNN inference
  4. Combine: YOLO localization + CNN classification
- Record end-to-end metrics: detection mAP, classification accuracy on detected regions, total pipeline latency
- Compare against YOLO-only classification and CNN-only classification
- Save results to `results/tables/combined_pipeline_results.csv`

## 5.4 `measure_performance.py`

Measures latency, FPS, memory for ALL model variants (both YOLO and CNN):
```
python measure_performance.py --models_dir ./models --datasets_dir ./datasets --results_dir ./results
```

For each model variant:
- **Warm-up:** 10 inference passes (discard)
- **Timed runs:** 100 inference passes, record individual latency
- **Repeat 3 times** for statistical reliability
- **Batch testing** (CNN only): batch sizes [1, 4, 8, 16]
- **Record per variant:**
  - Mean latency (ms) ± std
  - FPS = 1000 / mean_latency_ms
  - Model load time (ms)
  - Peak memory usage (MB) via `tracemalloc`
  - 95th percentile latency
- **Test inference backends separately:**
  - PyTorch CPU (baseline)
  - ONNX Runtime CPU
  - TFLite Runtime CPU (using `tf.lite.Interpreter`)
  - Compare all three to show which framework is fastest on CPU
- Also include TensorRT GPU results from Colab (loaded from saved CSV) as a reference point
- Save to `results/tables/performance_all_models.csv`

## 5.5 `energy_estimation.py`

TDP-based energy estimation for all model variants:
```
python energy_estimation.py --models_dir ./models --datasets_dir ./datasets --results_dir ./results
```

Methodology:
- CPU TDP = 15W (Intel i5-1335U PBP)
- During inference, sample `psutil.cpu_percent(interval=0.01)` in a separate thread
- `Energy_per_frame (J) = TDP * (avg_cpu_util / 100) * inference_time_seconds`
- Convert: `Wh_per_frame = Energy_J / 3600`
- Also compute `Wh_per_1000_frames`
- Run 50 inference passes per model while measuring
- Save to `results/tables/energy_estimation.csv`

## 5.6 `compare_results.py`

Aggregates ALL results:
```
python compare_results.py --results_dir ./results
```
- Reads all CSVs from previous scripts
- Creates:
  - `results/tables/master_results_yolo.csv` — all YOLO variants with all metrics
  - `results/tables/master_results_cnn.csv` — all CNN variants with all metrics
  - `results/tables/master_results_combined.csv` — combined pipeline results
  - `results/tables/master_results_all.csv` — everything in one table
  - `results/tables/pareto_optimal.csv` — Pareto-optimal configs (accuracy vs latency, accuracy vs energy, accuracy vs model size)
  - `results/tables/best_models_summary.csv` — best model per category
  - `results/tables/yolo_vs_cnn_comparison.csv` — direct comparison of detection vs classification pipelines

## 5.7 `realtime_simulation.py`

Real-time defect inspection simulation:
```
python realtime_simulation.py --mode yolo --model_path ./models/yolo/full/yolov8s_neu/best.pt --dataset_dir ./datasets/NEU-DET/images/test
python realtime_simulation.py --mode cnn --model_path ./models/cnn/full/resnet50_neu_best.pth --dataset_dir ./datasets/NEU-DET/images/test
python realtime_simulation.py --mode combined --yolo_model ./models/yolo/... --cnn_model ./models/cnn/... --dataset_dir ...
python realtime_simulation.py --mode yolo --model_path ... --webcam  # Live webcam mode
```

Features:
- Reads images sequentially simulating a conveyor belt camera (configurable delay: 0.1s default)
- OpenCV display window showing:
  - Original image with overlaid predictions
  - For YOLO: bounding boxes with class labels and confidence
  - For CNN: predicted class label and confidence bar
  - For combined: bounding boxes + per-box classification
  - Live FPS counter (top-left)
  - Inference time in ms (top-left)
  - Model name (top-right)
- Controls: 'q' quit, 's' screenshot, 'p' pause, '+'/'-' adjust speed
- At exit: print summary statistics (avg FPS, avg latency, min/max latency)
- Webcam mode: reads from default camera, runs inference on each frame
- **SAVE a screen recording** (use `cv2.VideoWriter`) of 30 seconds of operation for thesis presentation

## 5.8 `generate_local_figures.py`

Generates all CPU-inference-specific figures. Details in Phase 6.

---

# ============================================================
# PHASE 6: FIGURES AND TABLES (Master Thesis Quality)
# ============================================================

**Create TWO files:**
1. `colab_notebooks/05_generate_figures_and_tables.ipynb` — training-side figures (runs on Colab with training results)
2. `local_inference/generate_local_figures.py` — inference-side figures (runs locally after CPU benchmarking)

### Global Style (use in BOTH files):
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Color scheme
YOLO_COLORS = {'YOLOv8n': '#E91E63', 'YOLOv8s': '#9C27B0', 'YOLOv8m': '#673AB7'}
CNN_COLORS = {'ResNet50': '#2196F3', 'EfficientNet-B0': '#4CAF50', 'MobileNetV3': '#FF9800'}
DATASET_MARKERS = {'NEU': 'o', 'DAGM': 's', 'KSDD2': '^'}
OPT_HATCHES = {'full': '', 'quantized': '///', 'pruned_20': '...', 'pruned_40': 'xxx', 'pruned_60': '\\\\\\'}

plt.rcParams.update({
    'font.size': 14, 'font.family': 'serif',
    'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})
```

### ALL REQUIRED FIGURES:

Save every figure as `{name}.png` (300 DPI) AND `{name}.pdf`.

**Dataset Figures (Colab):**
1. **Fig 1: Dataset sample grid** — For each dataset, show 5 sample images per class. For detection datasets, overlay bounding boxes. 3 subfigures.
2. **Fig 2: Class distribution** — Grouped bar charts per dataset showing train/val/test split counts per class.
3. **Fig 3: Annotation examples** — Show 3 images per dataset with YOLO bounding box annotations overlaid (different colors per class).

**YOLO Training Figures (Colab):**
4. **Fig 4: YOLO training curves** — 3×3 grid (rows=YOLO sizes, cols=datasets). Each subplot shows train loss and val loss over epochs.
5. **Fig 5: YOLO mAP curves** — Same layout, showing mAP50 and mAP50-95 over epochs.
6. **Fig 6: YOLO confusion matrices** — 3×3 heatmaps (rows=YOLO sizes, cols=datasets).
7. **Fig 7: YOLO prediction samples** — For each YOLO size on NEU-DET, show 4 test images with predicted bounding boxes overlaid. Color-code by class. 3 rows (YOLOv8n/s/m) × 4 columns (images).

**CNN Training Figures (Colab):**
8. **Fig 8: CNN training loss curves** — 3 subplots (one per dataset), each with 3 lines (backbones).
9. **Fig 9: CNN validation loss curves** — Same layout.
10. **Fig 10: CNN validation F1 curves** — Same layout.
11. **Fig 11: CNN confusion matrices** — 3×3 heatmaps (rows=backbones, cols=datasets).
12. **Fig 12: CNN ROC curves** — Per dataset, one plot with macro-average ROC for each backbone. AUC in legend.
13. **Fig 13: CNN Precision-Recall curves** — Same layout.
14. **Fig 14: Grad-CAM visualizations** — For NEU dataset: 3 backbones × 4 images (3 correct + 1 misclassified). Show original image, Grad-CAM heatmap, and overlay. Total: 3×4 grid with 3 sub-images each.
15. **Fig 15: t-SNE embeddings** — For NEU test set: 3 subplots (one per backbone), 2D scatter colored by class.
16. **Fig 16: Feature similarity across backbones** — CKA or cosine similarity matrix between the 3 backbones' feature representations.

**Optimization Figures (Colab + local combined):**
17. **Fig 17: Model size comparison — YOLO** — Grouped bar: full vs quantized vs pruned for all 9 YOLO models.
18. **Fig 18: Model size comparison — CNN** — Same for all 9 CNN models.
19. **Fig 19: Accuracy degradation — YOLO** — Bar chart showing mAP50 drop (%) per optimization.
20. **Fig 20: Accuracy degradation — CNN** — Bar chart showing F1 drop (%) per optimization.

**CPU Inference Figures (local script):**
21. **Fig 21: Latency comparison — YOLO** — Grouped bars: inference backends (PyTorch vs ONNX Runtime vs TFLite) for each YOLO variant. Include TensorRT (GPU) as a dashed reference line.
22. **Fig 22: Latency comparison — CNN** — Grouped bars: full vs quantized vs pruned for each CNN variant.
23. **Fig 23: FPS comparison — All models** — Horizontal bar chart, all model variants sorted by FPS. Color by pipeline (YOLO=pink, CNN=blue).
24. **Fig 24: Energy comparison** — Bar chart of Wh/1000frames for all variants.
25. **Fig 25: Latency box plots — YOLO** — Box plots of latency distribution for YOLO variants.
26. **Fig 26: Latency box plots — CNN** — Box plots for CNN variants.
27. **Fig 27: Accuracy vs Latency scatter** — All variants plotted. Color by model family, marker by dataset. Draw Pareto frontier. Separate markers for YOLO vs CNN.
28. **Fig 28: Accuracy vs Model Size scatter** — Same style with Pareto frontier.
29. **Fig 29: Accuracy vs Energy scatter** — Same style with Pareto frontier.
30. **Fig 30: Multi-objective radar chart** — 6 radar charts (3 YOLO sizes + 3 CNN backbones). Axes: Accuracy/mAP, 1/Latency, 1/ModelSize, 1/Energy, FPS. Normalized 0-1.
31. **Fig 31: Heatmap of all metrics** — Rows = all model variants (YOLO + CNN), Columns = metrics. Color-coded, annotated with values.
32. **Fig 32: YOLO vs CNN pipeline comparison** — Side-by-side bar charts comparing best YOLO vs best CNN on each dataset across all metrics.
33. **Fig 33: Combined pipeline results** — Bar chart showing the two-stage (YOLO+CNN) pipeline metrics vs standalone pipelines.

**Architecture & Planning Figures (either):**
34. **Fig 34: System architecture diagram** — Create using matplotlib patches/arrows. Show complete dual pipeline: Data → Preprocessing → [YOLO Detection Branch, CNN Classification Branch] → Optimization → CPU Deployment → Real-time Inference. Professional-looking.
35. **Fig 35: Gantt chart** — 24-week research timeline with overlapping phases: Literature Review, Data Prep, YOLO Training, CNN Training, Optimization, CPU Testing, Analysis, Writing.

### ALL REQUIRED TABLES (save as CSV + formatted markdown):

1. **Table 1: Dataset overview** — Dataset, #classes, #images, #defective, #non-defective, image dims, annotation type, source
2. **Table 2: Hardware specifications** — Colab T4 specs vs local i5-1335U specs (VRAM, RAM, TDP, CUDA cores vs CPU cores, etc.)
3. **Table 3: YOLO training hyperparameters** — All YOLO training settings
4. **Table 4: CNN training hyperparameters** — All CNN training settings
5. **Table 5: YOLO detection results (full models)** — 9 rows: mAP50, mAP50-95, Precision, Recall, F1, training time
6. **Table 6: CNN classification results (full models)** — 9 rows: Accuracy, Precision, Recall, F1 (macro), AUC, training time
7. **Table 7: CNN per-class F1 scores** — For NEU (6 classes): F1 per class per backbone
8. **Table 8: YOLO per-class AP** — For NEU (6 classes): AP per class per YOLO size
9. **Table 9: YOLO optimization results** — model, opt_type, model_size_mb, mAP50, mAP50_drop, latency_ms
10. **Table 10: CNN optimization results** — model, opt_type, sparsity, model_size_mb, f1, f1_drop, latency_ms
11. **Table 11: Combined optimization summary** — All models (YOLO+CNN), all opt methods, all metrics
12. **Table 12: CPU latency by inference backend** — YOLO and CNN models across PyTorch/ONNX Runtime/TFLite backends on CPU, plus TensorRT on Colab T4 GPU as reference
13. **Table 13: CPU performance — full** — ALL variants: latency_ms, fps, memory_mb, energy_wh, model_size_mb
14. **Table 14: Energy estimation results** — All variants: cpu_util%, inference_time_s, energy_J, Wh_per_frame, Wh_per_1000
15. **Table 15: Combined pipeline results** — YOLO+CNN vs YOLO-only vs CNN-only per dataset
16. **Table 16: Pareto-optimal configurations** — Best trade-off models
17. **Table 17: Cross-pipeline comparison** — Best YOLO vs Best CNN vs Combined for each dataset on each metric
18. **Table 18: Statistical significance** — Mean ± std for repeated runs of key metrics

---

# ============================================================
# PHASE 7: EXPLANATION DOCUMENT
# ============================================================

**Create: `docs/thesis_explanation_document.md`**

Comprehensive document (~35-40 pages equivalent), clear academic English.

### Structure:

**1. Executive Summary (1 page)**
- Dual-pipeline approach: YOLO detection + CNN classification
- Key findings summary
- Best performing configurations

**2. Dataset Analysis (3-4 pages)**
- Each dataset: origin, characteristics, challenges
- Preprocessing pipeline with justification
- Bounding box derivation methodology (for DAGM, KSDD2)
- Class distributions, imbalance handling
- Reference: Fig 1-3, Table 1

**3. YOLO Detection Pipeline (4-5 pages)**
- Why YOLOv8 (architecture overview, advantages)
- Why 3 sizes (nano/small/medium) — maps to edge deployment trade-offs
- Training strategy: pretrained COCO → fine-tune on industrial defects
- Results analysis: mAP across datasets, per-class performance
- Which YOLO size performs best where and why
- Reference: Fig 4-7, Tables 5, 8

**4. CNN Classification Pipeline (4-5 pages)**
- Why ResNet50, EfficientNet-B0, MobileNetV3 (architecture rationale)
- Transfer learning two-stage strategy explanation
- Training results: convergence, best performers
- Per-class analysis, failure modes
- Grad-CAM insights: what the models learned
- t-SNE: feature space quality
- Reference: Fig 8-16, Tables 6, 7

**5. Detection vs Classification: Comparative Analysis (2-3 pages)**
- When to use detection (localization needed) vs classification (binary quality check)
- Accuracy comparison: YOLO mAP vs CNN F1
- Speed comparison
- Combined pipeline: does two-stage beat single-stage?
- Reference: Fig 32-33, Tables 15, 17

**6. Model Optimization Analysis (4-5 pages)**
- **Quantization theory** + results for both YOLO and CNN
- **Pruning theory** + results at different sparsity levels
- **ONNX Runtime optimization** — quantized ONNX inference on CPU
- **TensorFlow Lite optimization** — FP32 and INT8 TFLite models for edge-like deployment
- **TensorRT GPU benchmarks** — as comparison baseline showing GPU vs CPU gap
- Which architectures respond best to which optimization
- Combined optimization effectiveness
- Reference: Fig 17-20, Tables 9-11

**7. CPU Deployment & Edge Substitute Results (4-5 pages)**
- Hardware description and edge-device substitute justification
- **Inference backend comparison:** PyTorch vs ONNX Runtime vs TFLite on CPU (Table 12)
- **GPU vs CPU comparison:** TensorRT on T4 vs ONNX Runtime on i5 — quantifying the speed gap
- Latency and FPS analysis across all models
- Energy estimation methodology + results + honest limitations
- Memory footprint analysis
- Real-time capability assessment: which models achieve ≥30 FPS?
- Reference: Fig 21-31, Tables 13-14

**8. Research Questions — Answered with Evidence (5-6 pages)**

**RQ1: How effectively can transfer learning adapt pre-trained CNNs to detect manufacturing defects with limited domain-specific data?**
- Evidence: CNN F1 scores (Table 6) — high F1 with only hundreds of images per class
- YOLO: COCO-pretrained → industrial defects also shows strong transfer
- Grad-CAM evidence of meaningful feature learning (Fig 14)
- Comparison: what accuracy would we expect without transfer learning?

**RQ2: What level of inference speed and energy efficiency can be achieved on low-power devices without compromising accuracy?**
- Define "without compromising": <2% F1/mAP drop
- Show which optimized variants meet this (Table 16)
- Best FPS achievable (likely YOLOv8n + ONNX Runtime or TFLite INT8)
- Energy per 1000 frames analysis
- "Low-power device" = 15W TDP CPU (comparable to edge devices)

**RQ3: Which optimization techniques provide the best trade-off between performance and latency?**
- Pareto analysis (Fig 27-29, Table 16)
- Quantization vs pruning vs combined — clear recommendation
- For YOLO: ONNX Runtime likely gives best CPU speed; TFLite for minimal footprint
- For CNN: dynamic quantization gives best size reduction with minimal accuracy loss
- TensorRT (GPU) is fastest but requires NVIDIA hardware — quantify the gap

**RQ4: Can the proposed framework serve as a reproducible and scalable solution for SMEs adopting Industry 4.0?**
- Reproducibility: all code open, all datasets public, documented pipeline
- Cost analysis: Colab Pro ($12/mo for training) + $0 for CPU inference
- Scalability: adding new defect types = retrain with new labels
- What an SME deployment would look like in practice

**9. Figures and Tables Interpretation Guide (3-4 pages)**
- For EACH figure and table: what it shows, how to read it, key takeaway
- Suggested talking points for thesis defense

**10. Key Findings (1 page)**
- Top 10-15 numbered findings

**11. Limitations and Future Work (1-2 pages)**
- No real factory testing
- TDP-based energy estimation (vs actual power meter)
- Public datasets may not represent production variability
- CPU ≠ actual Raspberry Pi/Jetson (but comparable thermal envelope)
- Future: real edge deployment, sensor fusion, online learning, anomaly detection approaches

**12. Reproduction Guide (1-2 pages)**
- Step-by-step: clone repo → run Colab notebooks → download results → run local scripts
- Expected total Colab runtime: ~6-8 hours on T4
- Expected storage: ~5-10 GB on Drive
- Expected local runtime: ~2-3 hours for all benchmarks

---

# ============================================================
# PHASE 8: REQUIREMENTS AND README
# ============================================================

### `requirements_colab.txt`
```
ultralytics>=8.0.0
torch>=2.0
torchvision>=0.15
onnx>=1.14
onnxruntime>=1.15
onnxruntime-extensions
onnx-tf
tensorflow>=2.13
albumentations>=1.3
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
scikit-learn>=1.3
tqdm
tensorboard
grad-cam>=1.4
umap-learn
openpyxl
```

### `requirements_local.txt`
```
ultralytics>=8.0.0
torch>=2.0
torchvision>=0.15
onnxruntime>=1.15
tensorflow>=2.13
opencv-python>=4.8
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
scikit-learn>=1.3
psutil>=5.9
tqdm
openpyxl
```

### `README.md`
- Project title, author, university, thesis topic
- Architecture overview (dual pipeline diagram in ASCII)
- Directory structure with explanations
- Prerequisites (Python 3.10+, Colab Pro account)
- **Complete execution guide:**
  1. Clone/setup project
  2. Upload `01_data_download_and_preprocessing.ipynb` to Colab → Run
  3. Upload `02_yolo_detection_training.ipynb` → Run (est. 2-3 hours)
  4. Upload `03_cnn_classification_training.ipynb` → Run (est. 2-3 hours)
  5. Upload `04_model_optimization.ipynb` → Run (est. 1-2 hours)
  6. Upload `05_generate_figures_and_tables.ipynb` → Run
  7. Download from Google Drive: models/, results/, training_logs/, datasets/ test splits
  8. Place in local thesis_project/ directory
  9. `pip install -r requirements_local.txt`
  10. Run local scripts in order:
      ```
      python local_inference/run_cpu_inference_yolo.py --models_dir ./models ...
      python local_inference/run_cpu_inference_cnn.py ...
      python local_inference/run_combined_pipeline.py ...
      python local_inference/measure_performance.py ...
      python local_inference/energy_estimation.py ...
      python local_inference/compare_results.py ...
      python local_inference/generate_local_figures.py ...
      python local_inference/realtime_simulation.py ...
      ```
- Expected results summary
- Troubleshooting common issues

---

# ============================================================
# EXECUTION ORDER — START NOW
# ============================================================

Phase 1 is done. Please execute in this order:

1. ✅ Phase 1 (project structure) — DONE
2. Create `requirements_colab.txt` and `requirements_local.txt`
3. Create `colab_notebooks/01_data_download_and_preprocessing.ipynb`
4. Create `colab_notebooks/02_yolo_detection_training.ipynb`
5. Create `colab_notebooks/03_cnn_classification_training.ipynb`
6. Create `colab_notebooks/04_model_optimization.ipynb`
7. Create all scripts in `local_inference/` (8 scripts)
8. Create `colab_notebooks/05_generate_figures_and_tables.ipynb`
9. Create `local_inference/generate_local_figures.py`
10. Create `docs/thesis_explanation_document.md` (template with placeholders for results)
11. Create `README.md`

## FINAL CRITICAL NOTES

- **Do NOT ask me questions** — make the best engineering decisions based on this prompt and standard ML best practices
- **Every notebook must be self-contained** and runnable top-to-bottom on Colab with a T4 GPU
- **Every local script must work on Windows 11** with CPU-only Python
- **Checkpoint-resume everywhere** — Colab disconnects are expected
- **Deployment frameworks must match the proposal**: ONNX Runtime (primary CPU inference), TensorFlow Lite (edge-oriented lightweight inference), NVIDIA TensorRT (GPU benchmarks on Colab T4 only). These three are explicitly named in the thesis proposal. Always export and test models in all three formats where possible.
- **TensorRT benchmarks happen on Colab only** (requires NVIDIA GPU). Create a separate notebook section or notebook `04b_tensorrt_benchmarks.ipynb` for this. Save results as CSV so they can be loaded locally for figure generation.
- **Add extensive comments** — my professor will review the code
- **Use `tqdm`** for all long-running operations
- **Error handling**: try/except with meaningful messages for downloads, model loading, quantization (which may fail for some architectures)
- **Seeds: 42** everywhere for reproducibility
- **Figure quality**: 300 DPI, serif font, consistent colors, proper labels, legends, grid. Publication quality.
- If a dataset download URL is dead, provide 2-3 alternative sources
- If DAGM is > 2GB, use classes 1–6 only (document the decision)
- If static quantization fails for an architecture, document it and use dynamic quantization as fallback

**Begin now. Start with requirements files, then notebook 01.**
