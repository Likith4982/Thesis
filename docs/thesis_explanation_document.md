# Thesis Explanation Document

**Title:** Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing
**Program:** Master of Science (Computer Science / Artificial Intelligence)
**Status:** Complete.

---

## 1. Research Motivation

Manufacturing defect detection has traditionally relied on human inspectors — a process that is
slow, expensive, and error-prone. Deep learning–based computer vision systems can automate this
with higher consistency, but two challenges persist:

1. **Localization vs. Classification:** Simply classifying "defective/OK" misses the location and
   type of defect. A detection-first approach provides actionable spatial information.
2. **Edge deployment:** Training on GPU and deploying on CPU-only edge hardware requires careful
   model selection, optimization, and format conversion.

This thesis addresses both with a **dual-pipeline** approach, asking:

> *Which combination of detection model (YOLOv8 variant), classification backbone, optimization
> technique, and inference format offers the best accuracy–latency–energy trade-off for
> real-time defect inspection on a CPU-only edge device?*

---

## 2. System Architecture

### 2.1 Dual-Pipeline Design

**Pipeline A — Object Detection (YOLOv8):**
YOLOv8 variants (nano/small/medium) trained to localize and classify defect regions using
bounding boxes. Produces spatial predictions (where + what class) for each image.

**Pipeline B — Image Classification (CNN Transfer Learning):**
ResNet-50, EfficientNet-B0, and MobileNetV3-Large trained to classify entire images or
cropped defect patches. Evaluated on ImageFolder-structured datasets.

**Combined Pipeline:**
YOLO detects defect regions → crops are passed to CNN for fine-grained classification.
Evaluated as a two-stage system (detection F/R + classification accuracy).

### 2.2 Deployment Target

Local Windows CPU simulates an edge device:
- Intel Core i5-1335U (13th Gen, 1.30 GHz, TDP 15W)
- 16 GB RAM, no GPU
- Inference runtimes: ONNX Runtime (primary), TFLite (edge), TensorRT (Colab GPU reference only)

---

## 3. Datasets

### 3.1 NEU Surface Defect Database (NEU-DET)

- **Source:** Northeastern University, China
- **Version:** DET version with PASCAL VOC XML bounding box annotations
- **Classes:** 6 — crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratched
- **Images:** ~1,800 (300 per class), 200×200 px, grayscale
- **Annotation processing:** XML parsed with `xml.etree.ElementTree`; `<bndbox>` elements
  converted to YOLO format (normalized x_center, y_center, width, height)
- **Classification version:** Full images (labeled by dominant class) + per-bbox crops
- **Split:** 70/15/15 stratified by dominant defect class
- **Why chosen:** Widely cited benchmark; balanced classes; explicit bbox annotations

### 3.2 DAGM 2007

- **Source:** German Working Group on Pattern Recognition competition dataset
- **Task:** Binary optical inspection — defective vs. non-defective surface texture
- **Classes used:** 1–6 of 10 total (see design decision below)
- **Images:** ~7,000 total for classes 1–6; ~1,000 non-defective + ~150 defective per class
- **Image size:** 512×512 px, grayscale
- **Annotation processing:** Defective images have elliptical mask `.PNG` files in `Label/`
  subdirectory. Masks binarized and passed to `cv2.findContours()` → `cv2.boundingRect()`
  → YOLO format bounding box. Non-defective images receive empty label files.
- **Detection:** Single class — `defect`. Non-defective = background (empty label file).
- **Classification:** Binary — `defective` / `non_defective` (all 6 texture classes merged)
- **Split:** 70/15/15 stratified
- **Design decision:** Classes 7–10 excluded to stay within Google Drive storage limits (~15 GB
  free). Classes 1–6 are representative of the defect texture distribution. This decision is
  documented in the thesis (Section 4.2).

### 3.3 Kolektor Surface Defect Dataset 2 (KSDD2)

- **Source:** Kolektor Group / University of Ljubljana
- **Task:** Binary binary defect inspection on metallic surfaces
- **Images:** ~3,335 (246 defective ≈ 7.4%, 3,089 non-defective ≈ 92.6%)
- **Image size:** ~230×630 px (non-square), grayscale; resized to 640×640 for YOLO
- **Annotation processing:** Pixel-level segmentation masks (`*_label.png`) for defective
  images. `cv2.findContours()` → `cv2.boundingRect()` → YOLO format. Non-defective = empty
  label file.
- **Class imbalance ratio:** ~12.5:1 (non-defective:defective)
- **Imbalance handling:**
  - Detection: Documented; YOLO handles internally via augmentation
  - Classification: `WeightedRandomSampler` for balanced training batches;
    class-weighted `CrossEntropyLoss`; class weights saved to `ksdd2_class_weights.json`
- **Split:** 70/15/15 stratified (preserves defect ratio in all partitions)
- **Why chosen:** Real industrial binary inspection scenario; severe class imbalance challenge

---

## 4. Model Architectures

### 4.1 Detection — YOLOv8 Variants

All three variants use Ultralytics pretrained COCO weights, fine-tuned on each dataset.

| Variant | Params | Expected mAP50 | Use case |
|---------|--------|----------------|----------|
| YOLOv8n (nano) | ~3 M | Lower | Fastest edge inference |
| YOLOv8s (small) | ~11 M | Medium | Balanced |


YOLO augmentation applied internally by Ultralytics:
`mosaic=1.0, mixup=0.1, flipud=0.5, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5`

### 4.2 Classification — CNN Backbones

All three use **transfer learning** from ImageNet pretrained weights (torchvision).
Classification head replaced with `nn.Linear(in_features, num_classes)`.

| Backbone | Params | ImageNet Top-1 | Strength |
|----------|--------|----------------|----------|
| ResNet-50 | ~25.6 M | 76.1% | Strong baseline |
| EfficientNet-B0 | ~5.3 M | 77.7% | Best accuracy/parameter ratio |
| MobileNetV3-Large | ~5.4 M | 75.3% | Fastest CPU inference |

---

## 5. Training Protocol

### 5.1 YOLO Detection Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Patience (early stopping) | 15 |
| Image size | 640 px |
| Batch size | 32 (reduce to 16 if OOM) |
| Optimizer | AdamW |
| Initial LR (lr0) | 0.001 |
| Final LR (lrf) | 0.01 × lr0 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| Seed | 42 |
| Device | T4 GPU (Colab) |

Checkpoint-resume: training skipped if `weights/best.pt` already exists for that run.

### 5.2 CNN Classification Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 50 |
| Patience (early stopping) | 10 |
| Input resolution | 224×224 px |
| Batch size | 32 |
| Optimizer | AdamW |
| LR | 1e-4 |
| LR scheduler | StepLR (step=10, γ=0.5) |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss (+ class weights for KSDD2) |
| Seed | 42 |

**Data augmentation (torchvision, training only):**
```
Resize(256) → RandomResizedCrop(224, scale=(0.8, 1.0)) → RandomHorizontalFlip(0.5)
→ RandomVerticalFlip(0.3) → RandomRotation(15) → ColorJitter(brightness=0.2, contrast=0.2)
→ GaussianBlur(kernel_size=3) → ToTensor() → Normalize(ImageNet mean/std)
```

Grayscale → RGB: `PIL Image.convert('RGB')` before transforms.

---

## 6. Model Optimization

### 6.1 ONNX Export

YOLO: `model.export(format='onnx', imgsz=640, dynamic=True, simplify=True)`
CNN: `torch.onnx.export(..., opset_version=17, dynamic_axes={'input': {0: 'batch'}})`

### 6.2 TFLite Conversion (ONNX → TF SavedModel → TFLite)

```python
onnx_tf.backend.prepare(onnx_model).export_graph(saved_model_path)
tf.lite.TFLiteConverter.from_saved_model(saved_model_path).convert()
```

### 6.3 INT8 Dynamic Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(input_model_path, output_model_path, weight_type=QuantType.QInt8)
```
Expected: ~4× size reduction, ~1.5–2× latency speedup, <1% accuracy drop.

### 6.4 Structured Pruning (L1 Magnitude)

Remove 20% least-important filters per conv layer, followed by 1-epoch fine-tuning.
Expected: ~15–20% parameter reduction, <2% accuracy drop.

### 6.5 TensorRT (Colab GPU only — reference benchmarks)

YOLO: `model.export(format='engine', half=True)` (FP16)
CNN: ONNX → TRT engine via Python TensorRT API
Results used as upper-bound GPU performance reference in thesis comparisons.

---

## 7. Inference & Edge Evaluation

### 7.1 Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel Core i5-1335U, 13th Gen, 1.30 GHz |
| TDP | 15 W (sustained PL1) |
| RAM | 16 GB |
| OS | Windows 11 Home 64-bit |
| GPU | None |

### 7.2 Latency Measurement

Median of 100 inference calls after 10 warmup runs. Reported in ms/image.
Measured with `time.perf_counter()` (Python) for ONNX Runtime and TFLite.

### 7.3 Throughput

FPS = 1000 / median_latency_ms

### 7.4 Energy Proxy

```
energy_mJ = cpu_util_fraction × TDP_W × latency_s × 1000
```
`cpu_util_fraction` sampled via `psutil.cpu_percent()` during inference window.

### 7.5 Combined Pipeline Latency

Total = YOLO detection latency + crop time + CNN classification latency
Measured end-to-end on same CPU.

---

## 8. Figures & Tables (62 Figures, 17+ Tables)

A total of 49 figures were generated from Colab training/evaluation results and 13 figures from local CPU inference benchmarks. Full figure and table inventories are listed in Section 12.

---

## 9. Research Questions

1. **RQ1:** Can YOLOv8 achieve mAP@50 ≥ 0.80 on all three industrial defect datasets?
2. **RQ2:** Which CNN backbone achieves the highest classification accuracy–speed trade-off
   on CPU-only hardware?
3. **RQ3:** Does the combined pipeline (YOLO → CNN) outperform standalone YOLO detection
   in classification accuracy while remaining within real-time latency constraints (<100 ms)?
4. **RQ4:** What is the accuracy cost of INT8 quantization vs. the latency/energy benefit
   across ONNX Runtime and TFLite on a CPU-only edge device?

---

## 10. Final Experimental Setup Used

### 10.1 Training Environment (Colab)
- Google Colab T4 GPU, CUDA 12.x
- Datasets: NEU-DET (6-class detection + 5-class classification), DAGM 2007 classes 1–6 (binary), KSDD2 (binary, 12.5:1 imbalance)
- YOLO: 6 models (YOLOv8n, YOLOv8s) x 3 datasets — 100 epochs, patience 15, AdamW
- CNN: 9 models (ResNet-50, EfficientNet-B0, MobileNetV3-L) x 3 datasets — 50 epochs, patience 10, AdamW
- Optimization: ONNX FP32 export, ONNX INT8 dynamic quantization, TFLite FP32/INT8, structured pruning (20%), TensorRT FP16 (GPU reference)

### 10.2 Edge Inference Environment (Local CPU)
- Intel Core i5-1335U (13th Gen, 1.30 GHz, TDP 15 W), 16 GB RAM, Windows 11, no GPU
- Runtimes: ONNX Runtime (FP32), TFLite (FP32/INT8), PyTorch (FP32)
- Benchmarking: 100 timed runs after 10 warmup runs, repeated 3 times
- Energy proxy: psutil CPU utilization x TDP x latency

### 10.3 Source-of-Truth Split

Results are drawn from two complementary sources:

- **Colab results** = training/test-set model quality (accuracy, mAP, F1) + GPU/TensorRT upper-bound reference
  - Source tables: `TABLE_YOLO_DetectionBenchmark_All.csv`, `TABLE_CNN_ClassificationBenchmark_All.csv`, `TABLE_TensorRT_Benchmarks.csv`
  - Source figures: `results/Colab_results/results/figures/` (49 figures)
- **Local CPU results** = real-world edge latency, FPS, energy consumption, and combined pipeline behavior
  - Source tables: `master_results_yolo.csv`, `master_results_cnn.csv`, `master_results_combined.csv`, `best_models_summary.csv`
  - Source figures: `results/figures/fig21–fig33` (13 figures)

---

## 11. Key Quantitative Results

### 11.1 YOLO Detection Accuracy (Colab Test Set)

| Dataset | Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|---------|-------|--------|-----------|-----------|--------|----|
| NEU-DET | YOLOv8n | 0.7235 | 0.4000 | 0.6336 | 0.7187 | 0.6735 |
| NEU-DET | YOLOv8s | 0.7191 | 0.4047 | 0.6812 | 0.6813 | 0.6813 |
| DAGM | YOLOv8n | 0.9910 | 0.6363 | 0.9641 | 1.0000 | 0.9817 |
| DAGM | YOLOv8s | 0.9931 | 0.6223 | 0.9674 | 0.9898 | 0.9785 |
| KSDD2 | YOLOv8n | 0.6673 | 0.3552 | 0.7466 | 0.6481 | 0.6939 |
| KSDD2 | YOLOv8s | 0.6422 | 0.3417 | 0.6447 | 0.7055 | 0.6737 |

DAGM achieves near-perfect detection (mAP@50 > 0.99). NEU-DET is moderate (~0.72) and KSDD2 is hardest (~0.65) due to severe class imbalance.

### 11.2 CNN Classification Accuracy (Colab Test Set)

| Dataset | Model | Test Accuracy | Test F1 (macro) | Train Time (min) |
|---------|-------|--------------|-----------------|------------------|
| NEU-DET | ResNet-50 | 99.88% | 0.999 | 34.9 |
| NEU-DET | EfficientNet-B0 | 98.51% | 0.9855 | 24.5 |
| NEU-DET | MobileNetV3-L | 98.63% | 0.9857 | 13.8 |
| DAGM | ResNet-50 | 99.48% | 0.9882 | 47.8 |
| DAGM | EfficientNet-B0 | 86.29% | 0.7499 | 27.9 |
| DAGM | MobileNetV3-L | 96.34% | 0.9159 | 32.2 |
| KSDD2 | ResNet-50 | 97.21% | 0.9328 | 49.2 |
| KSDD2 | EfficientNet-B0 | 92.61% | 0.8436 | 46.1 |
| KSDD2 | MobileNetV3-L | 93.81% | 0.8673 | 65.9 |

ResNet-50 dominates accuracy across all datasets. EfficientNet-B0 underperforms on DAGM (86.29%), suggesting texture-heavy binary classification favors larger backbones.

### 11.3 Best Models per Dataset (Local CPU, Weighted Ranking)

The weighted ranking considers accuracy (40%), latency (25–30%), energy (10–15%), and model size (10–15%).

| Pipeline | Dataset | Best Model | Format | Key Metric | Latency (ms) | FPS | Energy (Wh/1000) |
|----------|---------|------------|--------|-----------|--------------|-----|-------------------|
| YOLO | DAGM | YOLOv8n | PyTorch | mAP50 = 0.995 | 69.36 | 14.4 | 0.146 |
| YOLO | KSDD2 | YOLOv8n | PyTorch | mAP50 = 0.327 | 80.36 | 12.4 | 0.161 |
| YOLO | NEU-DET | YOLOv8n | PyTorch | mAP50 = 0.715 | 55.67 | 18.0 | 0.339 |
| CNN | DAGM | MobileNetV3-L | ONNX FP32 | Acc = 96.34% | 5.34 | 187.3 | 0.011 |
| CNN | KSDD2 | MobileNetV3-L | ONNX FP32 | Acc = 93.81% | 5.35 | 186.8 | 0.010 |
| CNN | NEU-DET | MobileNetV3-L | ONNX FP32 | Acc = 98.63% | 5.37 | 186.1 | 0.012 |

### 11.4 Combined Pipeline Performance (Local CPU)

Best combined configurations (YOLOv8n + CNN, PyTorch format):

| Dataset | CNN Model | End-to-End Latency (ms) | FPS | Det % | Cls % |
|---------|-----------|------------------------|-----|-------|-------|
| NEU-DET | MobileNetV3-L | 104.33 | 9.58 | 63.8% | 36.0% |
| NEU-DET | EfficientNet-B0 | 140.00 | 7.14 | 54.0% | 45.9% |
| NEU-DET | ResNet-50 | 157.14 | 6.36 | 34.9% | 64.9% |
| DAGM | MobileNetV3-L | 79.06 | 12.65 | 97.3% | 2.5% |
| KSDD2 | MobileNetV3-L | 87.81 | 11.39 | 95.0% | 4.8% |

Detection dominates total latency for most configurations. On DAGM/KSDD2 (binary, few detections), the combined pipeline approaches real-time (<100 ms).

### 11.5 TensorRT GPU Reference Benchmarks (Colab T4)

| Model | Type | Mean Latency (ms) | FPS |
|-------|------|-------------------|-----|
| MobileNetV3-L | CNN | 1.17–1.30 | 768–853 |
| EfficientNet-B0 | CNN | 1.75–2.03 | 493–571 |
| ResNet-50 | CNN | 3.63–3.74 | 267–275 |
| YOLOv8n | YOLO | 3.49–4.89 | 205–287 |
| YOLOv8s | YOLO | 4.55–5.05 | 198–220 |

These serve as upper-bound GPU performance references — real edge deployment on i5-1335U CPU is 10–50x slower.

### 11.6 Format Comparison Summary (Local CPU)

| Pipeline | Format | Mean Accuracy/mAP50 | Mean Latency (ms) | Mean FPS | Mean Size (MB) |
|----------|--------|--------------------|--------------------|----------|----------------|
| YOLO | PyTorch | 0.668 | 124.86 | 10.25 | 14.39 |
| YOLO | TFLite | 0.227 | 84.20 | 11.93 | 12.28 |
| CNN | ONNX FP32 | 0.959 | 16.36 | 105.92 | 42.59 |
| CNN | TFLite FP32 | 0.959 | 33.97 | 85.23 | 42.28 |
| CNN | TFLite INT8 | 0.954 | 31.79 | 34.86 | 10.99 |

ONNX FP32 is the best CPU runtime for CNN models. YOLO TFLite shows severe accuracy degradation (mAP drops from 0.668 to 0.227), indicating format conversion issues.

### 11.7 Answering the Research Questions

**RQ1: Can YOLOv8 achieve mAP@50 >= 0.80 on all three datasets?**
Partially. DAGM achieves 0.99 (yes). NEU-DET reaches 0.72 and KSDD2 reaches 0.67 — both fall short of the 0.80 threshold. KSDD2's severe class imbalance (12.5:1) and NEU-DET's 6-class complexity are the primary factors.

**RQ2: Which CNN backbone achieves the best accuracy-speed trade-off on CPU?**
MobileNetV3-Large in ONNX FP32 format. It achieves 93.8–98.6% accuracy across all datasets at 186 FPS (5.3 ms/image) on the i5-1335U CPU — 3–6x faster than ResNet-50 with only 0.4–3.1% accuracy loss.

**RQ3: Does the combined pipeline (YOLO -> CNN) remain within real-time constraints (<100 ms)?**
Only for low-detection-count datasets. DAGM and KSDD2 combined pipelines achieve <100 ms end-to-end with MobileNetV3-L (79–88 ms). NEU-DET exceeds 100 ms (104 ms best case) due to higher average detections per image (2.1 crops).

**RQ4: What is the accuracy cost of INT8 quantization?**
Mixed results. CNN TFLite INT8 preserves accuracy well (0.954 vs 0.959 FP32, <0.5% drop). However, ONNX INT8 quantization caused `ConvInteger` operator failures on the test CPU runtime, making those models non-functional. YOLO TFLite conversion caused severe mAP degradation (0.668 -> 0.227), indicating a format conversion issue rather than a quantization issue.

### 11.8 Methodology and Validity Notes

1. **ONNX INT8 `ConvInteger` limitation:** Dynamic INT8 quantization via `onnxruntime.quantization` produces `ConvInteger` nodes that are not implemented in the default ONNX Runtime CPU execution provider on this hardware. This affects all 15 ONNX INT8 model variants (6 YOLO + 9 CNN). These rows are excluded from latency/FPS comparisons and marked `status=error` in `performance_all_models.csv`.

2. **YOLO TFLite accuracy drop:** YOLO models exported to TFLite FP32 via the `onnx-tf` -> `tf.lite` conversion path show near-zero mAP on DAGM and KSDD2 (only NEU-DET retains partial accuracy). This is a known compatibility issue with YOLO post-processing layers in the TFLite format, not a model quality issue.

3. **Energy measurement is a proxy:** Energy is estimated as `CPU_utilization% x TDP(15W) x time`, sampled via `psutil.cpu_percent()`. This provides relative comparisons between models but is not equivalent to direct power measurement.

4. **Latency measurement methodology:** Mean of 100 inference runs after 10 warmup runs, repeated 3 times. Standard deviation and P95 latency are also recorded to capture variance.

---

## 12. Figures & Tables Reference

### 12.1 Colab Figures (Training/Evaluation)

**Detection:**
- `FIG_Detection_mAP_Heatmap` — mAP50 heatmap (model x dataset)
- `FIG_YOLO_mAP_Comparison` — grouped bar chart of mAP50 across all YOLO models
- `FIG_YOLO_PrecisionRecall_Scatter` — precision vs recall scatter
- `FIG_YOLO_ConfusionMatrix_{dataset}_{model}` x 6 — per-model confusion matrices
- `FIG_YOLO_Training_{dataset}_{model}` x 6 — per-model training curves
- `FIG_YOLO_TrainingCurve_{dataset}` x 3 — combined training curves per dataset
- `FIG_Tradeoff_mAP_vs_ModelSize` — mAP50 vs model size scatter

**Classification:**
- `FIG_Classification_Accuracy_Comparison` — grouped bar chart
- `FIG_Classification_Accuracy_Heatmap` — accuracy heatmap (model x dataset)
- `FIG_Classification_F1_Heatmap` — F1 macro heatmap
- `FIG_Classification_AccuracyVsSize` — accuracy vs model size
- `FIG_Classification_ConfusionMatrix_{dataset}_{model}` x 9 — per-model confusion matrices
- `FIG_Classification_Training_Dynamics_{dataset}` x 3 — training loss/accuracy curves

**Optimization & Reference:**
- `FIG_TRT_GPU_Benchmarks` — TensorRT FP16 latency/FPS on Colab T4
- `FIG_ModelSize_Comparison` — model size across formats
- `FIG_Radar_BestModels` — radar chart of best model per pipeline
- `FIG_SampleGrid_{dataset}` x 3 — example images from each dataset

### 12.2 Local CPU Figures (Edge Inference)

- `fig21_yolo_latency_comparison` — YOLO latency by model and format (grouped bar)
- `fig22_cnn_latency_comparison` — CNN latency by model and format
- `fig23_fps_comparison` — FPS comparison (all models, horizontal bar)
- `fig24_energy_comparison` — energy consumption (Wh per 1000 frames)
- `fig25_yolo_latency_boxplot` — YOLO latency distribution per format
- `fig26_cnn_latency_boxplot` — CNN latency distribution per format
- `fig27_yolo_pareto_acc_vs_latency` — YOLO mAP50 vs latency Pareto frontier
- `fig28_cnn_pareto_acc_vs_latency` — CNN accuracy vs latency Pareto frontier
- `fig29_cnn_acc_vs_size` — CNN accuracy vs model size scatter
- `fig30_radar_best_models` — radar chart for best models (multi-metric)
- `fig31_all_metrics_heatmap` — all metrics normalized heatmap
- `fig32_combined_pipeline_latency` — combined pipeline end-to-end latency
- `fig33_pipeline_breakdown` — detection vs classification latency breakdown

### 12.3 Deployment Recommendations

| Scenario | YOLO Model | CNN Model | Format | Target FPS | Platform |
|----------|-----------|-----------|--------|------------|----------|
| Highest accuracy (cloud GPU) | YOLOv8m | ResNet-50 | PyTorch FP32 | 5–15 | T4 / A100 GPU |
| Balanced (edge CPU) | YOLOv8n | EfficientNet-B0 | ONNX INT8 | 3–8 | Intel i5 / RPi 5 |
| Fastest (mobile) | YOLOv8n | MobileNetV3-L | TFLite INT8 | 1–5 | ARM Cortex-A / NPU |

---

## 13. Limitations & Scope

- DAGM classes 7–10 excluded due to Google Drive storage constraints (~15 GB free). Classes 1–6 are representative of the defect texture distribution.
- KSDD2 severe class imbalance (12.5:1) makes detection F1 less informative than classification F1; `WeightedRandomSampler` and class-weighted loss partially mitigate this.
- TensorRT results are Colab-specific (T4 GPU) — not representative of local CPU deployment; included only as upper-bound GPU reference.
- Energy estimation is a proxy (`psutil.cpu_percent()` x TDP), not direct hardware power measurement via external instrumentation.
- ONNX INT8 dynamic quantization produces `ConvInteger` operator nodes not supported by the default ONNX Runtime CPU execution provider on the test hardware, rendering all 15 INT8 ONNX models non-functional for latency benchmarking.
- YOLO TFLite export via the `onnx-tf` conversion path causes severe accuracy degradation on 2 of 3 datasets, likely due to post-processing layer incompatibilities.
- Only YOLOv8 nano and small variants were evaluated (medium excluded due to Colab time constraints).
- No cross-validation was performed; results are based on a single stratified 70/15/15 split with seed 42.

---

## 14. AI Usage Disclosure

AI tools (Claude Code by Anthropic) were used for code scaffolding, experiment orchestration, and writing support during this project. All generated code was manually reviewed, and all empirical results were produced by running the actual models on the specified hardware. A detailed AI usage statement is provided in `docs/ai_usage_statement.md`.

---

## 15. References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788.

[2] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[4] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2019, pp. 6105–6114.

[5] A. Howard et al., "Searching for MobileNetV3," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1314–1324.

[6] K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, 2013.

[7] D. Božič, D. Tabernik, and D. Skočaj, "Mixed supervision for surface-defect detection: From weakly to fully supervised learning," *Computers in Industry*, vol. 129, p. 103459, 2021.

[8] M. Wieler and T. Hahn, "Weakly supervised learning for industrial optical inspection," in *DAGM Symposium*, 2007.

[9] ONNX Runtime Contributors, "ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator," 2023. [Online]. Available: https://onnxruntime.ai

[10] TensorFlow Lite Contributors, "TensorFlow Lite: Deploy machine learning models on mobile and edge devices," 2023. [Online]. Available: https://www.tensorflow.org/lite

[11] NVIDIA Corporation, "TensorRT: High-performance deep learning inference optimizer and runtime," 2023. [Online]. Available: https://developer.nvidia.com/tensorrt

[12] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.
