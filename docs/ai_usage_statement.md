# AI Usage Statement

**Project:** Computer Vision and Deep Learning for Real-Time Quality Inspection in Manufacturing
**Author:** Likith
**Date:** February 2026

---

## AI Tools Used

- **Claude Code** (Anthropic) — an AI-powered coding assistant used via the command-line interface

---

## Where AI Was Used

### 1. Code Scaffolding
AI was used to generate initial boilerplate and structure for:
- Colab notebook pipelines (data preprocessing, training loops, model export)
- Local inference scripts (CPU benchmarking, energy estimation, figure generation)
- Results aggregation and comparison logic (`compare_results.py`)

All generated code was reviewed, tested, and modified by the author before use.

### 2. Experiment Orchestration
AI assisted with:
- Designing the experiment execution order (Phases 2–6)
- Structuring the directory layout for models, results, and scripts
- Debugging runtime errors (e.g., TFLite export path resolution, ONNX compatibility issues)

### 3. Writing Support
AI assisted with:
- Drafting sections of the thesis explanation document
- Formatting results tables and figure references
- Structuring the README and documentation

### 4. Debugging and Troubleshooting
AI was used to diagnose and resolve:
- TFLite conversion failures and dependency issues
- ONNX INT8 `ConvInteger` runtime errors
- Missing data pipeline issues (e.g., `latency_ms` column not being merged into master tables)

---

## Where AI Was NOT the Source

### Empirical Results
All experimental results (accuracy metrics, latency measurements, energy estimates, FPS numbers) were produced by:
- Running actual model training on Google Colab T4 GPU
- Running actual model inference on the local i5-1335U CPU
- Using standard ML frameworks (PyTorch, Ultralytics, ONNX Runtime, TFLite)

AI did not generate, fabricate, or modify any numerical results.

### Model Training
All 15 models (6 YOLO + 9 CNN) were trained using standard deep learning frameworks with documented hyperparameters. AI did not train models or select hyperparameters autonomously.

### Dataset Selection and Preparation
The choice of NEU-DET, DAGM 2007, and KSDD2 datasets was made by the author based on literature review. Data preprocessing (annotation parsing, format conversion, splitting) followed standard practices.

### Research Questions and Methodology
The research questions, dual-pipeline architecture, and evaluation methodology were designed by the author. AI provided implementation support but did not define the research direction.

---

## Provenance Artifacts

The master prompt used to guide AI-assisted code generation is preserved in:
- `FINAL_prompt_with_detection_and_classification.md` (1,151 lines)

This file documents the full context provided to Claude Code, including project constraints, dataset specifications, model configurations, and deployment targets.

---

## Validation Statement

All AI-generated code and documentation were:
1. Manually reviewed by the author before inclusion
2. Tested against real data and hardware
3. Verified to produce correct and reproducible results
4. Modified as needed to fix errors or improve quality

No AI-generated output was included without human review and validation.
