# Thesis Project TODO

## Context Snapshot (from current folder scan)
- `thesis_project/docs/thesis_explanation_document.md` is still marked "in progress" and contains explicit TODO placeholders.
- Local result artifacts exist in `thesis_project/results/tables` and `thesis_project/results/figures` (currently `fig21` to `fig33`).
- Colab result artifacts also exist in `thesis_project/results/Colab_results/results/{tables,figures,tensorrt_benchmarks}` and must be included in thesis summary.
- `thesis_project/results/tables/performance_all_models.csv` contains 15 `error` rows (ONNX INT8 `ConvInteger` not implemented on current CPU runtime path).
- Repository is heavy for Git upload (approx.): `models` ~3.6 GB, `venv` ~2.7 GB, `datasets` ~2.3 GB, `runs` ~89 MB.
- AI prompt provenance exists in `thesis_project/FINAL_prompt_with_detection_and_classification.md`, but no formal AI-usage statement exists in docs.

## Priority Legend
- `P0` = high priority (do first)
- `P1` = important
- `P2` = improvement

## Core Tasks

### 1. Task 1 - Update `@thesis_project/docs/thesis_explanation_document.md` with obtained results, summarized tables, figures, and current methodology (`P0`)
- [ ] Replace "in progress" wording and remove TODO placeholders in Sections 10 and 12.
- [ ] Add a "Final Experimental Setup Used" section with the actually executed setup (datasets, model variants, backends, hardware, metrics).
- [ ] Add a "Key Quantitative Results" section sourced from:
  - Colab benchmark tables (training/test accuracy and GPU references):
    - `thesis_project/results/Colab_results/results/tables/TABLE_YOLO_DetectionBenchmark_All.csv`
    - `thesis_project/results/Colab_results/results/tables/TABLE_CNN_ClassificationBenchmark_All.csv`
    - `thesis_project/results/Colab_results/results/tables/TABLE_Optimization_Summary.csv`
    - `thesis_project/results/Colab_results/results/tables/TABLE_TensorRT_Benchmarks.csv`
    - `thesis_project/results/Colab_results/results/tables/TABLE_Recommended_Deployment.csv`
  - Local CPU benchmark tables:
  - `thesis_project/results/tables/master_results_yolo.csv`
  - `thesis_project/results/tables/master_results_cnn.csv`
  - `thesis_project/results/tables/master_results_combined.csv`
  - `thesis_project/results/tables/best_models_summary.csv`
- [ ] Add an explicit "Source-of-Truth Split" note:
  - Colab results = training/test-set model quality + GPU/TensorRT reference
  - Local results = CPU latency/FPS/energy and edge-deployment behavior
- [ ] Add compact summary tables directly in the thesis doc:
  - Best YOLO per dataset (mAP50, latency, FPS, energy)
  - Best CNN per dataset (accuracy, latency, FPS, energy)
  - Best combined pipeline configurations per dataset
- [ ] Link and interpret both figure sets:
  - Colab figures in `thesis_project/results/Colab_results/results/figures`
  - Local CPU figures (`fig21`-`fig33`) in `thesis_project/results/figures`
- [ ] Add a "Methodology and Validity Notes" subsection documenting known runtime constraints (for example ONNX INT8 `ConvInteger` CPU limitation) and how affected rows are treated.
- [ ] Finalize references with complete citation formatting (replace placeholder list with thesis-ready bibliography format).

### 2. Task 2 - Update `README` (`P1`)
- [ ] Add a "Current Results Snapshot" section (top-performing models + known limitations).
- [ ] Add a "Results Source Map" section that separates:
  - Colab (training/evaluation + TensorRT)
  - Local machine (CPU inference/energy/combined pipeline)
- [ ] Add a "Reproducibility Path" section: exact command order from inference to table/figure generation.
- [ ] Add a "Repository Modes" section:
  - Full research workspace
  - Lightweight Git-upload version
- [ ] Add a "Known Issues" section (for example unsupported ONNX INT8 ops on some CPU setups).
- [ ] Ensure README content matches actual implemented variants/files in this folder.

### 3. Task 3 - Clearly define and document the use of AI across the project (`P0`)
- [ ] Create `thesis_project/docs/ai_usage_statement.md`.
- [ ] Document where AI was used (prompting, code scaffolding, writing support, planning) and where it was not the source of empirical results.
- [ ] Reference provenance artifact(s):
  - `thesis_project/FINAL_prompt_with_detection_and_classification.md`
- [ ] Add validation statement: all generated code/results were manually reviewed and empirically verified before inclusion.
- [ ] Add short AI-disclosure links in both:
  - `thesis_project/README.md`
  - `thesis_project/docs/thesis_explanation_document.md`

### 4. Task 4 - Clean codebase/folder for Git upload readiness (`P1`)
- [ ] Add `.gitignore` in `thesis_project/` to exclude:
  - `venv/`, `datasets/`, `models/`, `runs/`, large generated artifacts, cache/temp files
- [ ] Add `.gitattributes` (LFS plan if keeping selected binaries).
- [ ] Keep only lightweight reproducible assets in Git:
  - scripts, notebooks, docs, requirements, small summary CSVs, selected final figures
- [ ] Decide whether `thesis_project/results/Colab_results` is kept fully, partially, or archived externally to avoid duplicating similar outputs.
- [ ] Strip notebook outputs (especially large embedded images) before commit.
- [ ] Add `scripts/prepare_repo_for_git.ps1` (or `.py`) to automate cleanup and checks.
- [ ] Run a pre-upload audit:
  - file-size threshold check (for example >50 MB)
  - broken path check
  - results/doc cross-reference check

## Extra Assumption-Based Improvement Tasks

### 5. Result consistency and quality control (`P1`)
- [ ] Build a `results_validation.py` script that flags:
  - rows with `status=error`
  - impossible/placeholder metrics (for example zero detection scores where not expected)
  - missing latency/energy fields in ranking inputs
- [ ] Update ranking logic to explicitly exclude invalid rows before selecting "best models".

### 6. Reproducibility hardening (`P1`)
- [ ] Pin dependency versions in both requirements files (or provide lock files).
- [ ] Add deterministic seed tracking and metadata export for each run.
- [ ] Add one command/script to regenerate all summary tables from raw outputs.

### 7. Thesis figure-table alignment pass (`P1`)
- [ ] Create a figure/table index map (`docs/figure_table_index.md`) with:
  - Thesis figure/table number
  - Source file path
  - Script/notebook that generated it
- [ ] Ensure numbering in thesis text matches actual available files.

### 8. Code cleanup and maintainability (`P2`)
- [ ] Add basic linting/formatting setup (for example `ruff` + `black`) and run on `thesis_project/local_inference`.
- [ ] Refactor repeated helper logic shared by inference scripts into a common utility module.
- [ ] Improve exception logging so failed formats are reported consistently and traceably.

### 9. Research depth extensions (`P2`)
- [ ] Add repeated-run statistical reporting (mean +/- std, confidence intervals) for key metrics.
- [ ] Add sensitivity analysis on confidence threshold and IoU threshold for deployment recommendations.
- [ ] Add a short cost-performance deployment recommendation matrix (accuracy, latency, energy, model size).

## Suggested Execution Order
1. Task 1 (P0)
2. Task 3 (P0)
3. Task 2 (P1)
4. Task 4 (P1)
5. Tasks 5-7 (P1)
6. Tasks 8-9 (P2)
