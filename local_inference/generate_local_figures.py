#!/usr/bin/env python3
"""
Generate Local CPU-Inference Figures.

Creates all figures that depend on local CPU benchmark data:
  - Fig 21: YOLO latency comparison (grouped bar, all formats)
  - Fig 22: CNN latency comparison (grouped bar, all formats)
  - Fig 23: FPS comparison — all models horizontal bar
  - Fig 24: Energy (Wh/1000 frames) comparison bar
  - Fig 25: YOLO latency box plot (per format)
  - Fig 26: CNN latency box plot (per format)
  - Fig 27: mAP50 vs Latency scatter with Pareto frontier (YOLO)
  - Fig 28: Accuracy vs Latency scatter with Pareto frontier (CNN)
  - Fig 29: Accuracy vs Model-size scatter (CNN)
  - Fig 30: Radar chart — best models per dataset
  - Fig 31: All-metrics heatmap
  - Fig 32: Combined pipeline comparison (e2e latency by dataset/format)
  - Fig 33: Detection % vs Classification % stacked bar (combined pipeline)

Usage:
    python generate_local_figures.py --results_dir ./results
    python generate_local_figures.py --results_dir ./results --dpi 150 --format png
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette('tab10')
plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi':   100,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

FORMAT_ORDER  = ['pytorch', 'onnx_fp32', 'onnx_int8', 'tflite', 'tflite_fp32', 'tflite_int8']
DATASET_ORDER = ['NEU-DET', 'DAGM', 'KSDD2']
MODEL_COLOURS = {
    'yolov8n':           PALETTE[0],
    'yolov8s':           PALETTE[1],
    'yolov8m':           PALETTE[2],
    'resnet50':          PALETTE[3],
    'efficientnet_b0':   PALETTE[4],
    'mobilenet_v3_large':PALETTE[5],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Local Inference Figures')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--format', type=str, nargs='+', default=['png'])
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f'Missing: {path}')
        return pd.DataFrame()
    return pd.read_csv(path)


def save_fig(fig, out_dir: str, name: str, formats: list, dpi: int):
    for fmt in formats:
        p = os.path.join(out_dir, f'{name}.{fmt}')
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        logger.info(f'Saved: {p}')
    plt.close(fig)


def pareto_front(xs, ys, minimize_x=True, maximize_y=True):
    """Return boolean mask of Pareto-optimal points."""
    n = len(xs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, yi = xs[i], ys[i]
            xj, yj = xs[j], ys[j]
            x_ok = (xj <= xi) if minimize_x else (xj >= xi)
            y_ok = (yj >= yi) if maximize_y else (yj <= yi)
            x_strict = (xj < xi) if minimize_x else (xj > xi)
            y_strict = (yj > yi) if maximize_y else (yj < yi)
            if x_ok and y_ok and (x_strict or y_strict):
                mask[i] = False
                break
    return mask


def fmt_order_key(fmt):
    return FORMAT_ORDER.index(fmt) if fmt in FORMAT_ORDER else 99


# ── Figure functions ──────────────────────────────────────────────────────────

def fig21_yolo_latency(perf: pd.DataFrame, out_dir, fmts, dpi):
    """Grouped bar: YOLO latency (ms) by model and format."""
    df = perf[perf.get('type', pd.Series()) == 'yolo'] if 'type' in perf.columns else perf
    if df.empty or 'mean_ms' not in df.columns:
        logger.warning('Fig 21: no YOLO perf data'); return

    df = df.copy()
    df['model_fmt'] = df['model'] + '\n' + df['format']
    pivot = df.pivot_table(index='model', columns='format', values='mean_ms', aggfunc='mean')
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=fmt_order_key))

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind='bar', ax=ax, colormap='tab10')
    ax.set_title('Fig 21 — YOLO Latency by Model & Format (CPU)')
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Mean Latency (ms)')
    ax.legend(title='Format', bbox_to_anchor=(1, 1))
    ax.tick_params(axis='x', rotation=0)
    save_fig(fig, out_dir, 'fig21_yolo_latency_comparison', fmts, dpi)


def fig22_cnn_latency(perf: pd.DataFrame, out_dir, fmts, dpi):
    """Grouped bar: CNN latency by model and format."""
    df = perf[perf.get('type', pd.Series()) == 'cnn'] if 'type' in perf.columns else perf
    if df.empty or 'mean_ms' not in df.columns:
        logger.warning('Fig 22: no CNN perf data'); return

    pivot = df.pivot_table(index='model', columns='format', values='mean_ms', aggfunc='mean')
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=fmt_order_key))

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind='bar', ax=ax, colormap='tab10')
    ax.set_title('Fig 22 — CNN Latency by Model & Format (CPU)')
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Mean Latency (ms)')
    ax.legend(title='Format', bbox_to_anchor=(1, 1))
    ax.tick_params(axis='x', rotation=0)
    save_fig(fig, out_dir, 'fig22_cnn_latency_comparison', fmts, dpi)


def fig23_fps_comparison(perf: pd.DataFrame, out_dir, fmts, dpi):
    """Horizontal bar: FPS for all model+format combos."""
    if perf.empty or 'fps' not in perf.columns:
        logger.warning('Fig 23: no fps data'); return

    df = perf.copy()
    df['label'] = df['model'] + ' / ' + df['format']
    df_sorted = df.sort_values('fps', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.35)))
    colours = [MODEL_COLOURS.get(m, PALETTE[0]) for m in df_sorted['model']]
    ax.barh(df_sorted['label'], df_sorted['fps'], color=colours)
    ax.set_title('Fig 23 — FPS Comparison (All Models, CPU)')
    ax.set_xlabel('Frames per Second')
    ax.axvline(x=25, color='red', linestyle='--', linewidth=1, label='25 FPS target')
    ax.legend()
    save_fig(fig, out_dir, 'fig23_fps_comparison', fmts, dpi)


def fig24_energy_comparison(energy: pd.DataFrame, out_dir, fmts, dpi):
    """Bar: Wh per 1000 frames for each model+format."""
    if energy.empty or 'wh_per_1000' not in energy.columns:
        logger.warning('Fig 24: no energy data'); return

    df = energy.copy()
    df['label'] = df['model'] + '\n' + df['format']
    df_sorted = df.sort_values('wh_per_1000', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    colours = [MODEL_COLOURS.get(m, PALETTE[0]) for m in df_sorted['model']]
    ax.bar(range(len(df_sorted)), df_sorted['wh_per_1000'], color=colours)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['label'], rotation=45, ha='right', fontsize=8)
    ax.set_title('Fig 24 — Energy Consumption (Wh / 1000 Frames, CPU)')
    ax.set_ylabel('Wh per 1000 Frames')
    save_fig(fig, out_dir, 'fig24_energy_comparison', fmts, dpi)


def fig25_yolo_boxplot(perf: pd.DataFrame, out_dir, fmts, dpi):
    """Box plot of YOLO latency distribution per format."""
    df = perf[perf.get('type', pd.Series()) == 'yolo'] if 'type' in perf.columns else perf
    if df.empty or 'mean_ms' not in df.columns:
        logger.warning('Fig 25: no YOLO perf data'); return
    # Use mean_ms as proxy (full latency arrays not stored in CSV — plot as scatter+bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    formats = sorted(df['format'].unique(), key=fmt_order_key)
    data = [df[df['format'] == f]['mean_ms'].dropna().values for f in formats]
    bp = ax.boxplot(data, labels=formats, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
    ax.set_title('Fig 25 — YOLO Latency Distribution per Format (CPU)')
    ax.set_xlabel('Format')
    ax.set_ylabel('Mean Latency (ms)')
    ax.tick_params(axis='x', rotation=20)
    save_fig(fig, out_dir, 'fig25_yolo_latency_boxplot', fmts, dpi)


def fig26_cnn_boxplot(perf: pd.DataFrame, out_dir, fmts, dpi):
    """Box plot of CNN latency distribution per format."""
    df = perf[perf.get('type', pd.Series()) == 'cnn'] if 'type' in perf.columns else perf
    if df.empty or 'mean_ms' not in df.columns:
        logger.warning('Fig 26: no CNN perf data'); return

    fig, ax = plt.subplots(figsize=(8, 5))
    formats = sorted(df['format'].unique(), key=fmt_order_key)
    data = [df[df['format'] == f]['mean_ms'].dropna().values for f in formats]
    bp = ax.boxplot(data, labels=formats, patch_artist=True)
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
    ax.set_title('Fig 26 — CNN Latency Distribution per Format (CPU)')
    ax.set_xlabel('Format')
    ax.set_ylabel('Mean Latency (ms)')
    ax.tick_params(axis='x', rotation=20)
    save_fig(fig, out_dir, 'fig26_cnn_latency_boxplot', fmts, dpi)


def fig27_yolo_pareto(master_yolo: pd.DataFrame, out_dir, fmts, dpi):
    """mAP50 vs Latency scatter with Pareto frontier for YOLO."""
    if master_yolo.empty or 'latency_ms' not in master_yolo.columns or 'map50' not in master_yolo.columns:
        logger.warning('Fig 27: no YOLO master data (missing latency_ms or map50)'); return
    df = master_yolo.dropna(subset=['map50', 'latency_ms'])
    if df.empty:
        logger.warning('Fig 27: no YOLO master data'); return

    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in DATASET_ORDER:
        sub = df[df['dataset'] == ds]
        if sub.empty: continue
        ax.scatter(sub['latency_ms'], sub['map50'], label=ds, s=60, alpha=0.7)

    # Pareto
    xs = df['latency_ms'].values
    ys = df['map50'].values
    mask = pareto_front(xs, ys, minimize_x=True, maximize_y=True)
    px = xs[mask]; py = ys[mask]
    order = np.argsort(px)
    ax.plot(px[order], py[order], 'k--', linewidth=1.5, label='Pareto frontier')
    ax.scatter(px, py, c='black', s=80, zorder=5)

    ax.set_title('Fig 27 — YOLO: mAP50 vs Latency (Pareto Frontier)')
    ax.set_xlabel('Mean Latency (ms)')
    ax.set_ylabel('mAP50')
    ax.legend()
    save_fig(fig, out_dir, 'fig27_yolo_pareto_acc_vs_latency', fmts, dpi)


def fig28_cnn_pareto(master_cnn: pd.DataFrame, out_dir, fmts, dpi):
    """Accuracy vs Latency scatter with Pareto frontier for CNN."""
    if master_cnn.empty or 'latency_ms' not in master_cnn.columns or 'accuracy' not in master_cnn.columns:
        logger.warning('Fig 28: no CNN master data (missing latency_ms or accuracy)'); return
    df = master_cnn.dropna(subset=['accuracy', 'latency_ms'])
    if df.empty:
        logger.warning('Fig 28: no CNN master data'); return

    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in DATASET_ORDER:
        sub = df[df['dataset'] == ds]
        if sub.empty: continue
        ax.scatter(sub['latency_ms'], sub['accuracy'], label=ds, s=60, alpha=0.7)

    xs = df['latency_ms'].values
    ys = df['accuracy'].values
    mask = pareto_front(xs, ys, minimize_x=True, maximize_y=True)
    px = xs[mask]; py = ys[mask]
    order = np.argsort(px)
    ax.plot(px[order], py[order], 'k--', linewidth=1.5, label='Pareto frontier')
    ax.scatter(px, py, c='black', s=80, zorder=5)

    ax.set_title('Fig 28 — CNN: Accuracy vs Latency (Pareto Frontier)')
    ax.set_xlabel('Mean Latency (ms)')
    ax.set_ylabel('Accuracy')
    ax.legend()
    save_fig(fig, out_dir, 'fig28_cnn_pareto_acc_vs_latency', fmts, dpi)


def fig29_cnn_size_scatter(master_cnn: pd.DataFrame, out_dir, fmts, dpi):
    """Accuracy vs model size scatter for CNN variants."""
    if master_cnn.empty or 'size_mb' not in master_cnn.columns or 'accuracy' not in master_cnn.columns:
        logger.warning('Fig 29: no CNN size data (missing size_mb or accuracy)'); return
    df = master_cnn.dropna(subset=['accuracy', 'size_mb'])
    if df.empty:
        logger.warning('Fig 29: no CNN size data'); return

    fig, ax = plt.subplots(figsize=(8, 6))
    for mv in df['model'].unique():
        sub = df[df['model'] == mv]
        c = MODEL_COLOURS.get(mv, PALETTE[0])
        ax.scatter(sub['size_mb'], sub['accuracy'], label=mv, color=c, s=70, alpha=0.8)

    ax.set_title('Fig 29 — CNN: Accuracy vs Model Size')
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Accuracy')
    ax.legend(title='Model')
    save_fig(fig, out_dir, 'fig29_cnn_acc_vs_size', fmts, dpi)


def fig30_radar(master_yolo: pd.DataFrame, master_cnn: pd.DataFrame, out_dir, fmts, dpi):
    """Radar chart comparing best YOLO and CNN models per dataset."""
    metrics_yolo = ['map50', 'fps', 'wh_per_1000_inv', 'size_mb_inv']
    labels_radar = ['mAP50', 'FPS', '1/Energy', '1/Size']

    rows = []
    for ds in DATASET_ORDER:
        for mdf, mtype, acc_col in [(master_yolo, 'yolo', 'map50'),
                                    (master_cnn, 'cnn', 'accuracy')]:
            if mdf.empty or acc_col not in mdf.columns: continue
            sub = mdf[mdf['dataset'] == ds]
            if sub.empty: continue
            best = sub.loc[sub[acc_col].idxmax()]
            rows.append({
                'label': f'{ds} ({mtype})',
                'acc': float(best.get(acc_col, 0)),
                'fps': float(best.get('fps', 0)) / 100,
                'energy_inv': 1.0 / (float(best.get('wh_per_1000', 1)) + 1e-9) * 0.001,
                'size_inv': 1.0 / (float(best.get('size_mb', 1)) + 1e-9) * 10,
            })

    if not rows:
        logger.warning('Fig 30: no data for radar'); return

    cats = ['Accuracy', 'FPS (norm)', '1/Energy', '1/Size']
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, row in enumerate(rows):
        vals = [row['acc'], row['fps'], row['energy_inv'], row['size_inv']]
        # Normalise to [0,1]
        max_vals = np.array([1.0, max(r['fps'] for r in rows) + 1e-9,
                             max(r['energy_inv'] for r in rows) + 1e-9,
                             max(r['size_inv'] for r in rows) + 1e-9])
        norm_vals = (np.array(vals) / max_vals).tolist()
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, label=row['label'], linewidth=2)
        ax.fill(angles, norm_vals, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), cats)
    ax.set_title('Fig 30 — Radar: Best Models per Dataset', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    save_fig(fig, out_dir, 'fig30_radar_best_models', fmts, dpi)


def fig31_all_metrics_heatmap(master_all: pd.DataFrame, out_dir, fmts, dpi):
    """Heatmap of normalised metrics for all model+format combos."""
    if master_all.empty:
        logger.warning('Fig 31: no master_all data'); return

    num_cols = ['map50', 'accuracy', 'f1_macro', 'latency_ms', 'fps', 'wh_per_1000', 'size_mb']
    avail = [c for c in num_cols if c in master_all.columns]
    if not avail:
        logger.warning('Fig 31: no numeric cols'); return

    df = master_all[['dataset', 'model', 'format'] + avail].dropna(subset=avail[:1])
    df['row_label'] = df['dataset'] + ' | ' + df['model'] + ' | ' + df['format']
    heat = df.set_index('row_label')[avail]
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(8, len(avail) * 1.2), max(6, len(df) * 0.3)))
    sns.heatmap(heat_norm, ax=ax, cmap='RdYlGn', annot=heat.round(3).astype(str),
                fmt='', linewidths=0.3, cbar_kws={'label': 'Normalised value'})
    ax.set_title('Fig 31 — All Metrics Heatmap (Normalised)')
    ax.tick_params(axis='y', labelsize=7)
    save_fig(fig, out_dir, 'fig31_all_metrics_heatmap', fmts, dpi)


def fig32_pipeline_latency(combined: pd.DataFrame, out_dir, fmts, dpi):
    """Grouped bar: combined pipeline e2e latency by dataset and format."""
    if combined.empty or 'mean_e2e_ms' not in combined.columns:
        logger.warning('Fig 32: no combined pipeline data'); return

    pivot = combined.pivot_table(index='dataset', columns='format',
                                 values='mean_e2e_ms', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title('Fig 32 — Combined Pipeline: End-to-End Latency per Dataset')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean E2E Latency (ms)')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Format', bbox_to_anchor=(1, 1))
    save_fig(fig, out_dir, 'fig32_combined_pipeline_latency', fmts, dpi)


def fig33_pipeline_breakdown(combined: pd.DataFrame, out_dir, fmts, dpi):
    """Stacked bar: detection% vs classification% of pipeline time."""
    if combined.empty or 'det_pct' not in combined.columns:
        logger.warning('Fig 33: no pipeline breakdown data'); return

    df = combined[['dataset', 'yolo_model', 'cnn_model', 'format', 'det_pct', 'cls_pct']].dropna()
    df = df.copy()
    df['label'] = df['dataset'] + '\n' + df['format']
    df_grouped = df.groupby('label')[['det_pct', 'cls_pct']].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(df_grouped))
    ax.bar(df_grouped.index, df_grouped['det_pct'], label='Detection %', color=PALETTE[0])
    ax.bar(df_grouped.index, df_grouped['cls_pct'], bottom=df_grouped['det_pct'],
           label='Classification %', color=PALETTE[1])
    other = 100 - df_grouped['det_pct'] - df_grouped['cls_pct']
    ax.bar(df_grouped.index, other.clip(lower=0),
           bottom=df_grouped['det_pct'] + df_grouped['cls_pct'],
           label='Other %', color=PALETTE[7])

    ax.set_title('Fig 33 — Pipeline Time Breakdown: Detection vs Classification')
    ax.set_ylabel('% of E2E Time')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    save_fig(fig, out_dir, 'fig33_pipeline_breakdown', fmts, dpi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join(args.results_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f'Generate Local Figures — results_dir={args.results_dir}  output={out_dir}')

    tbl = os.path.join(args.results_dir, 'tables')
    perf        = safe_read(os.path.join(tbl, 'performance_all_models.csv'))
    energy      = safe_read(os.path.join(tbl, 'energy_estimation.csv'))
    master_yolo = safe_read(os.path.join(tbl, 'master_results_yolo.csv'))
    master_cnn  = safe_read(os.path.join(tbl, 'master_results_cnn.csv'))
    master_all  = safe_read(os.path.join(tbl, 'master_results_all.csv'))
    combined    = safe_read(os.path.join(tbl, 'combined_pipeline_results.csv'))

    fmts = args.format
    dpi  = args.dpi

    fig21_yolo_latency(perf, out_dir, fmts, dpi)
    fig22_cnn_latency(perf, out_dir, fmts, dpi)
    fig23_fps_comparison(perf, out_dir, fmts, dpi)
    fig24_energy_comparison(energy, out_dir, fmts, dpi)
    fig25_yolo_boxplot(perf, out_dir, fmts, dpi)
    fig26_cnn_boxplot(perf, out_dir, fmts, dpi)
    fig27_yolo_pareto(master_yolo, out_dir, fmts, dpi)
    fig28_cnn_pareto(master_cnn, out_dir, fmts, dpi)
    fig29_cnn_size_scatter(master_cnn, out_dir, fmts, dpi)
    fig30_radar(master_yolo, master_cnn, out_dir, fmts, dpi)
    fig31_all_metrics_heatmap(master_all, out_dir, fmts, dpi)
    fig32_pipeline_latency(combined, out_dir, fmts, dpi)
    fig33_pipeline_breakdown(combined, out_dir, fmts, dpi)

    saved = list(Path(out_dir).glob('fig*.png')) + list(Path(out_dir).glob('fig*.pdf'))
    print('\n' + '='*60)
    print(f'Generated {len(saved)} figure files in: {out_dir}')
    for p in sorted(saved):
        print(f'  {p.name}')
    print('='*60)


if __name__ == '__main__':
    main()
