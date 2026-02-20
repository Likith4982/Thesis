#!/usr/bin/env python3
"""
Compare Results — Aggregate all benchmark results into master tables.

Reads all CSV results from inference, performance, and energy scripts.
Creates master results tables, Pareto-optimal configurations, and
cross-pipeline (YOLO vs CNN vs Combined) comparisons.

Usage:
    python compare_results.py --results_dir ./results
    python compare_results.py --results_dir ./results --pareto_metric accuracy_vs_latency
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare All Results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing all result CSVs')
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_read(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        logger.warning(f'Missing: {path}')
        return None
    try:
        df = pd.read_csv(path)
        logger.info(f'Loaded {len(df)} rows from {path}')
        return df
    except Exception as e:
        logger.warning(f'Could not read {path}: {e}')
        return None


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str,
                 minimize_x: bool = True, maximize_y: bool = True) -> pd.DataFrame:
    """Return rows on the Pareto frontier for (x_col, y_col)."""
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return df.iloc[:0]

    idx = sub.index.tolist()
    pareto_idx = []
    for i in idx:
        dominated = False
        xi, yi = sub.loc[i, x_col], sub.loc[i, y_col]
        for j in idx:
            if i == j:
                continue
            xj, yj = sub.loc[j, x_col], sub.loc[j, y_col]
            # j dominates i if it is at least as good on both axes and strictly better on one
            x_ok = (xj <= xi) if minimize_x else (xj >= xi)
            y_ok = (yj >= yi) if maximize_y else (yj <= yi)
            x_strict = (xj < xi) if minimize_x else (xj > xi)
            y_strict = (yj > yi) if maximize_y else (yj < yi)
            if x_ok and y_ok and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)
    return df.loc[pareto_idx].copy()


def weighted_rank(df: pd.DataFrame, metrics: dict) -> pd.Series:
    """
    metrics: {col: (weight, higher_is_better)}
    Returns a score Series (higher = better overall).
    """
    score = pd.Series(0.0, index=df.index)
    for col, (w, higher) in metrics.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            norm = pd.Series(0.5, index=df.index)
        else:
            norm = (vals - lo) / (hi - lo)
        if not higher:
            norm = 1 - norm
        score += w * norm.fillna(0)
    return score


# ── Section builders ──────────────────────────────────────────────────────────

def build_yolo_master(yolo_inf: pd.DataFrame | None,
                      perf: pd.DataFrame | None,
                      energy: pd.DataFrame | None) -> pd.DataFrame | None:
    if yolo_inf is None:
        return None
    base = yolo_inf.copy()
    key = ['dataset', 'model', 'format']

    if perf is not None:
        yolo_perf = perf[perf.get('type', pd.Series()) == 'yolo'][
            key + ['mean_ms', 'p95_ms', 'fps', 'size_mb']
        ].rename(columns={'mean_ms': 'latency_ms', 'p95_ms': 'p95_ms'})
        base = base.merge(yolo_perf, on=key, how='left')

    if energy is not None:
        yolo_en = energy[energy.get('type', pd.Series()) == 'yolo'][
            key + ['wh_per_1000', 'avg_cpu_pct']
        ]
        base = base.merge(yolo_en, on=key, how='left')

    return base


def build_cnn_master(cnn_inf: pd.DataFrame | None,
                     perf: pd.DataFrame | None,
                     energy: pd.DataFrame | None) -> pd.DataFrame | None:
    if cnn_inf is None:
        return None
    base = cnn_inf.copy()
    key = ['dataset', 'model', 'format']

    if perf is not None:
        cnn_perf = perf[perf.get('type', pd.Series()) == 'cnn'][
            key + ['mean_ms', 'p95_ms', 'fps', 'size_mb']
        ].rename(columns={'mean_ms': 'latency_ms'})
        base = base.merge(cnn_perf, on=key, how='left')

    if energy is not None:
        cnn_en = energy[energy.get('type', pd.Series()) == 'cnn'][
            key + ['wh_per_1000', 'avg_cpu_pct']
        ]
        base = base.merge(cnn_en, on=key, how='left')

    return base


def build_pareto_tables(yolo_master: pd.DataFrame | None,
                        cnn_master: pd.DataFrame | None) -> dict:
    """Return dict of labelled pareto DataFrames."""
    tables = {}

    if yolo_master is not None and not yolo_master.empty:
        # mAP50 vs latency
        if 'map50' in yolo_master.columns and 'latency_ms' in yolo_master.columns:
            tables['yolo_pareto_acc_vs_latency'] = pareto_front(
                yolo_master, 'latency_ms', 'map50', minimize_x=True, maximize_y=True)
        # mAP50 vs size
        if 'map50' in yolo_master.columns and 'size_mb' in yolo_master.columns:
            tables['yolo_pareto_acc_vs_size'] = pareto_front(
                yolo_master, 'size_mb', 'map50', minimize_x=True, maximize_y=True)
        # mAP50 vs energy
        if 'map50' in yolo_master.columns and 'wh_per_1000' in yolo_master.columns:
            tables['yolo_pareto_acc_vs_energy'] = pareto_front(
                yolo_master, 'wh_per_1000', 'map50', minimize_x=True, maximize_y=True)

    if cnn_master is not None and not cnn_master.empty:
        if 'accuracy' in cnn_master.columns and 'latency_ms' in cnn_master.columns:
            tables['cnn_pareto_acc_vs_latency'] = pareto_front(
                cnn_master, 'latency_ms', 'accuracy', minimize_x=True, maximize_y=True)
        if 'accuracy' in cnn_master.columns and 'size_mb' in cnn_master.columns:
            tables['cnn_pareto_acc_vs_size'] = pareto_front(
                cnn_master, 'size_mb', 'accuracy', minimize_x=True, maximize_y=True)
        if 'accuracy' in cnn_master.columns and 'wh_per_1000' in cnn_master.columns:
            tables['cnn_pareto_acc_vs_energy'] = pareto_front(
                cnn_master, 'wh_per_1000', 'accuracy', minimize_x=True, maximize_y=True)

    return tables


def build_best_models(yolo_master: pd.DataFrame | None,
                      cnn_master: pd.DataFrame | None) -> pd.DataFrame:
    """Best model per dataset per pipeline using weighted scoring."""
    rows = []

    if yolo_master is not None and not yolo_master.empty:
        metrics = {
            'map50':       (0.40, True),
            'latency_ms':  (0.30, False),
            'wh_per_1000': (0.15, False),
            'size_mb':     (0.15, False),
        }
        yolo_master = yolo_master.copy()
        yolo_master['score'] = weighted_rank(yolo_master, metrics)
        for ds, grp in yolo_master.groupby('dataset'):
            best = grp.loc[grp['score'].idxmax()]
            rows.append({
                'pipeline': 'yolo', 'dataset': ds,
                'model': best.get('model', ''), 'format': best.get('format', ''),
                'map50': best.get('map50', None), 'latency_ms': best.get('latency_ms', None),
                'fps': best.get('fps', None), 'wh_per_1000': best.get('wh_per_1000', None),
                'score': round(float(best['score']), 4),
            })

    if cnn_master is not None and not cnn_master.empty:
        metrics = {
            'accuracy':    (0.40, True),
            'f1_macro':    (0.15, True),
            'latency_ms':  (0.25, False),
            'wh_per_1000': (0.10, False),
            'size_mb':     (0.10, False),
        }
        cnn_master = cnn_master.copy()
        cnn_master['score'] = weighted_rank(cnn_master, metrics)
        for ds, grp in cnn_master.groupby('dataset'):
            best = grp.loc[grp['score'].idxmax()]
            rows.append({
                'pipeline': 'cnn', 'dataset': ds,
                'model': best.get('model', ''), 'format': best.get('format', ''),
                'accuracy': best.get('accuracy', None), 'latency_ms': best.get('latency_ms', None),
                'fps': best.get('fps', None), 'wh_per_1000': best.get('wh_per_1000', None),
                'score': round(float(best['score']), 4),
            })

    return pd.DataFrame(rows)


def build_format_comparison(yolo_master: pd.DataFrame | None,
                            cnn_master: pd.DataFrame | None) -> pd.DataFrame:
    """Mean metrics per format across all datasets and model variants."""
    rows = []

    def _agg(df, pipeline, acc_col):
        if df is None or df.empty:
            return
        num_cols = [acc_col, 'latency_ms', 'fps', 'size_mb', 'wh_per_1000']
        for fmt, grp in df.groupby('format'):
            row = {'pipeline': pipeline, 'format': fmt}
            for c in num_cols:
                if c in grp.columns:
                    row[f'mean_{c}'] = round(float(pd.to_numeric(grp[c], errors='coerce').mean()), 4)
            rows.append(row)

    _agg(yolo_master, 'yolo', 'map50')
    _agg(cnn_master, 'cnn', 'accuracy')
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tbl = f'{args.results_dir}/tables'
    os.makedirs(tbl, exist_ok=True)
    logger.info(f'Compare Results — results_dir={args.results_dir}')

    # Load input CSVs
    yolo_inf = safe_read(f'{tbl}/yolo_cpu_inference.csv')
    cnn_inf  = safe_read(f'{tbl}/cnn_cpu_inference.csv')
    perf     = safe_read(f'{tbl}/performance_all_models.csv')
    energy   = safe_read(f'{tbl}/energy_estimation.csv')
    combined = safe_read(f'{tbl}/combined_pipeline_results.csv')

    if all(x is None for x in [yolo_inf, cnn_inf, perf, energy, combined]):
        logger.error('No result CSVs found. Run inference/performance/energy scripts first.')
        sys.exit(1)

    # Build master tables
    yolo_master = build_yolo_master(yolo_inf, perf, energy)
    cnn_master  = build_cnn_master(cnn_inf, perf, energy)

    saved = []

    def _save(df, name):
        if df is not None and not df.empty:
            p = f'{tbl}/{name}.csv'
            df.to_csv(p, index=False)
            logger.info(f'Saved: {p} ({len(df)} rows)')
            saved.append(p)

    _save(yolo_master, 'master_results_yolo')
    _save(cnn_master,  'master_results_cnn')
    _save(combined,    'master_results_combined')

    # Combined "all" table
    parts = []
    if yolo_master is not None:
        ym = yolo_master.copy(); ym['pipeline'] = 'yolo'; parts.append(ym)
    if cnn_master is not None:
        cm = cnn_master.copy(); cm['pipeline'] = 'cnn'; parts.append(cm)
    if parts:
        all_df = pd.concat(parts, ignore_index=True)
        _save(all_df, 'master_results_all')

    # Pareto tables
    pareto = build_pareto_tables(yolo_master, cnn_master)
    for name, df in pareto.items():
        _save(df, name)

    # Best models summary
    best = build_best_models(yolo_master, cnn_master)
    _save(best, 'best_models_summary')

    # Format comparison
    fmt_cmp = build_format_comparison(yolo_master, cnn_master)
    _save(fmt_cmp, 'format_comparison')

    # ── Print summary ──
    print('\n' + '='*70)
    print('Compare Results Summary')
    print('='*70)

    if yolo_master is not None and not yolo_master.empty:
        print('\n--- YOLO Master (head) ---')
        cols = ['dataset', 'model', 'format', 'map50', 'latency_ms', 'fps', 'wh_per_1000']
        avail = [c for c in cols if c in yolo_master.columns]
        print(yolo_master[avail].head(10).to_string(index=False))

    if cnn_master is not None and not cnn_master.empty:
        print('\n--- CNN Master (head) ---')
        cols = ['dataset', 'model', 'format', 'accuracy', 'f1_macro', 'latency_ms', 'fps']
        avail = [c for c in cols if c in cnn_master.columns]
        print(cnn_master[avail].head(10).to_string(index=False))

    if not best.empty:
        print('\n--- Best Models Per Dataset ---')
        print(best.to_string(index=False))

    if not fmt_cmp.empty:
        print('\n--- Format Comparison ---')
        print(fmt_cmp.to_string(index=False))

    print(f'\nSaved {len(saved)} tables to {tbl}/')


if __name__ == '__main__':
    main()
