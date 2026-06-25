from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRAIN_METRIC = "train_total_loss"
DEFAULT_VAL_METRIC = "val_total_loss_epoch"
DEFAULT_METRICS = (DEFAULT_TRAIN_METRIC, DEFAULT_VAL_METRIC)
DEFAULT_OUTPUT_DIR = "evaluation_results/loss_visiualization"
AUTO_X_AXIS_PRIORITY = ("epoch_exact", "epoch", "step")
EPSILON = 1e-12
PLOT_COLORS = ["#0F4C81", "#D1495B", "#2E8B57", "#7A5CFA", "#C17C00", "#008B8B"]


@dataclass
class MetricSeries:
    label: str
    metric: str
    x_column: str
    x_values: list[float]
    y_values: list[float]
    source: Path


@dataclass
class ExperimentRun:
    label: str
    knn_connectivity: int
    metrics_csv: Path
    hparams_yaml: Path | None
    dataframe: pd.DataFrame
    train_series: MetricSeries
    val_series: MetricSeries


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def fraction_between_zero_and_one(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def infer_label(csv_path: Path) -> str:
    if csv_path.name == "metrics.csv":
        parents = csv_path.parents
        if len(parents) >= 3 and parents[0].name.startswith("version_") and parents[1].name == "csv_logs":
            return parents[2].name
    return csv_path.stem


def humanize_metric_name(metric: str) -> str:
    aliases = {
        "train_total_loss": "Training Total Loss",
        "val_total_loss_epoch": "Validation Total Loss",
        "val_total_loss_step": "Validation Step Loss",
    }
    if metric in aliases:
        return aliases[metric]
    return metric.replace("_", " ").title()


def smooth_values(values: Sequence[float], window: int) -> list[float]:
    numeric = pd.Series(values, dtype="float64")
    if window <= 1:
        return numeric.tolist()
    return numeric.rolling(window=window, min_periods=1).mean().tolist()


def pick_x_column(metric_rows: pd.DataFrame, requested_axis: str) -> str | None:
    if requested_axis != "auto":
        if requested_axis in metric_rows.columns and metric_rows[requested_axis].notna().any():
            return requested_axis
        return None

    for candidate in AUTO_X_AXIS_PRIORITY:
        if candidate in metric_rows.columns and metric_rows[candidate].notna().any():
            return candidate
    return None


def load_metrics_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"metrics CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def extract_metric_series(
    df: pd.DataFrame,
    metric: str,
    label: str,
    source: Path,
    x_axis: str = "auto",
    smooth_window: int = 1,
) -> MetricSeries | None:
    if metric not in df.columns:
        return None

    metric_values = pd.to_numeric(df[metric], errors="coerce")
    metric_rows = df.loc[metric_values.notna()].copy()
    if metric_rows.empty:
        return None

    metric_rows[metric] = pd.to_numeric(metric_rows[metric], errors="coerce")
    x_column = pick_x_column(metric_rows, x_axis)

    if x_column is not None:
        metric_rows["_x"] = pd.to_numeric(metric_rows[x_column], errors="coerce")
        metric_rows = metric_rows.loc[metric_rows["_x"].notna()].sort_values("_x").copy()
        if metric_rows.empty:
            return None
        x_values = metric_rows["_x"].astype(float).tolist()
    else:
        metric_rows = metric_rows.copy()
        x_values = list(range(len(metric_rows)))
        x_column = "index"

    y_values = smooth_values(metric_rows[metric].tolist(), smooth_window)

    return MetricSeries(
        label=label,
        metric=metric,
        x_column=x_column,
        x_values=x_values,
        y_values=y_values,
        source=source,
    )


def build_metric_groups(
    csv_files: Sequence[Path],
    labels: Sequence[str],
    metrics: Sequence[str],
    x_axis: str,
    smooth_window: int,
) -> tuple[dict[str, list[MetricSeries]], list[str]]:
    metric_groups: dict[str, list[MetricSeries]] = {metric: [] for metric in metrics}
    warnings: list[str] = []

    for csv_path, label in zip(csv_files, labels):
        df = load_metrics_csv(csv_path)
        for metric in metrics:
            series = extract_metric_series(
                df=df,
                metric=metric,
                label=label,
                source=csv_path,
                x_axis=x_axis,
                smooth_window=smooth_window,
            )
            if series is None:
                warnings.append(f"Skipped '{metric}' for '{label}' because the column is missing or empty.")
                continue
            metric_groups[metric].append(series)

    return metric_groups, warnings


def prune_empty_metric_groups(metric_groups: dict[str, list[MetricSeries]]) -> dict[str, list[MetricSeries]]:
    return {metric: series_list for metric, series_list in metric_groups.items() if series_list}


def axis_label_for(metric_series: Sequence[MetricSeries]) -> str:
    labels = {series.x_column for series in metric_series}
    if len(labels) == 1:
        return labels.pop()
    return "training progress"


def configure_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_metric_groups(
    metric_groups: dict[str, list[MetricSeries]],
    output_path: Path,
    title: str | None = None,
    dpi: int = 160,
) -> None:
    configure_plot_style()
    metrics = list(metric_groups.keys())
    if not metrics:
        raise ValueError("No metrics were available to plot.")

    figure_height = max(4, 3.8 * len(metrics))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, figure_height), squeeze=False)
    flat_axes = axes.flatten()

    for ax, metric in zip(flat_axes, metrics):
        series_list = metric_groups[metric]
        for color, series in zip(PLOT_COLORS, series_list):
            ax.plot(series.x_values, series.y_values, label=series.label, linewidth=2.0, color=color)

        ax.set_title(humanize_metric_name(metric))
        ax.set_xlabel(axis_label_for(series_list))
        ax.set_ylabel("Loss" if "loss" in metric else "Value")
        ax.legend(frameon=False, ncol=min(3, len(series_list)))

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_knn_connectivity(hparams_path: Path | None) -> int | None:
    if hparams_path is None or not hparams_path.is_file():
        return None
    match = re.search(r"knn_connectivity:\s*(\d+)", hparams_path.read_text(encoding="utf-8", errors="ignore"))
    if match is None:
        return None
    return int(match.group(1))


def discover_knn_runs(
    knn_root: Path,
    train_metric: str,
    val_metric: str,
    x_axis: str,
    smooth_window: int,
) -> tuple[list[ExperimentRun], list[str]]:
    warnings: list[str] = []
    if not knn_root.is_dir():
        raise FileNotFoundError(f"KNN checkpoints directory not found: {knn_root}")

    runs: list[ExperimentRun] = []
    for metrics_csv in sorted(knn_root.glob("SDE-k=*/csv_logs/version_*/metrics.csv")):
        hparams_yaml = metrics_csv.with_name("hparams.yaml")
        knn_value = parse_knn_connectivity(hparams_yaml)
        if knn_value is None:
            warnings.append(f"Skipped '{metrics_csv}' because knn_connectivity was not found in hparams.yaml.")
            continue

        df = load_metrics_csv(metrics_csv)
        label = f"K={knn_value}"
        train_series = extract_metric_series(
            df=df,
            metric=train_metric,
            label=label,
            source=metrics_csv,
            x_axis=x_axis,
            smooth_window=smooth_window,
        )
        val_series = extract_metric_series(
            df=df,
            metric=val_metric,
            label=label,
            source=metrics_csv,
            x_axis=x_axis,
            smooth_window=smooth_window,
        )

        missing_metrics = []
        if train_series is None:
            missing_metrics.append(train_metric)
        if val_series is None:
            missing_metrics.append(val_metric)
        if missing_metrics:
            warnings.append(
                f"Skipped K={knn_value} because required metric columns were missing or empty: {', '.join(missing_metrics)}."
            )
            continue

        runs.append(
            ExperimentRun(
                label=label,
                knn_connectivity=knn_value,
                metrics_csv=metrics_csv,
                hparams_yaml=hparams_yaml if hparams_yaml.is_file() else None,
                dataframe=df,
                train_series=train_series,
                val_series=val_series,
            )
        )

    deduped_runs: list[ExperimentRun] = []
    seen_knn: set[int] = set()
    for run in sorted(runs, key=lambda item: (item.knn_connectivity, item.label)):
        if run.knn_connectivity in seen_knn:
            warnings.append(
                f"Skipped duplicate run for K={run.knn_connectivity}: {run.metrics_csv}. Only the first run is used."
            )
            continue
        seen_knn.add(run.knn_connectivity)
        deduped_runs.append(run)

    if not deduped_runs:
        raise ValueError("No valid KNN runs were found under the provided root directory.")
    return deduped_runs, warnings


def window_length(n_points: int, fraction: float) -> int:
    return min(n_points, max(2, int(math.ceil(n_points * fraction))))


def compute_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    if np.allclose(x_array, x_array[0]):
        return 0.0
    return float(np.polyfit(x_array, y_array, deg=1)[0])


def compute_drop_rate(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    delta_x = float(x_values[-1]) - float(x_values[0])
    if abs(delta_x) < EPSILON:
        return 0.0
    return float((float(y_values[0]) - float(y_values[-1])) / delta_x)


def tail_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pd.Series(values, dtype="float64").std(ddof=0))


def build_summary_dataframe(
    runs: Sequence[ExperimentRun],
    early_fraction: float,
    late_fraction: float,
    improvement_threshold_pct: float,
    convergence_ratio_threshold: float,
    performance_gap_threshold_pct: float,
) -> tuple[pd.DataFrame, int]:
    summary_rows: list[dict[str, float | int | bool | str]] = []

    for run in runs:
        raw_train_series = extract_metric_series(
            df=run.dataframe,
            metric=run.train_series.metric,
            label=run.label,
            source=run.metrics_csv,
            x_axis=run.train_series.x_column if run.train_series.x_column != "index" else "auto",
            smooth_window=1,
        )
        raw_val_series = extract_metric_series(
            df=run.dataframe,
            metric=run.val_series.metric,
            label=run.label,
            source=run.metrics_csv,
            x_axis=run.val_series.x_column if run.val_series.x_column != "index" else "auto",
            smooth_window=1,
        )
        if raw_train_series is None or raw_val_series is None:
            raise ValueError(f"Raw metric extraction failed for {run.label} during summary computation.")

        train_x = raw_train_series.x_values
        train_y = raw_train_series.y_values
        val_x = raw_val_series.x_values
        val_y = raw_val_series.y_values

        early_train_count = window_length(len(train_y), early_fraction)
        early_val_count = window_length(len(val_y), early_fraction)
        late_train_count = window_length(len(train_y), late_fraction)
        late_val_count = window_length(len(val_y), late_fraction)

        early_train_drop_rate = compute_drop_rate(train_x[:early_train_count], train_y[:early_train_count])
        early_val_drop_rate = compute_drop_rate(val_x[:early_val_count], val_y[:early_val_count])
        late_train_slope = compute_slope(train_x[-late_train_count:], train_y[-late_train_count:])
        late_val_slope = compute_slope(val_x[-late_val_count:], val_y[-late_val_count:])
        convergence_ratio = abs(late_val_slope) / max(abs(early_val_drop_rate), EPSILON)

        best_train_loss = float(min(train_y))
        final_train_loss = float(train_y[-1])
        best_val_loss = float(min(val_y))
        final_val_loss = float(val_y[-1])

        summary_rows.append(
            {
                "label": run.label,
                "knn_connectivity": run.knn_connectivity,
                "train_metric": run.train_series.metric,
                "val_metric": run.val_series.metric,
                "best_train_loss": best_train_loss,
                "final_train_loss": final_train_loss,
                "best_val_loss": best_val_loss,
                "final_val_loss": final_val_loss,
                "early_train_drop_rate": early_train_drop_rate,
                "early_val_drop_rate": early_val_drop_rate,
                "late_train_slope": late_train_slope,
                "late_val_slope": late_val_slope,
                "late_train_std": tail_std(train_y[-late_train_count:]),
                "late_val_std": tail_std(val_y[-late_val_count:]),
                "final_train_val_gap": final_val_loss - final_train_loss,
                "convergence_ratio": convergence_ratio,
                "metrics_csv": str(run.metrics_csv),
                "hparams_yaml": str(run.hparams_yaml) if run.hparams_yaml else "",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("knn_connectivity").reset_index(drop=True)
    global_best_val_loss = float(summary_df["best_val_loss"].min())
    summary_df["performance_gap_to_best_pct"] = (
        (summary_df["best_val_loss"] - global_best_val_loss) / max(global_best_val_loss, EPSILON) * 100.0
    )
    summary_df["relative_improvement_vs_prev_k"] = pd.Series(np.nan, index=summary_df.index, dtype="float64")

    for idx in range(1, len(summary_df)):
        prev_best = float(summary_df.loc[idx - 1, "best_val_loss"])
        curr_best = float(summary_df.loc[idx, "best_val_loss"])
        improvement = (prev_best - curr_best) / max(prev_best, EPSILON) * 100.0
        summary_df.loc[idx, "relative_improvement_vs_prev_k"] = improvement

    summary_df["is_plateau_candidate"] = (
        summary_df["relative_improvement_vs_prev_k"].fillna(np.inf) < improvement_threshold_pct
    ) & (
        summary_df["convergence_ratio"] < convergence_ratio_threshold
    ) & (
        summary_df["performance_gap_to_best_pct"] <= performance_gap_threshold_pct
    )

    plateau_candidates = summary_df.loc[summary_df["is_plateau_candidate"]]
    if not plateau_candidates.empty:
        recommended_k = int(plateau_candidates.iloc[0]["knn_connectivity"])
    else:
        recommended_k = int(summary_df.loc[summary_df["best_val_loss"].idxmin(), "knn_connectivity"])

    return summary_df, recommended_k


def color_mapping(runs: Sequence[ExperimentRun]) -> dict[int, str]:
    return {
        run.knn_connectivity: PLOT_COLORS[idx % len(PLOT_COLORS)]
        for idx, run in enumerate(sorted(runs, key=lambda item: item.knn_connectivity))
    }


def plot_smoothed_and_raw(ax: plt.Axes, run: ExperimentRun, series: MetricSeries, color: str, raw_metric: str) -> None:
    raw_series = extract_metric_series(
        df=run.dataframe,
        metric=raw_metric,
        label=run.label,
        source=run.metrics_csv,
        x_axis=series.x_column if series.x_column != "index" else "auto",
        smooth_window=1,
    )
    if raw_series is not None:
        ax.plot(raw_series.x_values, raw_series.y_values, color=color, linewidth=0.9, alpha=0.2)
    ax.plot(series.x_values, series.y_values, color=color, linewidth=2.3, label=run.label)


def plot_main_knn_figure(
    runs: Sequence[ExperimentRun],
    summary_df: pd.DataFrame,
    recommended_k: int,
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    configure_plot_style()
    colors = color_mapping(runs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax_curve, ax_drop, ax_summary = axes

    for run in runs:
        plot_smoothed_and_raw(ax_curve, run, run.val_series, colors[run.knn_connectivity], run.val_series.metric)
    ax_curve.set_title("Validation Loss Curves")
    ax_curve.set_xlabel(axis_label_for([run.val_series for run in runs]))
    ax_curve.set_ylabel("Validation total loss")
    ax_curve.legend(frameon=False, ncol=2)

    x_positions = np.arange(len(summary_df))
    knn_labels = [f"K={int(value)}" for value in summary_df["knn_connectivity"]]
    bar_colors = [colors[int(value)] for value in summary_df["knn_connectivity"]]
    edge_colors = ["black" if int(value) == recommended_k else "none" for value in summary_df["knn_connectivity"]]
    edge_widths = [1.8 if int(value) == recommended_k else 0.0 for value in summary_df["knn_connectivity"]]

    ax_drop.bar(
        x_positions,
        summary_df["early_val_drop_rate"],
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=edge_widths,
        width=0.68,
    )
    ax_drop.set_xticks(x_positions, knn_labels)
    ax_drop.set_title("Early-stage Loss Drop Rate")
    ax_drop.set_xlabel("KNN neighbors (K)")
    ax_drop.set_ylabel("Average validation loss drop / progress")

    best_line = ax_summary.plot(
        x_positions,
        summary_df["best_val_loss"],
        color="#111111",
        linewidth=2.2,
        marker="o",
        label="Best validation loss",
    )[0]
    ax_summary.scatter(
        x_positions,
        summary_df["best_val_loss"],
        s=80,
        color=[colors[int(value)] for value in summary_df["knn_connectivity"]],
        zorder=3,
        edgecolors="white",
        linewidths=0.9,
    )
    ax_summary.set_xticks(x_positions, knn_labels)
    ax_summary.set_title("Validation Plateau Summary")
    ax_summary.set_xlabel("KNN neighbors (K)")
    ax_summary.set_ylabel("Best validation loss")

    ax_summary_secondary = ax_summary.twinx()
    improvement_values = summary_df["relative_improvement_vs_prev_k"].fillna(0.0)
    improvement_bars = ax_summary_secondary.bar(
        x_positions,
        improvement_values,
        alpha=0.2,
        color=[colors[int(value)] for value in summary_df["knn_connectivity"]],
        width=0.52,
        label="Improvement vs previous K (%)",
    )
    ax_summary_secondary.set_ylabel("Improvement vs previous K (%)")

    recommended_index = int(summary_df.index[summary_df["knn_connectivity"] == recommended_k][0])
    recommended_y = float(summary_df.loc[recommended_index, "best_val_loss"])
    ax_summary.annotate(
        f"Recommended K={recommended_k}",
        xy=(recommended_index, recommended_y),
        xytext=(recommended_index, recommended_y + 0.12),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        ha="center",
        fontsize=9,
    )

    ax_summary.legend(
        [best_line, improvement_bars],
        ["Best validation loss", "Improvement vs previous K (%)"],
        frameon=False,
        loc="upper right",
    )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_figure(
    runs: Sequence[ExperimentRun],
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    configure_plot_style()
    colors = color_mapping(runs)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)

    for run in runs:
        plot_smoothed_and_raw(axes[0], run, run.train_series, colors[run.knn_connectivity], run.train_series.metric)
        plot_smoothed_and_raw(axes[1], run, run.val_series, colors[run.knn_connectivity], run.val_series.metric)

    axes[0].set_title("Training Total Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, ncol=min(3, len(runs)))

    axes[1].set_title("Validation Total Loss")
    axes[1].set_xlabel(axis_label_for([run.val_series for run in runs]))
    axes[1].set_ylabel("Loss")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_optional(value: float | int | str | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def write_recommendation_report(
    summary_df: pd.DataFrame,
    recommended_k: int,
    output_path: Path,
    early_fraction: float,
    late_fraction: float,
    improvement_threshold_pct: float,
    convergence_ratio_threshold: float,
    performance_gap_threshold_pct: float,
) -> None:
    row = summary_df.loc[summary_df["knn_connectivity"] == recommended_k].iloc[0]
    lines = [
        f"Recommended K: {recommended_k}",
        "",
        "Selection rule:",
        f"- relative_improvement_vs_prev_k < {improvement_threshold_pct:.2f}%",
        f"- convergence_ratio < {convergence_ratio_threshold:.3f}",
        f"- performance_gap_to_best_pct <= {performance_gap_threshold_pct:.2f}%",
        "",
        "Window configuration:",
        f"- early_fraction = {early_fraction:.2f}",
        f"- late_fraction = {late_fraction:.2f}",
        "",
        "Recommended K metrics:",
        f"- best_val_loss = {format_optional(row['best_val_loss'])}",
        f"- final_val_loss = {format_optional(row['final_val_loss'])}",
        f"- early_val_drop_rate = {format_optional(row['early_val_drop_rate'])}",
        f"- late_val_slope = {format_optional(row['late_val_slope'])}",
        f"- convergence_ratio = {format_optional(row['convergence_ratio'])}",
        f"- relative_improvement_vs_prev_k = {format_optional(row['relative_improvement_vs_prev_k'])}",
        f"- performance_gap_to_best_pct = {format_optional(row['performance_gap_to_best_pct'])}",
        f"- is_plateau_candidate = {row['is_plateau_candidate']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and compare loss curves from Lightning metrics.csv files, or analyze KNN connectivity sweeps."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--csv",
        nargs="+",
        dest="csv_files",
        help="One or more metrics.csv files to compare.",
    )
    source_group.add_argument(
        "--knn-root",
        default=None,
        help="Root directory containing SDE-k=*/csv_logs/version_*/metrics.csv experiments.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=f"Metric columns to plot in CSV comparison mode. Defaults to {DEFAULT_TRAIN_METRIC} and {DEFAULT_VAL_METRIC}.",
    )
    parser.add_argument(
        "--train-metric",
        default=DEFAULT_TRAIN_METRIC,
        help=f"Training metric column for KNN analysis mode. Defaults to {DEFAULT_TRAIN_METRIC}.",
    )
    parser.add_argument(
        "--val-metric",
        default=DEFAULT_VAL_METRIC,
        help=f"Validation metric column for KNN analysis mode. Defaults to {DEFAULT_VAL_METRIC}.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels for each CSV file. Must match the number of CSV files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path for CSV comparison mode. Defaults to evaluation_results/loss_visiualization/loss_comparison.png.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated figures and reports. Defaults to evaluation_results/loss_visiualization.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--smooth-window",
        type=positive_int,
        default=9,
        help="Rolling-average window size for smoothed curves. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("auto", "epoch_exact", "epoch", "step"),
        default="auto",
        help="Preferred x-axis column. Auto tries epoch_exact, epoch, then step.",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=300,
        help="Saved image resolution in dots per inch.",
    )
    parser.add_argument(
        "--early-fraction",
        type=fraction_between_zero_and_one,
        default=0.2,
        help="Fraction of the training trajectory used to measure early-stage drop rate.",
    )
    parser.add_argument(
        "--late-fraction",
        type=fraction_between_zero_and_one,
        default=0.2,
        help="Fraction of the training trajectory used to measure late-stage convergence.",
    )
    parser.add_argument(
        "--improvement-threshold-pct",
        type=float,
        default=2.0,
        help="Relative improvement threshold used to identify the plateau K.",
    )
    parser.add_argument(
        "--convergence-ratio-threshold",
        type=float,
        default=0.15,
        help="Convergence ratio threshold used to identify the plateau K.",
    )
    parser.add_argument(
        "--performance-gap-threshold-pct",
        type=float,
        default=3.0,
        help="Allowed distance from the global-best validation loss when recommending a smaller plateau K.",
    )
    return parser.parse_args(argv)


def resolve_labels(csv_files: Sequence[Path], labels: Sequence[str] | None) -> list[str]:
    if labels is None:
        return [infer_label(csv_path) for csv_path in csv_files]
    if len(labels) != len(csv_files):
        raise ValueError("The number of labels must match the number of CSV files.")
    return list(labels)


def emit_warnings(messages: Iterable[str]) -> None:
    for message in messages:
        print(f"Warning: {message}", file=sys.stderr)


def run_csv_comparison_mode(args: argparse.Namespace) -> int:
    csv_files = [Path(path).expanduser().resolve() for path in args.csv_files]
    labels = resolve_labels(csv_files, args.labels)

    metric_groups, warnings = build_metric_groups(
        csv_files=csv_files,
        labels=labels,
        metrics=args.metrics,
        x_axis=args.x_axis,
        smooth_window=args.smooth_window,
    )
    emit_warnings(warnings)

    metric_groups = prune_empty_metric_groups(metric_groups)
    if not metric_groups:
        raise ValueError("None of the requested metrics were found in the provided CSV files.")

    output_path = Path(args.output).expanduser().resolve() if args.output else Path(args.output_dir).expanduser().resolve() / "loss_comparison.png"
    plot_metric_groups(
        metric_groups=metric_groups,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Saved comparison plot to {output_path}")
    return 0


def run_knn_analysis_mode(args: argparse.Namespace) -> int:
    knn_root = Path(args.knn_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runs, warnings = discover_knn_runs(
        knn_root=knn_root,
        train_metric=args.train_metric,
        val_metric=args.val_metric,
        x_axis=args.x_axis,
        smooth_window=args.smooth_window,
    )
    emit_warnings(warnings)

    summary_df, recommended_k = build_summary_dataframe(
        runs=runs,
        early_fraction=args.early_fraction,
        late_fraction=args.late_fraction,
        improvement_threshold_pct=args.improvement_threshold_pct,
        convergence_ratio_threshold=args.convergence_ratio_threshold,
        performance_gap_threshold_pct=args.performance_gap_threshold_pct,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    main_title = args.title or "KNN Connectivity Loss Analysis"
    aux_title = args.title or "Training and Validation Loss by KNN Connectivity"

    main_figure_path = output_dir / "knn_loss_main.png"
    aux_figure_path = output_dir / "knn_loss_train_val.png"
    summary_csv_path = output_dir / "knn_loss_summary.csv"
    recommendation_path = output_dir / "knn_recommendation.txt"

    plot_main_knn_figure(
        runs=runs,
        summary_df=summary_df,
        recommended_k=recommended_k,
        output_path=main_figure_path,
        dpi=args.dpi,
        title=main_title,
    )
    plot_train_val_figure(
        runs=runs,
        output_path=aux_figure_path,
        dpi=args.dpi,
        title=aux_title,
    )
    summary_df.to_csv(summary_csv_path, index=False)
    write_recommendation_report(
        summary_df=summary_df,
        recommended_k=recommended_k,
        output_path=recommendation_path,
        early_fraction=args.early_fraction,
        late_fraction=args.late_fraction,
        improvement_threshold_pct=args.improvement_threshold_pct,
        convergence_ratio_threshold=args.convergence_ratio_threshold,
        performance_gap_threshold_pct=args.performance_gap_threshold_pct,
    )

    print(f"Saved main figure to {main_figure_path}")
    print(f"Saved train/val figure to {aux_figure_path}")
    print(f"Saved summary CSV to {summary_csv_path}")
    print(f"Saved recommendation report to {recommendation_path}")
    print(f"Recommended K: {recommended_k}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.knn_root:
        return run_knn_analysis_mode(args)
    return run_csv_comparison_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
