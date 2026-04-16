#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PART_COLORS = {
    "attention": "#d95f02",
    "ffn": "#1b9e77",
    "others": "#7570b3",
    "transformer_others": "#7570b3",
    "embedding": "#e7298a",
    "final_norm": "#66a61e",
    "output_layer": "#e6ab02",
    "loss": "#a6761d",
    "non_transformer": "#666666",
    "non_transformer_others": "#666666",
}


def color_key_for_metric(metric_key: str) -> str:
    normalized = metric_key
    for suffix in (
        "_total_time_ms",
        "_backward_time_ms",
        "_time_ms",
        "_global_ratio",
        "_ratio",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    if normalized.startswith("attention"):
        return "attention"
    if normalized.startswith("ffn"):
        return "ffn"
    if normalized.startswith("embedding"):
        return "embedding"
    if normalized.startswith("final_norm"):
        return "final_norm"
    if normalized.startswith("output_layer"):
        return "output_layer"
    if normalized.startswith("loss"):
        return "loss"
    if normalized.startswith("non_transformer_others"):
        return "non_transformer_others"
    if normalized.startswith("transformer_others"):
        return "transformer_others"
    if normalized.startswith("others"):
        return "others"
    return normalized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/qwen8b_seq_study"),
        help="Root directory containing prof_seq*/torch_profile outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/qwen8b_seq_study/analysis"),
        help="Directory where CSV and PNG outputs will be written.",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to aggregate in order.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Profiler rank trace to analyze.",
    )
    parser.add_argument(
        "--label",
        default="Qwen3-8B",
        help="Short experiment label to include in chart titles.",
    )
    return parser.parse_args()


def load_trace(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    return trace.get("traceEvents", [])


def parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def average_ms(durations_us: dict[str, float], name: str, profiled_steps: int) -> float:
    return durations_us.get(name, 0.0) / 1000.0 / profiled_steps


def average_ms_prefer_gpu(
    gpu_durations_us: dict[str, float],
    cpu_durations_us: dict[str, float],
    name: str,
    profiled_steps: int,
) -> float:
    if name in gpu_durations_us:
        return gpu_durations_us[name] / 1000.0 / profiled_steps
    return cpu_durations_us.get(name, 0.0) / 1000.0 / profiled_steps


def build_forward_breakdown(
    gpu_durations_us: dict[str, float],
    cpu_durations_us: dict[str, float],
    profiled_steps: int,
) -> dict[str, float]:
    input_layernorm_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/input_layernorm", profiled_steps
    )
    self_attention_core_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/self_attention_core", profiled_steps
    )
    self_attn_bda_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/self_attn_bda", profiled_steps
    )
    pre_mlp_layernorm_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/pre_mlp_layernorm", profiled_steps
    )
    mlp_core_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/mlp_core", profiled_steps
    )
    mlp_bda_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/mlp_bda", profiled_steps
    )

    attention_time_ms = self_attention_core_time_ms
    ffn_time_ms = mlp_core_time_ms
    others_time_ms = (
        input_layernorm_time_ms
        + self_attn_bda_time_ms
        + pre_mlp_layernorm_time_ms
        + mlp_bda_time_ms
    )
    block_time_ms = attention_time_ms + ffn_time_ms + others_time_ms

    return {
        "transformer_block_forward_boundary_time_ms": average_ms(
            cpu_durations_us, "layer/transformer_block", profiled_steps
        ),
        "input_layernorm_time_ms": input_layernorm_time_ms,
        "self_attention_core_time_ms": self_attention_core_time_ms,
        "self_attn_bda_time_ms": self_attn_bda_time_ms,
        "pre_mlp_layernorm_time_ms": pre_mlp_layernorm_time_ms,
        "mlp_core_time_ms": mlp_core_time_ms,
        "mlp_bda_time_ms": mlp_bda_time_ms,
        "transformer_block_time_ms": block_time_ms,
        "attention_time_ms": attention_time_ms,
        "ffn_time_ms": ffn_time_ms,
        "others_time_ms": others_time_ms,
    }


def build_backward_breakdown(
    gpu_durations_us: dict[str, float],
    cpu_durations_us: dict[str, float],
    profiled_steps: int,
) -> dict[str, float]:
    attention_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/self_attention/backward", profiled_steps
    )
    ffn_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/mlp/backward", profiled_steps
    )
    input_layernorm_backward_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/input_layernorm/backward", profiled_steps
    )
    self_attn_bda_backward_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/self_attn_bda/backward", profiled_steps
    )
    pre_mlp_layernorm_backward_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/pre_mlp_layernorm/backward", profiled_steps
    )
    mlp_bda_backward_time_ms = average_ms_prefer_gpu(
        gpu_durations_us, cpu_durations_us, "layer/mlp_bda/backward", profiled_steps
    )
    others_time_ms = (
        input_layernorm_backward_time_ms
        + self_attn_bda_backward_time_ms
        + pre_mlp_layernorm_backward_time_ms
        + mlp_bda_backward_time_ms
    )
    block_time_ms = attention_time_ms + ffn_time_ms + others_time_ms

    return {
        "transformer_block_backward_boundary_time_ms": average_ms(
            cpu_durations_us, "layer/transformer_block/backward", profiled_steps
        ),
        "input_layernorm_backward_time_ms": input_layernorm_backward_time_ms,
        "self_attn_bda_backward_time_ms": self_attn_bda_backward_time_ms,
        "pre_mlp_layernorm_backward_time_ms": pre_mlp_layernorm_backward_time_ms,
        "mlp_bda_backward_time_ms": mlp_bda_backward_time_ms,
        "transformer_block_backward_time_ms": block_time_ms,
        "attention_backward_time_ms": attention_time_ms,
        "ffn_backward_time_ms": ffn_time_ms,
        "others_backward_time_ms": others_time_ms,
    }


def summarize_trace(events: list[dict]) -> dict[str, float]:
    cpu_durations_us: dict[str, float] = {}
    gpu_durations_us: dict[str, float] = {}
    profiled_steps = 0
    profiler_step_total_us = 0.0

    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name")
        duration = float(event.get("dur", 0.0))
        if not isinstance(name, str):
            continue
        category = event.get("cat")
        if category == "user_annotation":
            cpu_durations_us[name] = cpu_durations_us.get(name, 0.0) + duration
        elif category == "gpu_user_annotation":
            gpu_durations_us[name] = gpu_durations_us.get(name, 0.0) + duration
        else:
            continue
        if category == "user_annotation" and name.startswith("ProfilerStep#"):
            profiled_steps += 1
            profiler_step_total_us += duration

    if profiled_steps == 0:
        raise ValueError("No ProfilerStep events found in trace.")

    summary = {
        "profiled_steps": float(profiled_steps),
        "profiler_step_time_ms": profiler_step_total_us / 1000.0 / profiled_steps,
        "train_forward_backward_time_ms": average_ms(
            cpu_durations_us, "train/forward_backward", profiled_steps
        ),
        "train_optimizer_step_time_ms": average_ms(
            cpu_durations_us, "train/optimizer_step", profiled_steps
        ),
        "embedding_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/embedding", profiled_steps
        ),
        "embedding_backward_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/embedding/backward", profiled_steps
        ),
        "final_norm_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/final_norm", profiled_steps
        ),
        "final_norm_backward_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/final_norm/backward", profiled_steps
        ),
        "output_layer_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/output_layer", profiled_steps
        ),
        "output_layer_backward_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/output_layer/backward", profiled_steps
        ),
        "loss_time_ms": average_ms_prefer_gpu(
            gpu_durations_us, cpu_durations_us, "gpt/loss", profiled_steps
        ),
    }

    summary.update(build_forward_breakdown(gpu_durations_us, cpu_durations_us, profiled_steps))
    summary.update(build_backward_breakdown(gpu_durations_us, cpu_durations_us, profiled_steps))

    summary["transformer_block_total_time_ms"] = (
        summary["transformer_block_time_ms"] + summary["transformer_block_backward_time_ms"]
    )
    summary["attention_total_time_ms"] = (
        summary["attention_time_ms"] + summary["attention_backward_time_ms"]
    )
    summary["ffn_total_time_ms"] = summary["ffn_time_ms"] + summary["ffn_backward_time_ms"]
    summary["others_total_time_ms"] = max(
        summary["transformer_block_total_time_ms"]
        - summary["attention_total_time_ms"]
        - summary["ffn_total_time_ms"],
        0.0,
    )
    summary["embedding_total_time_ms"] = (
        summary["embedding_time_ms"] + summary["embedding_backward_time_ms"]
    )
    summary["final_norm_total_time_ms"] = (
        summary["final_norm_time_ms"] + summary["final_norm_backward_time_ms"]
    )
    summary["output_layer_total_time_ms"] = (
        summary["output_layer_time_ms"] + summary["output_layer_backward_time_ms"]
    )
    summary["loss_total_time_ms"] = summary["loss_time_ms"]
    summary["non_transformer_total_time_ms"] = max(
        summary["train_forward_backward_time_ms"] - summary["transformer_block_total_time_ms"],
        0.0,
    )
    labeled_global_total_time_ms = (
        summary["attention_total_time_ms"]
        + summary["ffn_total_time_ms"]
        + summary["others_total_time_ms"]
        + summary["embedding_total_time_ms"]
        + summary["final_norm_total_time_ms"]
        + summary["output_layer_total_time_ms"]
        + summary["loss_total_time_ms"]
    )
    summary["non_transformer_others_total_time_ms"] = max(
        summary["train_forward_backward_time_ms"] - labeled_global_total_time_ms,
        0.0,
    )
    summary["transformer_block_share_of_train_forward_backward"] = (
        summary["transformer_block_total_time_ms"] / summary["train_forward_backward_time_ms"]
        if summary["train_forward_backward_time_ms"] > 0
        else 0.0
    )
    summary["non_transformer_share_of_train_forward_backward"] = (
        summary["non_transformer_total_time_ms"] / summary["train_forward_backward_time_ms"]
        if summary["train_forward_backward_time_ms"] > 0
        else 0.0
    )
    return summary


def add_ratio_fields(
    row: dict[str, float | str],
    *,
    total_key: str,
    prefix: str,
    attention_key: str,
    ffn_key: str,
    others_key: str,
) -> None:
    block_total = float(row[total_key])
    attention_ratio_key = f"attention_{prefix}ratio"
    ffn_ratio_key = f"ffn_{prefix}ratio"
    others_ratio_key = f"others_{prefix}ratio"

    if block_total <= 0:
        row[attention_ratio_key] = 0.0
        row[ffn_ratio_key] = 0.0
        row[others_ratio_key] = 0.0
        return

    row[attention_ratio_key] = float(row[attention_key]) / block_total
    row[ffn_ratio_key] = float(row[ffn_key]) / block_total
    row[others_ratio_key] = float(row[others_key]) / block_total


def add_multi_ratio_fields(
    row: dict[str, float | str],
    *,
    total_key: str,
    ratio_key_pairs: list[tuple[str, str]],
) -> None:
    total = float(row[total_key])
    for value_key, ratio_key in ratio_key_pairs:
        row[ratio_key] = 0.0 if total <= 0 else float(row[value_key]) / total


def enrich_row(row: dict[str, float | str]) -> dict[str, float | str]:
    add_ratio_fields(
        row,
        total_key="transformer_block_time_ms",
        prefix="",
        attention_key="attention_time_ms",
        ffn_key="ffn_time_ms",
        others_key="others_time_ms",
    )
    add_ratio_fields(
        row,
        total_key="transformer_block_backward_time_ms",
        prefix="backward_",
        attention_key="attention_backward_time_ms",
        ffn_key="ffn_backward_time_ms",
        others_key="others_backward_time_ms",
    )
    add_ratio_fields(
        row,
        total_key="transformer_block_total_time_ms",
        prefix="total_",
        attention_key="attention_total_time_ms",
        ffn_key="ffn_total_time_ms",
        others_key="others_total_time_ms",
    )
    add_multi_ratio_fields(
        row,
        total_key="train_forward_backward_time_ms",
        ratio_key_pairs=[
            ("attention_total_time_ms", "attention_global_ratio"),
            ("ffn_total_time_ms", "ffn_global_ratio"),
            ("others_total_time_ms", "transformer_others_global_ratio"),
            ("non_transformer_total_time_ms", "non_transformer_global_ratio"),
        ],
    )
    return row


def draw_axes(draw: ImageDraw.ImageDraw, origin, width, height, *, x_ticks, y_ticks, labels):
    x0, y0 = origin
    draw.line((x0, y0, x0, y0 - height), fill="black", width=2)
    draw.line((x0, y0, x0 + width, y0), fill="black", width=2)
    for x, label in x_ticks:
        draw.line((x, y0, x, y0 + 6), fill="black", width=1)
        draw.text((x - 20, y0 + 10), label, fill="black", font=labels)
    for y, label in y_ticks:
        draw.line((x0 - 6, y, x0, y), fill="black", width=1)
        draw.text((x0 - 70, y - 8), label, fill="black", font=labels)


def save_stacked_bar_chart(
    rows,
    output_path: Path,
    *,
    title: str,
    total_key: str,
    part_keys: tuple[str, ...],
    legend_labels: tuple[str, ...] | None = None,
):
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((320, 20), title, fill="black", font=font)

    origin = (120, 620)
    width = 980
    height = 500
    max_total = max(float(row[total_key]) for row in rows) or 1.0
    max_total *= 1.1

    x_step = width / max(len(rows), 1)
    x_ticks = []
    y_ticks = []
    for i in range(6):
        value = max_total * i / 5
        y = origin[1] - (value / max_total) * height
        y_ticks.append((y, f"{value:.0f}"))

    for idx, row in enumerate(rows):
        x_center = origin[0] + x_step * idx + x_step / 2
        x_ticks.append((x_center, str(row["seq_length"])))
        bar_left = x_center - x_step * 0.22
        bar_right = x_center + x_step * 0.22
        y_cursor = origin[1]
        for key in part_keys:
            value = float(row[key])
            bar_height = (value / max_total) * height
            color_key = color_key_for_metric(key)
            draw.rectangle(
                (bar_left, y_cursor - bar_height, bar_right, y_cursor),
                fill=PART_COLORS[color_key],
                outline="black",
            )
            y_cursor -= bar_height
        draw.text(
            (bar_left - 8, y_cursor - 20),
            f"{float(row[total_key]):.1f}",
            fill="black",
            font=font,
        )

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, labels=font)
    draw.text((20, 20), "Y axis: avg ms per profiled step", fill="black", font=font)
    legend_items = legend_labels or part_keys
    for idx, (key, label) in enumerate(zip(part_keys, legend_items)):
        color_key = color_key_for_metric(key)
        draw.text((840, 80 + idx * 20), label, fill=PART_COLORS[color_key], font=font)
    image.save(output_path)


def save_ratio_line_chart(
    rows,
    output_path: Path,
    *,
    title: str,
    ratio_keys: tuple[str, ...],
    legend_labels: tuple[str, ...] | None = None,
):
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((340, 20), title, fill="black", font=font)

    origin = (120, 620)
    width = 980
    height = 500
    x_step = width / max(len(rows) - 1, 1)
    x_ticks = []
    y_ticks = []
    for i in range(6):
        value = i / 5
        y = origin[1] - value * height
        y_ticks.append((y, f"{value:.1f}"))

    draw_axes(draw, origin, width, height, x_ticks=[], y_ticks=y_ticks, labels=font)

    series = {}
    for key in ratio_keys:
        color_key = color_key_for_metric(key)
        series[key] = PART_COLORS[color_key]

    for idx, row in enumerate(rows):
        x = origin[0] + x_step * idx
        x_ticks.append((x, str(row["seq_length"])))

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, labels=font)

    for key, color in series.items():
        points = []
        for idx, row in enumerate(rows):
            x = origin[0] + x_step * idx
            y = origin[1] - float(row[key]) * height
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)

    legend_items = legend_labels or ratio_keys
    for idx, (key, label) in enumerate(zip(ratio_keys, legend_items)):
        color_key = color_key_for_metric(key)
        draw.text((840, 80 + idx * 20), label, fill=PART_COLORS[color_key], font=font)
    image.save(output_path)


def save_step_time_chart(rows, output_path: Path, *, title: str):
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((320, 20), title, fill="black", font=font)

    origin = (120, 620)
    width = 980
    height = 500
    max_time = max(float(row["profiler_step_time_ms"]) for row in rows) or 1.0
    max_time *= 1.1
    x_step = width / max(len(rows) - 1, 1)
    x_ticks = []
    y_ticks = []
    for i in range(6):
        value = max_time * i / 5
        y = origin[1] - (value / max_time) * height
        y_ticks.append((y, f"{value:.0f}"))

    draw_axes(draw, origin, width, height, x_ticks=[], y_ticks=y_ticks, labels=font)

    points = []
    for idx, row in enumerate(rows):
        x = origin[0] + x_step * idx
        y = origin[1] - (float(row["profiler_step_time_ms"]) / max_time) * height
        x_ticks.append((x, str(row["seq_length"])))
        points.append((x, y))

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, labels=font)
    if len(points) >= 2:
        draw.line(points, fill="#4c78a8", width=3)
    for point, row in zip(points, rows):
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#4c78a8")
        draw.text(
            (point[0] - 12, point[1] - 20),
            f"{float(row['profiler_step_time_ms']):.0f}",
            fill="black",
            font=font,
        )

    image.save(output_path)


def empty_summary() -> dict[str, float]:
    return {
        "profiled_steps": 0.0,
        "profiler_step_time_ms": 0.0,
        "train_forward_backward_time_ms": 0.0,
        "train_optimizer_step_time_ms": 0.0,
        "transformer_block_forward_boundary_time_ms": 0.0,
        "input_layernorm_time_ms": 0.0,
        "self_attention_core_time_ms": 0.0,
        "self_attn_bda_time_ms": 0.0,
        "pre_mlp_layernorm_time_ms": 0.0,
        "mlp_core_time_ms": 0.0,
        "mlp_bda_time_ms": 0.0,
        "embedding_time_ms": 0.0,
        "embedding_backward_time_ms": 0.0,
        "embedding_total_time_ms": 0.0,
        "final_norm_time_ms": 0.0,
        "final_norm_backward_time_ms": 0.0,
        "final_norm_total_time_ms": 0.0,
        "output_layer_time_ms": 0.0,
        "output_layer_backward_time_ms": 0.0,
        "output_layer_total_time_ms": 0.0,
        "loss_time_ms": 0.0,
        "loss_total_time_ms": 0.0,
        "transformer_block_time_ms": 0.0,
        "attention_time_ms": 0.0,
        "ffn_time_ms": 0.0,
        "others_time_ms": 0.0,
        "transformer_block_backward_time_ms": 0.0,
        "attention_backward_time_ms": 0.0,
        "ffn_backward_time_ms": 0.0,
        "others_backward_time_ms": 0.0,
        "transformer_block_total_time_ms": 0.0,
        "attention_total_time_ms": 0.0,
        "ffn_total_time_ms": 0.0,
        "others_total_time_ms": 0.0,
        "non_transformer_total_time_ms": 0.0,
        "non_transformer_others_total_time_ms": 0.0,
        "transformer_block_share_of_train_forward_backward": 0.0,
        "non_transformer_share_of_train_forward_backward": 0.0,
    }


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    legacy_outputs = [
        "qwen3_8b_seq_stacked.png",
        "qwen3_8b_seq_ratios.png",
        "qwen3_8b_seq_global_total_stacked.png",
        "qwen3_8b_seq_global_total_ratios.png",
    ]
    for name in legacy_outputs:
        legacy_path = args.output_root / name
        if legacy_path.exists():
            legacy_path.unlink()

    rows = []
    for seq in args.seq_lengths:
        run_dir = args.input_root / f"prof_seq{seq}"
        metadata = parse_env_file(run_dir / "run_metadata.env")
        status = metadata.get("status", "missing")

        row: dict[str, float | str] = {
            "seq_length": seq,
            "status": status,
            "selected_gpus": metadata.get("selected_gpus", ""),
        }

        trace_path = run_dir / "torch_profile" / f"rank-{args.rank}.json.gz"
        if status != "success" or not trace_path.exists():
            row.update(empty_summary())
            rows.append(enrich_row(row))
            continue

        summary = summarize_trace(load_trace(trace_path))
        row.update(summary)
        rows.append(enrich_row(row))

    csv_path = args.output_root / "qwen3_8b_seq_profile_summary.csv"
    fieldnames = [
        "seq_length",
        "status",
        "selected_gpus",
        "profiled_steps",
        "profiler_step_time_ms",
        "train_forward_backward_time_ms",
        "train_optimizer_step_time_ms",
        "transformer_block_forward_boundary_time_ms",
        "input_layernorm_time_ms",
        "self_attention_core_time_ms",
        "self_attn_bda_time_ms",
        "pre_mlp_layernorm_time_ms",
        "mlp_core_time_ms",
        "mlp_bda_time_ms",
        "embedding_time_ms",
        "embedding_backward_time_ms",
        "embedding_total_time_ms",
        "final_norm_time_ms",
        "final_norm_backward_time_ms",
        "final_norm_total_time_ms",
        "output_layer_time_ms",
        "output_layer_backward_time_ms",
        "output_layer_total_time_ms",
        "loss_time_ms",
        "loss_total_time_ms",
        "transformer_block_time_ms",
        "attention_time_ms",
        "ffn_time_ms",
        "others_time_ms",
        "transformer_block_backward_time_ms",
        "transformer_block_backward_boundary_time_ms",
        "attention_backward_time_ms",
        "ffn_backward_time_ms",
        "others_backward_time_ms",
        "input_layernorm_backward_time_ms",
        "self_attn_bda_backward_time_ms",
        "pre_mlp_layernorm_backward_time_ms",
        "mlp_bda_backward_time_ms",
        "transformer_block_total_time_ms",
        "attention_total_time_ms",
        "ffn_total_time_ms",
        "others_total_time_ms",
        "non_transformer_total_time_ms",
        "non_transformer_others_total_time_ms",
        "attention_ratio",
        "ffn_ratio",
        "others_ratio",
        "attention_backward_ratio",
        "ffn_backward_ratio",
        "others_backward_ratio",
        "attention_total_ratio",
        "ffn_total_ratio",
        "others_total_ratio",
        "attention_global_ratio",
        "ffn_global_ratio",
        "transformer_others_global_ratio",
        "non_transformer_global_ratio",
        "transformer_block_share_of_train_forward_backward",
        "non_transformer_share_of_train_forward_backward",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [row for row in rows if row["status"] == "success"]
    if successful_rows:
        save_stacked_bar_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_forward_stacked.png",
            title=f"{args.label} Transformer Block Forward Time Breakdown",
            total_key="transformer_block_time_ms",
            part_keys=("attention_time_ms", "ffn_time_ms", "others_time_ms"),
            legend_labels=("Attention", "FFN", "Transformer Others"),
        )
        save_ratio_line_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_forward_ratios.png",
            title=f"{args.label} Transformer Block Forward Ratios",
            ratio_keys=("attention_ratio", "ffn_ratio", "others_ratio"),
            legend_labels=("Attention", "FFN", "Transformer Others"),
        )
        save_stacked_bar_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_backward_stacked.png",
            title=f"{args.label} Transformer Block Backward Time Breakdown",
            total_key="transformer_block_backward_time_ms",
            part_keys=(
                "attention_backward_time_ms",
                "ffn_backward_time_ms",
                "others_backward_time_ms",
            ),
            legend_labels=("Attention", "FFN", "Transformer Others"),
        )
        save_ratio_line_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_backward_ratios.png",
            title=f"{args.label} Transformer Block Backward Ratios",
            ratio_keys=(
                "attention_backward_ratio",
                "ffn_backward_ratio",
                "others_backward_ratio",
            ),
            legend_labels=("Attention", "FFN", "Transformer Others"),
        )
        save_stacked_bar_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_total_stacked.png",
            title=f"{args.label} Transformer Block Forward+Backward Breakdown",
            total_key="transformer_block_total_time_ms",
            part_keys=("attention_total_time_ms", "ffn_total_time_ms", "others_total_time_ms"),
            legend_labels=("Attention", "FFN", "Transformer Others"),
        )
        save_ratio_line_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_total_ratios.png",
            title=f"{args.label} Transformer Block Forward+Backward Ratios",
            ratio_keys=("attention_total_ratio", "ffn_total_ratio", "others_total_ratio"),
            legend_labels=("Attention", "FFN", "Transformer Others"),
        )
        save_stacked_bar_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_global_stacked.png",
            title=f"{args.label} Transformer + Non-Transformer Global Breakdown",
            total_key="train_forward_backward_time_ms",
            part_keys=(
                "attention_total_time_ms",
                "ffn_total_time_ms",
                "others_total_time_ms",
                "non_transformer_total_time_ms",
            ),
            legend_labels=(
                "Attention",
                "FFN",
                "Transformer Others",
                "Non-Transformer",
            ),
        )
        save_ratio_line_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_global_ratios.png",
            title=f"{args.label} Transformer + Non-Transformer Global Ratios",
            ratio_keys=(
                "attention_global_ratio",
                "ffn_global_ratio",
                "transformer_others_global_ratio",
                "non_transformer_global_ratio",
            ),
            legend_labels=(
                "Attention",
                "FFN",
                "Transformer Others",
                "Non-Transformer",
            ),
        )
        save_step_time_chart(
            successful_rows,
            args.output_root / "qwen3_8b_seq_step_time.png",
            title=f"{args.label} Avg Profiled Step Time",
        )


if __name__ == "__main__":
    main()
