#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PART_COLORS = {
    "attention": "#d95f02",
    "ffn": "#1b9e77",
    "transformer_others": "#7570b3",
    "non_transformer": "#666666",
    "unattributed": "#e7298a",
}


def color_key_for_metric(metric_key: str) -> str:
    normalized = metric_key
    for suffix in (
        "_total_time_ms",
        "_backward_time_ms",
        "_time_ms",
        "_global_ratio",
        "_total_ratio",
        "_backward_ratio",
        "_ratio",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    if normalized.startswith("attention"):
        return "attention"
    if normalized.startswith("ffn"):
        return "ffn"
    if normalized.startswith("transformer_others"):
        return "transformer_others"
    if normalized.startswith("non_transformer"):
        return "non_transformer"
    if normalized.startswith("unattributed"):
        return "unattributed"
    return normalized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/qwen0p6b_seq_study_pp4_nsys_0p6b_base"),
        help="Root directory containing prof_seq*/seq*.nsys-rep outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/qwen0p6b_seq_study_pp4_nsys_0p6b_base/analysis"),
        help="Directory where CSV and PNG outputs will be written.",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        help="Sequence lengths to aggregate in order.",
    )
    parser.add_argument(
        "--label",
        default="Qwen3-0.6B PP4",
        help="Short experiment label to include in chart titles.",
    )
    return parser.parse_args()


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


def normalize_range_name(raw: str) -> str:
    raw = raw.removeprefix(":")
    marker = ", op_id = "
    if marker in raw:
        raw = raw.split(marker, 1)[0]
    return raw


def load_nsys_nvtx_projection(report_path: Path) -> dict[str, dict[str, float]]:
    cmd = [
        "nsys",
        "stats",
        "--report",
        "nvtx_gpu_proj_sum",
        "--format",
        "csv",
        "--output",
        "-",
        str(report_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    header_idx = next((idx for idx, line in enumerate(lines) if line.startswith("Range,")), None)
    if header_idx is None:
        raise ValueError(f"Unable to locate nvtx_gpu_proj_sum CSV header in {report_path}")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    totals: dict[str, dict[str, float]] = {}
    for row in reader:
        name = normalize_range_name(row["Range"])
        stats = totals.setdefault(name, {"total_proj_ns": 0.0, "range_instances": 0.0})
        stats["total_proj_ns"] += float(row["Total Proj Time (ns)"])
        stats["range_instances"] += float(row["Range Instances"])
    return totals


def per_step_ms(range_totals: dict[str, dict[str, float]], name: str, step_instances: float) -> float:
    if step_instances <= 0:
        return 0.0
    return range_totals.get(name, {}).get("total_proj_ns", 0.0) / step_instances / 1_000_000.0


def add_ratio_fields(row: dict[str, float | str], total_key: str, pairs: list[tuple[str, str]]) -> None:
    total = float(row[total_key])
    for value_key, ratio_key in pairs:
        row[ratio_key] = 0.0 if total <= 0 else float(row[value_key]) / total


def summarize_report(report_path: Path) -> dict[str, float]:
    range_totals = load_nsys_nvtx_projection(report_path)
    step_instances = range_totals.get("train/forward_backward", {}).get("range_instances", 0.0)
    if step_instances <= 0:
        raise ValueError(f"No train/forward_backward NVTX instances found in {report_path}")

    train_forward_backward_time_ms = per_step_ms(range_totals, "train/forward_backward", step_instances)
    train_optimizer_step_time_ms = per_step_ms(range_totals, "train/optimizer_step", step_instances)

    transformer_block_time_ms = per_step_ms(range_totals, "layer/transformer_block", step_instances)
    attention_time_ms = per_step_ms(range_totals, "layer/self_attention_core", step_instances)
    ffn_time_ms = per_step_ms(range_totals, "layer/mlp_core", step_instances)
    transformer_others_time_ms = max(
        transformer_block_time_ms - attention_time_ms - ffn_time_ms,
        0.0,
    )

    embedding_time_ms = per_step_ms(range_totals, "gpt/embedding", step_instances)
    final_norm_time_ms = per_step_ms(range_totals, "gpt/final_norm", step_instances)
    output_layer_time_ms = per_step_ms(range_totals, "gpt/output_layer", step_instances)
    loss_time_ms = per_step_ms(range_totals, "gpt/loss", step_instances)
    non_transformer_forward_time_ms = (
        embedding_time_ms + final_norm_time_ms + output_layer_time_ms + loss_time_ms
    )

    total_backward_time_ms = max(
        train_forward_backward_time_ms - transformer_block_time_ms - non_transformer_forward_time_ms,
        0.0,
    )
    attention_backward_time_ms = per_step_ms(range_totals, "layer/self_attention/backward", step_instances)
    ffn_backward_time_ms = per_step_ms(range_totals, "layer/mlp/backward", step_instances)
    embedding_backward_time_ms = per_step_ms(range_totals, "gpt/embedding/backward", step_instances)
    final_norm_backward_time_ms = per_step_ms(range_totals, "gpt/final_norm/backward", step_instances)
    output_layer_backward_time_ms = per_step_ms(range_totals, "gpt/output_layer/backward", step_instances)
    non_transformer_backward_time_ms = (
        embedding_backward_time_ms + final_norm_backward_time_ms + output_layer_backward_time_ms
    )
    transformer_others_backward_time_ms = max(
        total_backward_time_ms
        - attention_backward_time_ms
        - ffn_backward_time_ms
        - non_transformer_backward_time_ms,
        0.0,
    )

    attention_total_time_ms = attention_time_ms + attention_backward_time_ms
    ffn_total_time_ms = ffn_time_ms + ffn_backward_time_ms
    transformer_others_total_time_ms = (
        transformer_others_time_ms + transformer_others_backward_time_ms
    )
    non_transformer_total_time_ms = (
        non_transformer_forward_time_ms + non_transformer_backward_time_ms
    )
    unattributed_total_time_ms = max(
        train_forward_backward_time_ms
        - attention_total_time_ms
        - ffn_total_time_ms
        - transformer_others_total_time_ms
        - non_transformer_total_time_ms,
        0.0,
    )

    summary = {
        "step_instances": step_instances,
        "train_forward_backward_time_ms": train_forward_backward_time_ms,
        "train_optimizer_step_time_ms": train_optimizer_step_time_ms,
        "step_gpu_time_ms": train_forward_backward_time_ms + train_optimizer_step_time_ms,
        "transformer_block_time_ms": transformer_block_time_ms,
        "attention_time_ms": attention_time_ms,
        "ffn_time_ms": ffn_time_ms,
        "transformer_others_time_ms": transformer_others_time_ms,
        "embedding_time_ms": embedding_time_ms,
        "final_norm_time_ms": final_norm_time_ms,
        "output_layer_time_ms": output_layer_time_ms,
        "loss_time_ms": loss_time_ms,
        "non_transformer_forward_time_ms": non_transformer_forward_time_ms,
        "total_backward_time_ms": total_backward_time_ms,
        "attention_backward_time_ms": attention_backward_time_ms,
        "ffn_backward_time_ms": ffn_backward_time_ms,
        "transformer_others_backward_time_ms": transformer_others_backward_time_ms,
        "embedding_backward_time_ms": embedding_backward_time_ms,
        "final_norm_backward_time_ms": final_norm_backward_time_ms,
        "output_layer_backward_time_ms": output_layer_backward_time_ms,
        "non_transformer_backward_time_ms": non_transformer_backward_time_ms,
        "transformer_block_backward_time_ms": (
            attention_backward_time_ms
            + ffn_backward_time_ms
            + transformer_others_backward_time_ms
        ),
        "attention_total_time_ms": attention_total_time_ms,
        "ffn_total_time_ms": ffn_total_time_ms,
        "transformer_others_total_time_ms": transformer_others_total_time_ms,
        "non_transformer_total_time_ms": non_transformer_total_time_ms,
        "unattributed_total_time_ms": unattributed_total_time_ms,
        "transformer_block_total_time_ms": (
            attention_total_time_ms + ffn_total_time_ms + transformer_others_total_time_ms
        ),
    }

    add_ratio_fields(
        summary,
        "transformer_block_time_ms",
        [
            ("attention_time_ms", "attention_ratio"),
            ("ffn_time_ms", "ffn_ratio"),
            ("transformer_others_time_ms", "transformer_others_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_backward_time_ms",
        [
            ("attention_backward_time_ms", "attention_backward_ratio"),
            ("ffn_backward_time_ms", "ffn_backward_ratio"),
            ("transformer_others_backward_time_ms", "transformer_others_backward_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_total_time_ms",
        [
            ("attention_total_time_ms", "attention_total_ratio"),
            ("ffn_total_time_ms", "ffn_total_ratio"),
            ("transformer_others_total_time_ms", "transformer_others_total_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "train_forward_backward_time_ms",
        [
            ("attention_total_time_ms", "attention_global_ratio"),
            ("ffn_total_time_ms", "ffn_global_ratio"),
            ("transformer_others_total_time_ms", "transformer_others_global_ratio"),
            ("non_transformer_total_time_ms", "non_transformer_global_ratio"),
            ("unattributed_total_time_ms", "unattributed_global_ratio"),
        ],
    )
    return summary


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


def save_stacked_bar_chart(rows, output_path: Path, *, title: str, total_key: str, part_keys: tuple[str, ...], legend_labels: tuple[str, ...]):
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((260, 20), title, fill="black", font=font)

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
            color = PART_COLORS[color_key_for_metric(key)]
            draw.rectangle(
                (bar_left, y_cursor - bar_height, bar_right, y_cursor),
                fill=color,
                outline="black",
            )
            y_cursor -= bar_height
        draw.text((bar_left - 8, y_cursor - 20), f"{float(row[total_key]):.1f}", fill="black", font=font)

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, labels=font)
    draw.text((20, 20), "Y axis: avg projected GPU ms per profiled step per rank", fill="black", font=font)
    for idx, (key, label) in enumerate(zip(part_keys, legend_labels)):
        color = PART_COLORS[color_key_for_metric(key)]
        draw.text((820, 80 + idx * 20), label, fill=color, font=font)
    image.save(output_path)


def save_ratio_line_chart(rows, output_path: Path, *, title: str, ratio_keys: tuple[str, ...], legend_labels: tuple[str, ...]):
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((320, 20), title, fill="black", font=font)

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

    for idx, row in enumerate(rows):
        x = origin[0] + x_step * idx
        x_ticks.append((x, str(row["seq_length"])))
    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, labels=font)

    for key in ratio_keys:
        color = PART_COLORS[color_key_for_metric(key)]
        points = []
        for idx, row in enumerate(rows):
            x = origin[0] + x_step * idx
            y = origin[1] - float(row[key]) * height
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)

    for idx, (key, label) in enumerate(zip(ratio_keys, legend_labels)):
        draw.text((820, 80 + idx * 20), label, fill=PART_COLORS[color_key_for_metric(key)], font=font)
    image.save(output_path)


def save_step_time_chart(rows, output_path: Path, *, title: str):
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((320, 20), title, fill="black", font=font)

    origin = (120, 620)
    width = 980
    height = 500
    max_time = max(float(row["step_gpu_time_ms"]) for row in rows) or 1.0
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
        y = origin[1] - (float(row["step_gpu_time_ms"]) / max_time) * height
        x_ticks.append((x, str(row["seq_length"])))
        points.append((x, y))

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, labels=font)
    if len(points) >= 2:
        draw.line(points, fill="#4c78a8", width=3)
    for point, row in zip(points, rows):
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#4c78a8")
        draw.text((point[0] - 12, point[1] - 20), f"{float(row['step_gpu_time_ms']):.0f}", fill="black", font=font)

    image.save(output_path)


def empty_summary() -> dict[str, float]:
    return {
        "step_instances": 0.0,
        "train_forward_backward_time_ms": 0.0,
        "train_optimizer_step_time_ms": 0.0,
        "step_gpu_time_ms": 0.0,
        "transformer_block_time_ms": 0.0,
        "attention_time_ms": 0.0,
        "ffn_time_ms": 0.0,
        "transformer_others_time_ms": 0.0,
        "embedding_time_ms": 0.0,
        "final_norm_time_ms": 0.0,
        "output_layer_time_ms": 0.0,
        "loss_time_ms": 0.0,
        "non_transformer_forward_time_ms": 0.0,
        "total_backward_time_ms": 0.0,
        "attention_backward_time_ms": 0.0,
        "ffn_backward_time_ms": 0.0,
        "transformer_others_backward_time_ms": 0.0,
        "embedding_backward_time_ms": 0.0,
        "final_norm_backward_time_ms": 0.0,
        "output_layer_backward_time_ms": 0.0,
        "non_transformer_backward_time_ms": 0.0,
        "transformer_block_backward_time_ms": 0.0,
        "attention_total_time_ms": 0.0,
        "ffn_total_time_ms": 0.0,
        "transformer_others_total_time_ms": 0.0,
        "non_transformer_total_time_ms": 0.0,
        "unattributed_total_time_ms": 0.0,
        "transformer_block_total_time_ms": 0.0,
        "attention_ratio": 0.0,
        "ffn_ratio": 0.0,
        "transformer_others_ratio": 0.0,
        "attention_backward_ratio": 0.0,
        "ffn_backward_ratio": 0.0,
        "transformer_others_backward_ratio": 0.0,
        "attention_total_ratio": 0.0,
        "ffn_total_ratio": 0.0,
        "transformer_others_total_ratio": 0.0,
        "attention_global_ratio": 0.0,
        "ffn_global_ratio": 0.0,
        "transformer_others_global_ratio": 0.0,
        "non_transformer_global_ratio": 0.0,
        "unattributed_global_ratio": 0.0,
    }


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seq in args.seq_lengths:
        run_dir = args.input_root / f"prof_seq{seq}"
        metadata = parse_env_file(run_dir / "run_metadata.env")
        status = metadata.get("status", "missing")
        report_path = run_dir / f"seq{seq}.nsys-rep"

        row: dict[str, float | str] = {
            "seq_length": seq,
            "status": status,
            "selected_gpus": metadata.get("selected_gpus", ""),
            "pp_size": metadata.get("pp_size", ""),
            "model_variant": metadata.get("model_variant", ""),
            "num_layers": metadata.get("num_layers", ""),
        }

        if status != "success" or not report_path.exists():
            row.update(empty_summary())
            rows.append(row)
            continue

        row.update(summarize_report(report_path))
        rows.append(row)

    csv_path = args.output_root / "qwen3_0p6b_pp4_seq_profile_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [row for row in rows if row["status"] == "success"]
    if not successful_rows:
        return

    save_stacked_bar_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_forward_stacked.png",
        title=f"{args.label} Transformer Forward GPU Breakdown",
        total_key="transformer_block_time_ms",
        part_keys=("attention_time_ms", "ffn_time_ms", "transformer_others_time_ms"),
        legend_labels=("Attention", "FFN", "Transformer Others"),
    )
    save_ratio_line_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_forward_ratios.png",
        title=f"{args.label} Transformer Forward GPU Ratios",
        ratio_keys=("attention_ratio", "ffn_ratio", "transformer_others_ratio"),
        legend_labels=("Attention", "FFN", "Transformer Others"),
    )
    save_stacked_bar_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_backward_stacked.png",
        title=f"{args.label} Transformer Backward GPU Breakdown",
        total_key="transformer_block_backward_time_ms",
        part_keys=(
            "attention_backward_time_ms",
            "ffn_backward_time_ms",
            "transformer_others_backward_time_ms",
        ),
        legend_labels=("Attention", "FFN", "Transformer Others"),
    )
    save_ratio_line_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_backward_ratios.png",
        title=f"{args.label} Transformer Backward GPU Ratios",
        ratio_keys=(
            "attention_backward_ratio",
            "ffn_backward_ratio",
            "transformer_others_backward_ratio",
        ),
        legend_labels=("Attention", "FFN", "Transformer Others"),
    )
    save_stacked_bar_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_total_stacked.png",
        title=f"{args.label} Transformer Total GPU Breakdown",
        total_key="transformer_block_total_time_ms",
        part_keys=(
            "attention_total_time_ms",
            "ffn_total_time_ms",
            "transformer_others_total_time_ms",
        ),
        legend_labels=("Attention", "FFN", "Transformer Others"),
    )
    save_ratio_line_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_total_ratios.png",
        title=f"{args.label} Transformer Total GPU Ratios",
        ratio_keys=(
            "attention_total_ratio",
            "ffn_total_ratio",
            "transformer_others_total_ratio",
        ),
        legend_labels=("Attention", "FFN", "Transformer Others"),
    )
    save_stacked_bar_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_global_stacked.png",
        title=f"{args.label} Global GPU Breakdown",
        total_key="train_forward_backward_time_ms",
        part_keys=(
            "attention_total_time_ms",
            "ffn_total_time_ms",
            "transformer_others_total_time_ms",
            "non_transformer_total_time_ms",
            "unattributed_total_time_ms",
        ),
        legend_labels=("Attention", "FFN", "Transformer Others", "Non-Transformer", "Unattributed"),
    )
    save_ratio_line_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_global_ratios.png",
        title=f"{args.label} Global GPU Ratios",
        ratio_keys=(
            "attention_global_ratio",
            "ffn_global_ratio",
            "transformer_others_global_ratio",
            "non_transformer_global_ratio",
            "unattributed_global_ratio",
        ),
        legend_labels=("Attention", "FFN", "Transformer Others", "Non-Transformer", "Unattributed"),
    )
    save_step_time_chart(
        successful_rows,
        args.output_root / "qwen3_0p6b_pp4_seq_step_time.png",
        title=f"{args.label} Avg GPU Step Time",
    )


if __name__ == "__main__":
    main()
