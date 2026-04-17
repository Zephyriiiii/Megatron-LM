#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import math
import sqlite3
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_SEQ_LENGTHS = [
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]
DENSE_SEQ_LENGTHS = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
POW2_SEQ_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

ATTENTION_COLOR = "#d95f02"
MLP_COLOR = "#1b9e77"
OTHERS_COLOR = "#7570b3"

CHART_WIDTH = 2200
CHART_HEIGHT = 1400
CHART_ORIGIN = (220, 1160)
CHART_PLOT_WIDTH = 1780
CHART_PLOT_HEIGHT = 860
CHART_DPI = (300, 300)
FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
)
NSYS_CANDIDATES = (
    shutil.which("nsys"),
    "/usr/local/cuda-13.2/nsight-systems-2025.6.3/target-linux-x64/nsys",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/qwen8b_1layer_single_seq_nsys"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/qwen8b_1layer_single_seq_nsys/analysis"),
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENGTHS,
    )
    parser.add_argument(
        "--label",
        default="Qwen3-8B 1-layer Single GPU",
    )
    parser.add_argument(
        "--output-prefix",
        default="qwen3_8b_1layer_single_attn_mlp",
    )
    return parser.parse_args()


def resolve_nsys() -> str:
    for candidate in NSYS_CANDIDATES:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Unable to locate nsys binary")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[float, float]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


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
        resolve_nsys(),
        "stats",
        "--force-export=true",
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


def per_step_ms_any(range_totals: dict[str, dict[str, float]], names: tuple[str, ...], step_instances: float) -> float:
    return sum(per_step_ms(range_totals, name, step_instances) for name in names)


def load_nccl_overlap_ms(
    report_path: Path,
    range_name: str,
    *,
    step_instances: float,
) -> float:
    if step_instances <= 0:
        return 0.0
    sqlite_path = report_path.with_suffix(".sqlite")
    if not sqlite_path.exists():
        return 0.0
    query = """
        SELECT
            COALESCE(
                SUM(
                    (MIN(k.end, n.end) - MAX(k.start, n.start)) / 1000000.0
                ),
                0.0
            )
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
            ON k.correlationId = r.correlationId
        JOIN StringIds s
            ON k.shortName = s.id
        JOIN NVTX_EVENTS n
            ON k.start < n.end
           AND k.end > n.start
           AND (r.globalTid >> 24) = (n.globalTid >> 24)
        WHERE n.text = ?
          AND s.value LIKE 'nccl%'
    """
    with sqlite3.connect(sqlite_path) as conn:
        value = conn.execute(query, (range_name,)).fetchone()
    total_overlap_ms = float(value[0] or 0.0)
    return total_overlap_ms / step_instances


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def summarize_report(report_path: Path) -> dict[str, float]:
    range_totals = load_nsys_nvtx_projection(report_path)
    step_instances = range_totals.get("train/forward_backward", {}).get("range_instances", 0.0)
    if step_instances <= 0:
        raise ValueError(f"No train/forward_backward instances found in {report_path}")

    step_gpu_time_ms = per_step_ms(range_totals, "train/forward_backward", step_instances)
    forward_total_ms = per_step_ms_any(
        range_totals,
        ("layer/self_attention", "layer/mlp"),
        step_instances,
    )
    backward_total_ms = per_step_ms_any(
        range_totals,
        ("layer/self_attention/backward", "layer/mlp/backward"),
        step_instances,
    )
    attention_backward_total_raw_ms = per_step_ms(
        range_totals,
        "layer/self_attention/backward",
        step_instances,
    )
    mlp_backward_total_raw_ms = per_step_ms(
        range_totals,
        "layer/mlp/backward",
        step_instances,
    )
    attention_forward_ms = per_step_ms_any(
        range_totals,
        ("megatron.core.transformer.attention.forward.core_attention",),
        step_instances,
    )
    mlp_forward_ms = per_step_ms_any(
        range_totals,
        ("layer/mlp_core",),
        step_instances,
    )
    attention_backward_ms = per_step_ms_any(
        range_totals,
        ("layer/core_attention/backward",),
        step_instances,
    )
    mlp_backward_ms = per_step_ms_any(
        range_totals,
        ("layer/mlp_core/backward",),
        step_instances,
    )
    attention_backward_comm_overlap_ms = load_nccl_overlap_ms(
        report_path,
        "layer/self_attention/backward",
        step_instances=step_instances,
    )
    mlp_backward_comm_overlap_ms = load_nccl_overlap_ms(
        report_path,
        "layer/mlp/backward",
        step_instances=step_instances,
    )
    attention_backward_total_clean_ms = max(
        attention_backward_total_raw_ms - attention_backward_comm_overlap_ms,
        0.0,
    )
    mlp_backward_total_clean_ms = max(
        mlp_backward_total_raw_ms - mlp_backward_comm_overlap_ms,
        0.0,
    )
    backward_total_clean_ms = attention_backward_total_clean_ms + mlp_backward_total_clean_ms
    others_forward_ms = max(forward_total_ms - attention_forward_ms - mlp_forward_ms, 0.0)
    others_backward_ms = max(backward_total_clean_ms - attention_backward_ms - mlp_backward_ms, 0.0)
    attention_total_ms = attention_forward_ms + attention_backward_ms
    mlp_total_ms = mlp_forward_ms + mlp_backward_ms
    others_total_ms = others_forward_ms + others_backward_ms
    total_split_ms = forward_total_ms + backward_total_clean_ms

    return {
        "step_instances": step_instances,
        "step_gpu_time_ms": step_gpu_time_ms,
        "forward_total_ms": forward_total_ms,
        "backward_total_ms": backward_total_ms,
        "attention_backward_total_raw_ms": attention_backward_total_raw_ms,
        "mlp_backward_total_raw_ms": mlp_backward_total_raw_ms,
        "attention_backward_comm_overlap_ms": attention_backward_comm_overlap_ms,
        "mlp_backward_comm_overlap_ms": mlp_backward_comm_overlap_ms,
        "backward_comm_overlap_ms": attention_backward_comm_overlap_ms + mlp_backward_comm_overlap_ms,
        "attention_backward_total_clean_ms": attention_backward_total_clean_ms,
        "mlp_backward_total_clean_ms": mlp_backward_total_clean_ms,
        "backward_total_clean_ms": backward_total_clean_ms,
        "total_split_ms": total_split_ms,
        "attention_forward_ms": attention_forward_ms,
        "mlp_forward_ms": mlp_forward_ms,
        "others_forward_ms": others_forward_ms,
        "attention_backward_ms": attention_backward_ms,
        "mlp_backward_ms": mlp_backward_ms,
        "others_backward_ms": others_backward_ms,
        "attention_total_ms": attention_total_ms,
        "mlp_total_ms": mlp_total_ms,
        "others_total_ms": others_total_ms,
        "attention_forward_ratio": safe_ratio(attention_forward_ms, forward_total_ms),
        "mlp_forward_ratio": safe_ratio(mlp_forward_ms, forward_total_ms),
        "others_forward_ratio": safe_ratio(others_forward_ms, forward_total_ms),
        "attention_backward_ratio": safe_ratio(attention_backward_ms, backward_total_clean_ms),
        "mlp_backward_ratio": safe_ratio(mlp_backward_ms, backward_total_clean_ms),
        "others_backward_ratio": safe_ratio(others_backward_ms, backward_total_clean_ms),
        "attention_total_ratio": safe_ratio(attention_total_ms, total_split_ms),
        "mlp_total_ratio": safe_ratio(mlp_total_ms, total_split_ms),
        "others_total_ratio": safe_ratio(others_total_ms, total_split_ms),
        "attn_mlp_forward_ratio": safe_ratio(attention_forward_ms, mlp_forward_ms),
        "attn_mlp_backward_ratio": safe_ratio(attention_backward_ms, mlp_backward_ms),
        "attn_mlp_total_ratio": safe_ratio(attention_total_ms, mlp_total_ms),
    }


def empty_summary() -> dict[str, float]:
    return {
        "step_instances": 0.0,
        "step_gpu_time_ms": 0.0,
        "forward_total_ms": 0.0,
        "backward_total_ms": 0.0,
        "attention_backward_total_raw_ms": 0.0,
        "mlp_backward_total_raw_ms": 0.0,
        "attention_backward_comm_overlap_ms": 0.0,
        "mlp_backward_comm_overlap_ms": 0.0,
        "backward_comm_overlap_ms": 0.0,
        "attention_backward_total_clean_ms": 0.0,
        "mlp_backward_total_clean_ms": 0.0,
        "backward_total_clean_ms": 0.0,
        "total_split_ms": 0.0,
        "attention_forward_ms": 0.0,
        "mlp_forward_ms": 0.0,
        "others_forward_ms": 0.0,
        "attention_backward_ms": 0.0,
        "mlp_backward_ms": 0.0,
        "others_backward_ms": 0.0,
        "attention_total_ms": 0.0,
        "mlp_total_ms": 0.0,
        "others_total_ms": 0.0,
        "attention_forward_ratio": 0.0,
        "mlp_forward_ratio": 0.0,
        "others_forward_ratio": 0.0,
        "attention_backward_ratio": 0.0,
        "mlp_backward_ratio": 0.0,
        "others_backward_ratio": 0.0,
        "attention_total_ratio": 0.0,
        "mlp_total_ratio": 0.0,
        "others_total_ratio": 0.0,
        "attn_mlp_forward_ratio": 0.0,
        "attn_mlp_backward_ratio": 0.0,
        "attn_mlp_total_ratio": 0.0,
    }


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def measure_x_positions(rows, *, x_mode: str) -> list[float]:
    if not rows:
        return []
    if len(rows) == 1:
        return [CHART_ORIGIN[0] + CHART_PLOT_WIDTH / 2]
    if x_mode == "equal":
        step = CHART_PLOT_WIDTH / max(len(rows) - 1, 1)
        return [CHART_ORIGIN[0] + idx * step for idx in range(len(rows))]
    seqs = [int(row["seq_length"]) for row in rows]
    if x_mode == "log2":
        min_seq = min(seqs)
        max_seq = max(seqs)
        min_pos = math.log2(min_seq)
        max_pos = math.log2(max_seq)
        if max_pos <= min_pos:
            return [CHART_ORIGIN[0] + CHART_PLOT_WIDTH / 2 for _ in rows]
        return [
            CHART_ORIGIN[0] + CHART_PLOT_WIDTH * ((math.log2(seq) - min_pos) / (max_pos - min_pos))
            for seq in seqs
        ]
    min_seq = min(seqs)
    max_seq = max(seqs)
    if max_seq <= min_seq:
        return [CHART_ORIGIN[0] + CHART_PLOT_WIDTH / 2 for _ in rows]
    return [
        CHART_ORIGIN[0] + CHART_PLOT_WIDTH * ((seq - min_seq) / (max_seq - min_seq))
        for seq in seqs
    ]


def draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    tick_font: ImageFont.ImageFont,
):
    x0, y0 = CHART_ORIGIN
    draw.line((x0, y0, x0, y0 - CHART_PLOT_HEIGHT), fill="black", width=4)
    draw.line((x0, y0, x0 + CHART_PLOT_WIDTH, y0), fill="black", width=4)
    for x, label in x_ticks:
        draw.line((x, y0, x, y0 + 14), fill="black", width=2)
        label_width, _ = measure_text(draw, label, tick_font)
        draw.text((x - label_width / 2, y0 + 20), label, fill="black", font=tick_font)
    for y, label in y_ticks:
        draw.line((x0 - 14, y, x0, y), fill="black", width=2)
        label_width, label_height = measure_text(draw, label, tick_font)
        draw.text((x0 - 24 - label_width, y - label_height / 2), label, fill="black", font=tick_font)


def save_dual_line_chart(
    rows,
    output_path: Path,
    *,
    title: str,
    attention_key: str,
    mlp_key: str,
    ratio_chart: bool,
    x_mode: str,
):
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(46)
    subtitle_font = load_font(24)
    tick_font = load_font(22)
    value_font = load_font(18)
    legend_font = load_font(24)

    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((CHART_WIDTH - title_width) / 2, 28), title, fill="black", font=title_font)
    subtitle = (
        "Y axis: share of train/forward_backward step time (%)"
        if ratio_chart
        else "Y axis: avg GPU ms per profiled step"
    )
    draw.text((40, 92), subtitle, fill="black", font=subtitle_font)

    x_positions = measure_x_positions(rows, x_mode=x_mode)
    max_value = max(
        max(float(row[attention_key]), float(row[mlp_key]))
        for row in rows
    )
    max_value = max(max_value * 1.1, 1e-6)

    x_ticks = [(x, str(row["seq_length"])) for x, row in zip(x_positions, rows)]
    y_ticks = []
    for i in range(6):
        value = max_value * i / 5
        y = CHART_ORIGIN[1] - (value / max_value) * CHART_PLOT_HEIGHT
        label = f"{value * 100:.1f}%" if ratio_chart else f"{value:.1f}"
        y_ticks.append((y, label))

    draw_axes(draw, x_ticks=x_ticks, y_ticks=y_ticks, tick_font=tick_font)

    def y_of(value: float) -> float:
        return CHART_ORIGIN[1] - (value / max_value) * CHART_PLOT_HEIGHT

    attn_points = [(x, y_of(float(row[attention_key]))) for x, row in zip(x_positions, rows)]
    mlp_points = [(x, y_of(float(row[mlp_key]))) for x, row in zip(x_positions, rows)]
    draw.line(attn_points, fill=ATTENTION_COLOR, width=5)
    draw.line(mlp_points, fill=MLP_COLOR, width=5)

    for (x, y), row in zip(attn_points, rows):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=ATTENTION_COLOR, outline="black", width=2)
        value = float(row[attention_key])
        text = f"{value * 100:.1f}%" if ratio_chart else f"{value:.2f}"
        width, height = measure_text(draw, text, value_font)
        draw.text((x - width / 2, y - height - 12), text, fill=ATTENTION_COLOR, font=value_font)

    for (x, y), row in zip(mlp_points, rows):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=MLP_COLOR, outline="black", width=2)
        value = float(row[mlp_key])
        text = f"{value * 100:.1f}%" if ratio_chart else f"{value:.2f}"
        width, height = measure_text(draw, text, value_font)
        draw.text((x - width / 2, y + 10), text, fill=MLP_COLOR, font=value_font)

    legend_y = 124
    draw.rectangle((1500, legend_y, 1540, legend_y + 24), fill=ATTENTION_COLOR, outline="black", width=2)
    draw.text((1552, legend_y - 4), "Attention", fill="black", font=legend_font)
    draw.rectangle((1750, legend_y, 1790, legend_y + 24), fill=MLP_COLOR, outline="black", width=2)
    draw.text((1802, legend_y - 4), "MLP", fill="black", font=legend_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=CHART_DPI)


def save_ratio_split_chart(
    rows,
    output_path: Path,
    *,
    title: str,
    attention_key: str,
    mlp_key: str,
    others_key: str,
    x_mode: str,
):
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(46)
    subtitle_font = load_font(24)
    tick_font = load_font(22)
    value_font = load_font(18)
    legend_font = load_font(24)

    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((CHART_WIDTH - title_width) / 2, 28), title, fill="black", font=title_font)
    draw.text((40, 92), "Y axis: share within Attention+MLP split (%)", fill="black", font=subtitle_font)

    x_positions = measure_x_positions(rows, x_mode=x_mode)
    max_value = 1.0

    x_ticks = [(x, str(row["seq_length"])) for x, row in zip(x_positions, rows)]
    y_ticks = []
    for i in range(6):
        value = max_value * i / 5
        y = CHART_ORIGIN[1] - (value / max_value) * CHART_PLOT_HEIGHT
        y_ticks.append((y, f"{value * 100:.1f}%"))

    draw_axes(draw, x_ticks=x_ticks, y_ticks=y_ticks, tick_font=tick_font)

    def y_of(value: float) -> float:
        return CHART_ORIGIN[1] - (value / max_value) * CHART_PLOT_HEIGHT

    series = (
        ("Attention", ATTENTION_COLOR, attention_key, -12),
        ("MLP", MLP_COLOR, mlp_key, 10),
        ("Others", OTHERS_COLOR, others_key, 30),
    )
    legend_x = 1340
    legend_y = 124
    for idx, (label, color, key, offset_y) in enumerate(series):
        points = [(x, y_of(float(row[key]))) for x, row in zip(x_positions, rows)]
        draw.line(points, fill=color, width=5)
        for (x, y), row in zip(points, rows):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="black", width=2)
            value = float(row[key])
            text = f"{value * 100:.1f}%"
            width, height = measure_text(draw, text, value_font)
            draw.text((x - width / 2, y + offset_y), text, fill=color, font=value_font)
        lx = legend_x + idx * 250
        draw.rectangle((lx, legend_y, lx + 40, legend_y + 24), fill=color, outline="black", width=2)
        draw.text((lx + 52, legend_y - 4), label, fill="black", font=legend_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=CHART_DPI)


def emit_chart_bundle(
    rows,
    output_root: Path,
    *,
    prefix: str,
    output_prefix: str,
    x_mode: str,
):
    if not rows:
        return
    subdir = output_root / prefix
    subdir.mkdir(parents=True, exist_ok=True)
    chart_specs = (
        ("forward_absolute", "Forward Attention vs MLP(core)", "attention_forward_ms", "mlp_forward_ms", False),
        ("backward_absolute", "Backward Attention vs MLP(core)", "attention_backward_ms", "mlp_backward_ms", False),
        ("total_absolute", "Total Attention vs MLP(core)", "attention_total_ms", "mlp_total_ms", False),
    )
    for stem, title, attn_key, mlp_key, ratio_chart in chart_specs:
        save_dual_line_chart(
            rows,
            subdir / f"{output_prefix}_{prefix}_{stem}.png",
            title=title,
            attention_key=attn_key,
            mlp_key=mlp_key,
            ratio_chart=ratio_chart,
            x_mode=x_mode,
        )
    ratio_specs = (
        ("forward_split_ratio", "Forward Attention / MLP / Others", "attention_forward_ratio", "mlp_forward_ratio", "others_forward_ratio"),
        ("backward_split_ratio", "Backward Attention / MLP / Others", "attention_backward_ratio", "mlp_backward_ratio", "others_backward_ratio"),
        ("total_split_ratio", "Total Attention / MLP / Others", "attention_total_ratio", "mlp_total_ratio", "others_total_ratio"),
    )
    for stem, title, attn_key, mlp_key, others_key in ratio_specs:
        save_ratio_split_chart(
            rows,
            subdir / f"{output_prefix}_{prefix}_{stem}.png",
            title=title,
            attention_key=attn_key,
            mlp_key=mlp_key,
            others_key=others_key,
            x_mode=x_mode,
        )


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
            "parallel_mode": metadata.get("parallel_mode", ""),
        }
        if status != "success" or not report_path.exists():
            row.update(empty_summary())
            rows.append(row)
            continue
        row.update(summarize_report(report_path))
        rows.append(row)

    csv_path = args.output_root / f"{args.output_prefix}_profile_summary.csv"
    write_summary_csv(csv_path, rows)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        return

    dense_rows = [row for row in success_rows if int(row["seq_length"]) in DENSE_SEQ_LENGTHS]
    pow2_rows = [row for row in success_rows if int(row["seq_length"]) in POW2_SEQ_LENGTHS]

    emit_chart_bundle(
        success_rows,
        args.output_root,
        prefix="all",
        output_prefix=args.output_prefix,
        x_mode="actual",
    )
    emit_chart_bundle(
        dense_rows,
        args.output_root,
        prefix="dense",
        output_prefix=args.output_prefix,
        x_mode="actual",
    )
    emit_chart_bundle(
        pow2_rows,
        args.output_root,
        prefix="pow2",
        output_prefix=args.output_prefix,
        x_mode="log2",
    )


if __name__ == "__main__":
    main()
