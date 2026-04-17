#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXP_LINE_SEQ_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
LINEAR_LINE_SEQ_LENGTHS = [131072, 262144, 393216, 524288, 655360, 786432, 917504]
BOUNDARY_SEQ_LENGTHS = [1048576]
DEFAULT_SEQ_LENGTHS = sorted(
    set(EXP_LINE_SEQ_LENGTHS) | set(LINEAR_LINE_SEQ_LENGTHS) | set(BOUNDARY_SEQ_LENGTHS)
)

PART_COLORS = {
    "attention": "#d95f02",
    "attention_compute": "#d95f02",
    "attention_comm": "#e7298a",
    "attention_comm_related": "#e7298a",
    "attention_non_comm": "#d95f02",
    "attention_non_comm_only": "#d95f02",
    "attention_comm_only": "#e7298a",
    "attention_overlap": "#7570b3",
    "attention_other": "#a6761d",
    "attention_path_total": "#8c510a",
    "non_attention_only": "#666666",
    "ffn": "#1b9e77",
    "transformer_others": "#7570b3",
    "non_transformer": "#666666",
    "backward_unattributed": "#e6ab02",
    "unattributed": "#e7298a",
}

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


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[float, float]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def save_chart(image: Image.Image, output_path: Path):
    image.save(output_path, dpi=CHART_DPI)


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

    if normalized.startswith("attention_compute"):
        return "attention_compute"
    if normalized.startswith("attention_comm_related"):
        return "attention_comm_related"
    if normalized.startswith("attention_non_comm_only"):
        return "attention_non_comm_only"
    if normalized.startswith("attention_comm_only"):
        return "attention_comm_only"
    if normalized.startswith("attention_overlap"):
        return "attention_overlap"
    if normalized.startswith("attention_non_comm"):
        return "attention_non_comm"
    if normalized.startswith("attention_comm"):
        return "attention_comm"
    if normalized.startswith("attention_other"):
        return "attention_other"
    if normalized.startswith("attention_path_total"):
        return "attention_path_total"
    if normalized.startswith("non_attention_only"):
        return "non_attention_only"
    if normalized.startswith("attention"):
        return "attention"
    if normalized.startswith("ffn"):
        return "ffn"
    if normalized.startswith("transformer_others"):
        return "transformer_others"
    if normalized.startswith("non_transformer"):
        return "non_transformer"
    if normalized.startswith("backward_unattributed"):
        return "backward_unattributed"
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
        default=DEFAULT_SEQ_LENGTHS,
        help="Sequence lengths to aggregate in order.",
    )
    parser.add_argument(
        "--label",
        default="Qwen3-0.6B PP4",
        help="Short experiment label to include in chart titles.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("auto", "cp", "non_cp"),
        default="auto",
        help="How to render attention decomposition. 'cp' splits internal comm/overlap, 'non_cp' keeps a single attention bucket.",
    )
    parser.add_argument(
        "--output-prefix",
        default="qwen3_0p6b_pp4_seq",
        help="Prefix used for generated CSV and PNG filenames.",
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


def per_step_ms_any(
    range_totals: dict[str, dict[str, float]],
    names: tuple[str, ...],
    step_instances: float,
) -> float:
    return sum(per_step_ms(range_totals, name, step_instances) for name in names)


def load_nsys_nvtx_projection(report_path: Path) -> dict[str, dict[str, float]]:
    cmd = [
        "nsys",
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


def classify_attention_kernel(kernel_name: str) -> str:
    normalized = kernel_name.lower()
    if normalized.startswith("nccl"):
        return "comm"
    if normalized:
        return "compute"
    return "other"


def classify_attention_path_kernel(kernel_name: str) -> str:
    normalized = kernel_name.lower()
    if normalized.startswith("nccl"):
        return "comm"
    return "non_comm"


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def merged_interval_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def load_step_union_attention_accounting(sqlite_path: Path) -> dict[str, float]:
    if not sqlite_path.exists():
        return {
            "step_union_instances": 0.0,
            "step_gpu_active_union_ms": 0.0,
            "attention_non_comm_union_ms": 0.0,
            "attention_comm_union_ms": 0.0,
            "attention_path_union_ms": 0.0,
            "attention_overlap_ms": 0.0,
            "attention_non_comm_only_ms": 0.0,
            "attention_comm_only_ms": 0.0,
            "attention_non_comm_inclusive_ratio": 0.0,
            "attention_comm_inclusive_ratio": 0.0,
            "attention_non_comm_only_ratio": 0.0,
            "attention_comm_only_ratio": 0.0,
            "attention_overlap_ratio": 0.0,
            "attention_path_ratio": 0.0,
        }

    step_query = """
    SELECT rowid, start, end, globalTid
    FROM NVTX_EVENTS
    WHERE text = 'train/forward_backward'
      AND end > start
    ORDER BY start
    """
    step_kernels_query = """
    SELECT DISTINCT k.rowid, k.start, k.end, COALESCE(s.value, '') AS kernel_name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k
      ON k.correlationId = r.correlationId
    LEFT JOIN StringIds s
      ON s.id = k.shortName
    WHERE (r.globalTid >> 24) = ?
      AND r.start >= ?
      AND r.start < ?
    ORDER BY k.start, k.end
    """
    attention_kernels_query_template = """
    SELECT DISTINCT k.rowid, k.start, k.end, COALESCE(s.value, '') AS kernel_name
    FROM NVTX_EVENTS a
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
      ON r.globalTid = a.globalTid
     AND r.start >= a.start
     AND r.start < a.end
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k
      ON k.correlationId = r.correlationId
    LEFT JOIN StringIds s
      ON s.id = k.shortName
    WHERE (a.globalTid >> 24) = ?
      AND a.start >= ?
      AND a.start < ?
      AND a.text IN ({placeholders})
    ORDER BY k.start, k.end
    """

    totals = {
        "step_gpu_active_union_ns": 0.0,
        "attention_forward_non_comm_union_ns": 0.0,
        "attention_forward_comm_union_ns": 0.0,
        "attention_forward_path_union_ns": 0.0,
        "attention_forward_overlap_ns": 0.0,
        "attention_forward_non_comm_only_ns": 0.0,
        "attention_forward_comm_only_ns": 0.0,
        "attention_backward_non_comm_union_ns": 0.0,
        "attention_backward_comm_union_ns": 0.0,
        "attention_backward_path_union_ns": 0.0,
        "attention_backward_overlap_ns": 0.0,
        "attention_backward_non_comm_only_ns": 0.0,
        "attention_backward_comm_only_ns": 0.0,
        "attention_non_comm_union_ns": 0.0,
        "attention_comm_union_ns": 0.0,
        "attention_path_union_ns": 0.0,
        "attention_overlap_ns": 0.0,
        "attention_non_comm_only_ns": 0.0,
        "attention_comm_only_ns": 0.0,
        "non_attention_only_ns": 0.0,
    }

    def collect_attention_intervals(
        conn: sqlite3.Connection,
        *,
        process_key: int,
        step_start: int,
        step_end: int,
        range_names: tuple[str, ...],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        placeholders = ", ".join("?" for _ in range_names)
        query = attention_kernels_query_template.format(placeholders=placeholders)
        params = (process_key, step_start, step_end, *range_names)
        non_comm_intervals: list[tuple[int, int]] = []
        comm_intervals: list[tuple[int, int]] = []
        for _, kernel_start, kernel_end, kernel_name in conn.execute(query, params):
            if kernel_end <= kernel_start:
                continue
            interval = (int(kernel_start), int(kernel_end))
            if classify_attention_path_kernel(kernel_name) == "comm":
                comm_intervals.append(interval)
            else:
                non_comm_intervals.append(interval)
        return non_comm_intervals, comm_intervals

    with sqlite3.connect(sqlite_path) as conn:
        step_events = list(conn.execute(step_query))
        for _, step_start, step_end, global_tid in step_events:
            process_key = int(global_tid) >> 24
            step_intervals: list[tuple[int, int]] = []
            for _, kernel_start, kernel_end, _ in conn.execute(
                step_kernels_query, (process_key, step_start, step_end)
            ):
                if kernel_end <= kernel_start:
                    continue
                step_intervals.append((int(kernel_start), int(kernel_end)))
            step_union_ns = merged_interval_ns(step_intervals)

            forward_non_comm_intervals, forward_comm_intervals = collect_attention_intervals(
                conn,
                process_key=process_key,
                step_start=step_start,
                step_end=step_end,
                range_names=("layer/self_attention_core",),
            )
            backward_non_comm_intervals, backward_comm_intervals = collect_attention_intervals(
                conn,
                process_key=process_key,
                step_start=step_start,
                step_end=step_end,
                range_names=("layer/self_attention/backward",),
            )

            def compute_partition(
                non_comm_intervals: list[tuple[int, int]], comm_intervals: list[tuple[int, int]]
            ) -> tuple[int, int, int, int, int, int]:
                non_comm_union_ns = merged_interval_ns(non_comm_intervals)
                comm_union_ns = merged_interval_ns(comm_intervals)
                path_union_ns = merged_interval_ns(non_comm_intervals + comm_intervals)
                overlap_ns = max(
                    non_comm_union_ns + comm_union_ns - path_union_ns,
                    0,
                )
                non_comm_only_ns = max(non_comm_union_ns - overlap_ns, 0)
                comm_only_ns = max(comm_union_ns - overlap_ns, 0)
                return (
                    non_comm_union_ns,
                    comm_union_ns,
                    path_union_ns,
                    overlap_ns,
                    non_comm_only_ns,
                    comm_only_ns,
                )

            (
                attention_forward_non_comm_union_ns,
                attention_forward_comm_union_ns,
                attention_forward_path_union_ns,
                attention_forward_overlap_ns,
                attention_forward_non_comm_only_ns,
                attention_forward_comm_only_ns,
            ) = compute_partition(forward_non_comm_intervals, forward_comm_intervals)
            (
                attention_backward_non_comm_union_ns,
                attention_backward_comm_union_ns,
                attention_backward_path_union_ns,
                attention_backward_overlap_ns,
                attention_backward_non_comm_only_ns,
                attention_backward_comm_only_ns,
            ) = compute_partition(backward_non_comm_intervals, backward_comm_intervals)
            (
                attention_non_comm_union_ns,
                attention_comm_union_ns,
                attention_path_union_ns,
                attention_overlap_ns,
                attention_non_comm_only_ns,
                attention_comm_only_ns,
            ) = compute_partition(
                forward_non_comm_intervals + backward_non_comm_intervals,
                forward_comm_intervals + backward_comm_intervals,
            )
            non_attention_only_ns = max(
                step_union_ns - attention_path_union_ns,
                0,
            )

            totals["step_gpu_active_union_ns"] += step_union_ns
            totals["attention_forward_non_comm_union_ns"] += attention_forward_non_comm_union_ns
            totals["attention_forward_comm_union_ns"] += attention_forward_comm_union_ns
            totals["attention_forward_path_union_ns"] += attention_forward_path_union_ns
            totals["attention_forward_overlap_ns"] += attention_forward_overlap_ns
            totals["attention_forward_non_comm_only_ns"] += attention_forward_non_comm_only_ns
            totals["attention_forward_comm_only_ns"] += attention_forward_comm_only_ns
            totals["attention_backward_non_comm_union_ns"] += attention_backward_non_comm_union_ns
            totals["attention_backward_comm_union_ns"] += attention_backward_comm_union_ns
            totals["attention_backward_path_union_ns"] += attention_backward_path_union_ns
            totals["attention_backward_overlap_ns"] += attention_backward_overlap_ns
            totals["attention_backward_non_comm_only_ns"] += attention_backward_non_comm_only_ns
            totals["attention_backward_comm_only_ns"] += attention_backward_comm_only_ns
            totals["attention_non_comm_union_ns"] += attention_non_comm_union_ns
            totals["attention_comm_union_ns"] += attention_comm_union_ns
            totals["attention_path_union_ns"] += attention_path_union_ns
            totals["attention_overlap_ns"] += attention_overlap_ns
            totals["attention_non_comm_only_ns"] += attention_non_comm_only_ns
            totals["attention_comm_only_ns"] += attention_comm_only_ns
            totals["non_attention_only_ns"] += non_attention_only_ns

    step_union_instances = float(len(step_events))
    if step_union_instances <= 0:
        return {
            "step_union_instances": 0.0,
            "step_gpu_active_union_ms": 0.0,
            "attention_forward_non_comm_union_ms": 0.0,
            "attention_forward_comm_union_ms": 0.0,
            "attention_forward_path_union_ms": 0.0,
            "attention_forward_overlap_ms": 0.0,
            "attention_forward_non_comm_only_ms": 0.0,
            "attention_forward_comm_only_ms": 0.0,
            "attention_backward_non_comm_union_ms": 0.0,
            "attention_backward_comm_union_ms": 0.0,
            "attention_backward_path_union_ms": 0.0,
            "attention_backward_overlap_ms": 0.0,
            "attention_backward_non_comm_only_ms": 0.0,
            "attention_backward_comm_only_ms": 0.0,
            "attention_non_comm_union_ms": 0.0,
            "attention_comm_union_ms": 0.0,
            "attention_path_union_ms": 0.0,
            "attention_overlap_ms": 0.0,
            "attention_non_comm_only_ms": 0.0,
            "attention_comm_only_ms": 0.0,
            "non_attention_only_ms": 0.0,
            "attention_forward_non_comm_only_ratio": 0.0,
            "attention_forward_comm_only_ratio": 0.0,
            "attention_forward_overlap_ratio": 0.0,
            "attention_backward_non_comm_only_ratio": 0.0,
            "attention_backward_comm_only_ratio": 0.0,
            "attention_backward_overlap_ratio": 0.0,
            "attention_non_comm_only_total_ratio": 0.0,
            "attention_comm_only_total_ratio": 0.0,
            "attention_overlap_total_ratio": 0.0,
            "attention_non_comm_inclusive_ratio": 0.0,
            "attention_comm_inclusive_ratio": 0.0,
            "attention_non_comm_only_ratio": 0.0,
            "attention_comm_only_ratio": 0.0,
            "attention_overlap_ratio": 0.0,
            "attention_path_ratio": 0.0,
            "non_attention_only_ratio": 0.0,
        }

    step_gpu_active_union_ms = totals["step_gpu_active_union_ns"] / step_union_instances / 1_000_000.0
    attention_forward_non_comm_union_ms = (
        totals["attention_forward_non_comm_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_forward_comm_union_ms = (
        totals["attention_forward_comm_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_forward_path_union_ms = (
        totals["attention_forward_path_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_forward_overlap_ms = (
        totals["attention_forward_overlap_ns"] / step_union_instances / 1_000_000.0
    )
    attention_forward_non_comm_only_ms = (
        totals["attention_forward_non_comm_only_ns"] / step_union_instances / 1_000_000.0
    )
    attention_forward_comm_only_ms = (
        totals["attention_forward_comm_only_ns"] / step_union_instances / 1_000_000.0
    )
    attention_backward_non_comm_union_ms = (
        totals["attention_backward_non_comm_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_backward_comm_union_ms = (
        totals["attention_backward_comm_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_backward_path_union_ms = (
        totals["attention_backward_path_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_backward_overlap_ms = (
        totals["attention_backward_overlap_ns"] / step_union_instances / 1_000_000.0
    )
    attention_backward_non_comm_only_ms = (
        totals["attention_backward_non_comm_only_ns"] / step_union_instances / 1_000_000.0
    )
    attention_backward_comm_only_ms = (
        totals["attention_backward_comm_only_ns"] / step_union_instances / 1_000_000.0
    )
    attention_non_comm_union_ms = (
        totals["attention_non_comm_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_comm_union_ms = (
        totals["attention_comm_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_path_union_ms = (
        totals["attention_path_union_ns"] / step_union_instances / 1_000_000.0
    )
    attention_overlap_ms = totals["attention_overlap_ns"] / step_union_instances / 1_000_000.0
    attention_non_comm_only_ms = (
        totals["attention_non_comm_only_ns"] / step_union_instances / 1_000_000.0
    )
    attention_comm_only_ms = totals["attention_comm_only_ns"] / step_union_instances / 1_000_000.0
    non_attention_only_ms = totals["non_attention_only_ns"] / step_union_instances / 1_000_000.0

    ratio_denom = step_gpu_active_union_ms
    forward_ratio_denom = attention_forward_path_union_ms
    backward_ratio_denom = attention_backward_path_union_ms
    total_ratio_denom = attention_path_union_ms
    return {
        "step_union_instances": step_union_instances,
        "step_gpu_active_union_ms": step_gpu_active_union_ms,
        "attention_forward_non_comm_union_ms": attention_forward_non_comm_union_ms,
        "attention_forward_comm_union_ms": attention_forward_comm_union_ms,
        "attention_forward_path_union_ms": attention_forward_path_union_ms,
        "attention_forward_overlap_ms": attention_forward_overlap_ms,
        "attention_forward_non_comm_only_ms": attention_forward_non_comm_only_ms,
        "attention_forward_comm_only_ms": attention_forward_comm_only_ms,
        "attention_backward_non_comm_union_ms": attention_backward_non_comm_union_ms,
        "attention_backward_comm_union_ms": attention_backward_comm_union_ms,
        "attention_backward_path_union_ms": attention_backward_path_union_ms,
        "attention_backward_overlap_ms": attention_backward_overlap_ms,
        "attention_backward_non_comm_only_ms": attention_backward_non_comm_only_ms,
        "attention_backward_comm_only_ms": attention_backward_comm_only_ms,
        "attention_non_comm_union_ms": attention_non_comm_union_ms,
        "attention_comm_union_ms": attention_comm_union_ms,
        "attention_path_union_ms": attention_path_union_ms,
        "attention_overlap_ms": attention_overlap_ms,
        "attention_non_comm_only_ms": attention_non_comm_only_ms,
        "attention_comm_only_ms": attention_comm_only_ms,
        "non_attention_only_ms": non_attention_only_ms,
        "attention_forward_non_comm_only_ratio": (
            0.0 if forward_ratio_denom <= 0 else attention_forward_non_comm_only_ms / forward_ratio_denom
        ),
        "attention_forward_comm_only_ratio": (
            0.0 if forward_ratio_denom <= 0 else attention_forward_comm_only_ms / forward_ratio_denom
        ),
        "attention_forward_overlap_ratio": (
            0.0 if forward_ratio_denom <= 0 else attention_forward_overlap_ms / forward_ratio_denom
        ),
        "attention_backward_non_comm_only_ratio": (
            0.0
            if backward_ratio_denom <= 0
            else attention_backward_non_comm_only_ms / backward_ratio_denom
        ),
        "attention_backward_comm_only_ratio": (
            0.0
            if backward_ratio_denom <= 0
            else attention_backward_comm_only_ms / backward_ratio_denom
        ),
        "attention_backward_overlap_ratio": (
            0.0 if backward_ratio_denom <= 0 else attention_backward_overlap_ms / backward_ratio_denom
        ),
        "attention_non_comm_only_total_ratio": (
            0.0 if total_ratio_denom <= 0 else attention_non_comm_only_ms / total_ratio_denom
        ),
        "attention_comm_only_total_ratio": (
            0.0 if total_ratio_denom <= 0 else attention_comm_only_ms / total_ratio_denom
        ),
        "attention_overlap_total_ratio": (
            0.0 if total_ratio_denom <= 0 else attention_overlap_ms / total_ratio_denom
        ),
        "attention_non_comm_inclusive_ratio": (
            0.0 if ratio_denom <= 0 else attention_non_comm_union_ms / ratio_denom
        ),
        "attention_comm_inclusive_ratio": (
            0.0 if ratio_denom <= 0 else attention_comm_union_ms / ratio_denom
        ),
        "attention_non_comm_only_ratio": (
            0.0 if ratio_denom <= 0 else attention_non_comm_only_ms / ratio_denom
        ),
        "attention_comm_only_ratio": (
            0.0 if ratio_denom <= 0 else attention_comm_only_ms / ratio_denom
        ),
        "attention_overlap_ratio": (
            0.0 if ratio_denom <= 0 else attention_overlap_ms / ratio_denom
        ),
        "attention_path_ratio": (
            0.0 if ratio_denom <= 0 else attention_path_union_ms / ratio_denom
        ),
        "non_attention_only_ratio": (
            0.0 if ratio_denom <= 0 else non_attention_only_ms / ratio_denom
        ),
    }


def load_attention_kernel_breakdown(sqlite_path: Path, range_name: str) -> dict[str, float]:
    if not sqlite_path.exists():
        return {"compute_ns": 0.0, "comm_ns": 0.0, "other_ns": 0.0}

    query = """
    WITH target_events AS (
        SELECT start AS ev_start, end AS ev_end, globalTid
        FROM NVTX_EVENTS
        WHERE text = ?
    ),
    runtime_in_range AS (
        SELECT DISTINCT r.correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN target_events t
          ON r.globalTid = t.globalTid
         AND r.start >= t.ev_start
         AND r.end <= t.ev_end
    )
    SELECT COALESCE(s.value, '') AS kernel_name, SUM(k.end - k.start) AS total_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN runtime_in_range rr
      ON k.correlationId = rr.correlationId
    LEFT JOIN StringIds s
      ON k.shortName = s.id
    GROUP BY kernel_name
    """

    breakdown = {"compute_ns": 0.0, "comm_ns": 0.0, "other_ns": 0.0}
    with sqlite3.connect(sqlite_path) as conn:
        for kernel_name, total_ns in conn.execute(query, (range_name,)):
            kind = classify_attention_kernel(kernel_name)
            breakdown[f"{kind}_ns"] += float(total_ns or 0.0)
    return breakdown


def per_step_ms_from_ns(total_ns: float, step_instances: float) -> float:
    if step_instances <= 0:
        return 0.0
    return total_ns / step_instances / 1_000_000.0


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
    sqlite_path = report_path.with_suffix(".sqlite")
    step_union_accounting = load_step_union_attention_accounting(sqlite_path)
    step_instances = range_totals.get("train/forward_backward", {}).get("range_instances", 0.0)
    if step_instances <= 0:
        raise ValueError(f"No train/forward_backward NVTX instances found in {report_path}")

    train_forward_backward_time_ms = per_step_ms(range_totals, "train/forward_backward", step_instances)
    train_optimizer_step_time_ms = per_step_ms(range_totals, "train/optimizer_step", step_instances)

    transformer_block_time_ms = per_step_ms(range_totals, "layer/transformer_block", step_instances)
    attention_time_ms = per_step_ms(range_totals, "layer/self_attention_core", step_instances)
    ffn_time_ms = per_step_ms(range_totals, "layer/mlp_core", step_instances)
    attention_forward_breakdown = load_attention_kernel_breakdown(sqlite_path, "layer/self_attention_core")
    attention_compute_time_ms = per_step_ms_from_ns(
        attention_forward_breakdown["compute_ns"], step_instances
    )
    attention_comm_time_ms = per_step_ms_from_ns(
        attention_forward_breakdown["comm_ns"], step_instances
    )
    attention_other_time_ms = per_step_ms_from_ns(
        attention_forward_breakdown["other_ns"], step_instances
    )
    attention_path_total_forward_time_ms = (
        attention_compute_time_ms + attention_comm_time_ms + attention_other_time_ms
    )
    transformer_others_time_ms = max(
        transformer_block_time_ms - attention_time_ms - ffn_time_ms,
        0.0,
    )

    embedding_time_ms = per_step_ms_any(
        range_totals,
        (
            "gpt/embedding",
            "megatron.core.models.common.embeddings.language_model_embedding.forward",
        ),
        step_instances,
    )
    final_norm_time_ms = per_step_ms(range_totals, "gpt/final_norm", step_instances)
    output_layer_time_ms = per_step_ms(range_totals, "gpt/output_layer", step_instances)
    loss_time_ms = per_step_ms(range_totals, "gpt/loss", step_instances)
    lm_head_or_postprocess_time_ms = output_layer_time_ms + loss_time_ms
    non_transformer_forward_time_ms = (
        embedding_time_ms + final_norm_time_ms + lm_head_or_postprocess_time_ms
    )

    total_backward_time_ms = max(
        train_forward_backward_time_ms - transformer_block_time_ms - non_transformer_forward_time_ms,
        0.0,
    )
    attention_backward_time_ms = per_step_ms(range_totals, "layer/self_attention/backward", step_instances)
    attention_backward_breakdown = load_attention_kernel_breakdown(
        sqlite_path, "layer/self_attention/backward"
    )
    attention_compute_backward_time_ms = per_step_ms_from_ns(
        attention_backward_breakdown["compute_ns"], step_instances
    )
    attention_comm_backward_time_ms = per_step_ms_from_ns(
        attention_backward_breakdown["comm_ns"], step_instances
    )
    attention_other_backward_time_ms = per_step_ms_from_ns(
        attention_backward_breakdown["other_ns"], step_instances
    )
    attention_path_total_backward_time_ms = (
        attention_compute_backward_time_ms
        + attention_comm_backward_time_ms
        + attention_other_backward_time_ms
    )
    ffn_backward_time_ms = per_step_ms(range_totals, "layer/mlp/backward", step_instances)
    input_layernorm_backward_time_ms = per_step_ms(
        range_totals, "layer/input_layernorm/backward", step_instances
    )
    pre_mlp_layernorm_backward_time_ms = per_step_ms(
        range_totals, "layer/pre_mlp_layernorm/backward", step_instances
    )
    self_attn_bda_backward_time_ms = per_step_ms(
        range_totals, "layer/self_attn_bda/backward", step_instances
    )
    mlp_bda_backward_time_ms = per_step_ms(range_totals, "layer/mlp_bda/backward", step_instances)
    embedding_backward_time_ms = per_step_ms_any(
        range_totals,
        (
            "gpt/embedding/backward",
            "megatron.core.models.common.embeddings.language_model_embedding.backward",
        ),
        step_instances,
    )
    final_norm_backward_time_ms = per_step_ms(range_totals, "gpt/final_norm/backward", step_instances)
    output_layer_backward_time_ms = per_step_ms(range_totals, "gpt/output_layer/backward", step_instances)
    loss_backward_time_ms = per_step_ms(range_totals, "gpt/loss/backward", step_instances)
    non_transformer_backward_time_ms = (
        embedding_backward_time_ms
        + final_norm_backward_time_ms
        + output_layer_backward_time_ms
        + loss_backward_time_ms
    )
    transformer_others_backward_time_ms = (
        input_layernorm_backward_time_ms
        + pre_mlp_layernorm_backward_time_ms
        + self_attn_bda_backward_time_ms
        + mlp_bda_backward_time_ms
    )
    backward_unattributed_time_ms = max(
        total_backward_time_ms
        - attention_backward_time_ms
        - ffn_backward_time_ms
        - transformer_others_backward_time_ms
        - non_transformer_backward_time_ms,
        0.0,
    )

    attention_total_time_ms = attention_time_ms + attention_backward_time_ms
    attention_compute_total_time_ms = (
        attention_compute_time_ms + attention_compute_backward_time_ms
    )
    attention_comm_total_time_ms = attention_comm_time_ms + attention_comm_backward_time_ms
    attention_other_total_time_ms = attention_other_time_ms + attention_other_backward_time_ms
    attention_path_total_time_ms = (
        attention_compute_total_time_ms
        + attention_comm_total_time_ms
        + attention_other_total_time_ms
    )
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
        - backward_unattributed_time_ms,
        0.0,
    )

    summary = {
        "step_instances": step_instances,
        "train_forward_backward_time_ms": train_forward_backward_time_ms,
        "train_optimizer_step_time_ms": train_optimizer_step_time_ms,
        "step_gpu_time_ms": train_forward_backward_time_ms + train_optimizer_step_time_ms,
        "transformer_block_time_ms": transformer_block_time_ms,
        "attention_time_ms": attention_time_ms,
        "attention_compute_time_ms": attention_compute_time_ms,
        "attention_comm_time_ms": attention_comm_time_ms,
        "attention_other_time_ms": attention_other_time_ms,
        "attention_path_total_forward_time_ms": attention_path_total_forward_time_ms,
        "ffn_time_ms": ffn_time_ms,
        "transformer_others_time_ms": transformer_others_time_ms,
        "embedding_time_ms": embedding_time_ms,
        "final_norm_time_ms": final_norm_time_ms,
        "output_layer_time_ms": output_layer_time_ms,
        "loss_time_ms": loss_time_ms,
        "lm_head_or_postprocess_time_ms": lm_head_or_postprocess_time_ms,
        "non_transformer_forward_time_ms": non_transformer_forward_time_ms,
        "total_backward_time_ms": total_backward_time_ms,
        "attention_backward_time_ms": attention_backward_time_ms,
        "attention_compute_backward_time_ms": attention_compute_backward_time_ms,
        "attention_comm_backward_time_ms": attention_comm_backward_time_ms,
        "attention_other_backward_time_ms": attention_other_backward_time_ms,
        "attention_path_total_backward_time_ms": attention_path_total_backward_time_ms,
        "ffn_backward_time_ms": ffn_backward_time_ms,
        "transformer_others_backward_time_ms": transformer_others_backward_time_ms,
        "input_layernorm_backward_time_ms": input_layernorm_backward_time_ms,
        "pre_mlp_layernorm_backward_time_ms": pre_mlp_layernorm_backward_time_ms,
        "self_attn_bda_backward_time_ms": self_attn_bda_backward_time_ms,
        "mlp_bda_backward_time_ms": mlp_bda_backward_time_ms,
        "embedding_backward_time_ms": embedding_backward_time_ms,
        "final_norm_backward_time_ms": final_norm_backward_time_ms,
        "output_layer_backward_time_ms": output_layer_backward_time_ms,
        "loss_backward_time_ms": loss_backward_time_ms,
        "non_transformer_backward_time_ms": non_transformer_backward_time_ms,
        "backward_unattributed_time_ms": backward_unattributed_time_ms,
        "transformer_block_backward_time_ms": (
            attention_backward_time_ms
            + ffn_backward_time_ms
            + transformer_others_backward_time_ms
        ),
        "attention_total_time_ms": attention_total_time_ms,
        "attention_compute_total_time_ms": attention_compute_total_time_ms,
        "attention_comm_total_time_ms": attention_comm_total_time_ms,
        "attention_other_total_time_ms": attention_other_total_time_ms,
        "attention_path_total_time_ms": attention_path_total_time_ms,
        "ffn_total_time_ms": ffn_total_time_ms,
        "transformer_others_total_time_ms": transformer_others_total_time_ms,
        "non_transformer_total_time_ms": non_transformer_total_time_ms,
        "unattributed_total_time_ms": unattributed_total_time_ms,
        "transformer_block_total_time_ms": (
            attention_total_time_ms + ffn_total_time_ms + transformer_others_total_time_ms
        ),
    }
    summary.update(step_union_accounting)

    attention_compute_forward_proj_ms = (
        attention_time_ms * float(summary["attention_forward_non_comm_only_ratio"])
    )
    attention_comm_related_forward_proj_ms = max(
        attention_time_ms - attention_compute_forward_proj_ms,
        0.0,
    )
    attention_compute_backward_proj_ms = (
        attention_backward_time_ms * float(summary["attention_backward_non_comm_only_ratio"])
    )
    attention_comm_related_backward_proj_ms = max(
        attention_backward_time_ms - attention_compute_backward_proj_ms,
        0.0,
    )
    attention_compute_total_proj_ms = (
        attention_compute_forward_proj_ms + attention_compute_backward_proj_ms
    )
    attention_comm_related_total_proj_ms = (
        attention_comm_related_forward_proj_ms + attention_comm_related_backward_proj_ms
    )
    summary.update(
        {
            "attention_compute_forward_proj_ms": attention_compute_forward_proj_ms,
            "attention_comm_related_forward_proj_ms": attention_comm_related_forward_proj_ms,
            "attention_compute_backward_proj_ms": attention_compute_backward_proj_ms,
            "attention_comm_related_backward_proj_ms": attention_comm_related_backward_proj_ms,
            "attention_compute_total_proj_ms": attention_compute_total_proj_ms,
            "attention_comm_related_total_proj_ms": attention_comm_related_total_proj_ms,
        }
    )

    add_ratio_fields(
        summary,
        "transformer_block_time_ms",
        [
            ("attention_time_ms", "attention_ratio"),
            ("attention_compute_time_ms", "attention_compute_ratio"),
            ("attention_comm_time_ms", "attention_comm_ratio"),
            ("attention_path_total_forward_time_ms", "attention_path_total_ratio"),
            ("ffn_time_ms", "ffn_ratio"),
            ("transformer_others_time_ms", "transformer_others_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_backward_time_ms",
        [
            ("attention_backward_time_ms", "attention_backward_ratio"),
            ("attention_compute_backward_time_ms", "attention_compute_backward_ratio"),
            ("attention_comm_backward_time_ms", "attention_comm_backward_ratio"),
            ("attention_path_total_backward_time_ms", "attention_path_total_backward_ratio"),
            ("ffn_backward_time_ms", "ffn_backward_ratio"),
            ("transformer_others_backward_time_ms", "transformer_others_backward_ratio"),
            ("backward_unattributed_time_ms", "backward_unattributed_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_total_time_ms",
        [
            ("attention_total_time_ms", "attention_total_ratio"),
            ("attention_compute_total_time_ms", "attention_compute_total_ratio"),
            ("attention_comm_total_time_ms", "attention_comm_total_ratio"),
            ("attention_path_total_time_ms", "attention_path_total_total_ratio"),
            ("ffn_total_time_ms", "ffn_total_ratio"),
            ("transformer_others_total_time_ms", "transformer_others_total_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "train_forward_backward_time_ms",
        [
            ("attention_total_time_ms", "attention_global_ratio"),
            ("attention_compute_total_time_ms", "attention_compute_global_ratio"),
            ("attention_comm_total_time_ms", "attention_comm_global_ratio"),
            ("attention_path_total_time_ms", "attention_path_total_global_ratio"),
            ("ffn_total_time_ms", "ffn_global_ratio"),
            ("transformer_others_total_time_ms", "transformer_others_global_ratio"),
            ("non_transformer_total_time_ms", "non_transformer_global_ratio"),
            ("backward_unattributed_time_ms", "backward_unattributed_global_ratio"),
            ("unattributed_total_time_ms", "unattributed_global_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_time_ms",
        [
            ("attention_compute_forward_proj_ms", "attention_compute_forward_proj_ratio"),
            ("attention_comm_related_forward_proj_ms", "attention_comm_related_forward_proj_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_backward_time_ms",
        [
            ("attention_compute_backward_proj_ms", "attention_compute_backward_proj_ratio"),
            ("attention_comm_related_backward_proj_ms", "attention_comm_related_backward_proj_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "transformer_block_total_time_ms",
        [
            ("attention_compute_total_proj_ms", "attention_compute_total_proj_ratio"),
            ("attention_comm_related_total_proj_ms", "attention_comm_related_total_proj_ratio"),
        ],
    )
    add_ratio_fields(
        summary,
        "train_forward_backward_time_ms",
        [
            ("attention_compute_total_proj_ms", "attention_compute_global_proj_ratio"),
            ("attention_comm_related_total_proj_ms", "attention_comm_related_global_proj_ratio"),
        ],
    )
    return summary


def draw_axes(
    draw: ImageDraw.ImageDraw,
    origin,
    width,
    height,
    *,
    x_ticks,
    y_ticks,
    tick_font,
):
    x0, y0 = origin
    draw.line((x0, y0, x0, y0 - height), fill="black", width=4)
    draw.line((x0, y0, x0 + width, y0), fill="black", width=4)
    for x, label in x_ticks:
        draw.line((x, y0, x, y0 + 14), fill="black", width=2)
        label_width, _ = measure_text(draw, label, tick_font)
        draw.text((x - label_width / 2, y0 + 20), label, fill="black", font=tick_font)
    for y, label in y_ticks:
        draw.line((x0 - 14, y, x0, y), fill="black", width=2)
        label_width, label_height = measure_text(draw, label, tick_font)
        draw.text((x0 - 24 - label_width, y - label_height / 2), label, fill="black", font=tick_font)


def compute_x_positions(rows, *, origin_x: float, width: float, use_actual_x_spacing: bool) -> list[float]:
    if not rows:
        return []
    if len(rows) == 1:
        return [origin_x + width / 2]
    if not use_actual_x_spacing:
        step = width / max(len(rows) - 1, 1)
        return [origin_x + step * idx for idx in range(len(rows))]

    seqs = [int(row["seq_length"]) for row in rows]
    min_seq = min(seqs)
    max_seq = max(seqs)
    if max_seq <= min_seq:
        return [origin_x + width / 2 for _ in rows]
    return [origin_x + width * ((seq - min_seq) / (max_seq - min_seq)) for seq in seqs]


def compute_bar_half_width(x_positions: list[float], idx: int, *, default_width: float = 24.0) -> float:
    if len(x_positions) <= 1:
        return default_width
    left_gap = x_positions[idx] - x_positions[idx - 1] if idx > 0 else x_positions[1] - x_positions[0]
    right_gap = (
        x_positions[idx + 1] - x_positions[idx]
        if idx < len(x_positions) - 1
        else x_positions[-1] - x_positions[-2]
    )
    gap = min(left_gap, right_gap)
    return max(10.0, min(36.0, gap * 0.18))


def save_stacked_bar_chart(
    rows,
    output_path: Path,
    *,
    title: str,
    total_key: str,
    part_keys: tuple[str, ...],
    legend_labels: tuple[str, ...],
    use_actual_x_spacing: bool = False,
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
    draw.text((40, 92), "Y axis: avg GPU ms per profiled step per rank", fill="black", font=subtitle_font)

    origin = CHART_ORIGIN
    width = CHART_PLOT_WIDTH
    height = CHART_PLOT_HEIGHT
    max_total = max(float(row[total_key]) for row in rows) or 1.0
    max_total *= 1.1

    x_positions = compute_x_positions(rows, origin_x=origin[0], width=width, use_actual_x_spacing=use_actual_x_spacing)
    x_ticks = []
    y_ticks = []
    for i in range(6):
        value = max_total * i / 5
        y = origin[1] - (value / max_total) * height
        y_ticks.append((y, f"{value:.0f}"))

    for idx, row in enumerate(rows):
        x_center = x_positions[idx]
        x_ticks.append((x_center, str(row["seq_length"])))
        half_width = compute_bar_half_width(x_positions, idx)
        bar_left = x_center - half_width
        bar_right = x_center + half_width
        y_cursor = origin[1]
        for key in part_keys:
            value = float(row[key])
            bar_height = (value / max_total) * height
            color = PART_COLORS[color_key_for_metric(key)]
            draw.rectangle(
                (bar_left, y_cursor - bar_height, bar_right, y_cursor),
                fill=color,
                outline="black",
                width=2,
            )
            y_cursor -= bar_height
        value_text = f"{float(row[total_key]):.1f}"
        value_width, value_height = measure_text(draw, value_text, value_font)
        draw.text((x_center - value_width / 2, y_cursor - value_height - 10), value_text, fill="black", font=value_font)

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, tick_font=tick_font)
    for idx, (key, label) in enumerate(zip(part_keys, legend_labels)):
        color = PART_COLORS[color_key_for_metric(key)]
        legend_y = 150 + idx * 36
        draw.rectangle((1620, legend_y + 4, 1650, legend_y + 28), fill=color, outline="black", width=1)
        draw.text((1665, legend_y), label, fill="black", font=legend_font)
    save_chart(image, output_path)


def save_ratio_line_chart(
    rows,
    output_path: Path,
    *,
    title: str,
    ratio_keys: tuple[str, ...],
    legend_labels: tuple[str, ...],
    use_actual_x_spacing: bool = False,
):
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(46)
    tick_font = load_font(22)
    legend_font = load_font(24)
    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((CHART_WIDTH - title_width) / 2, 28), title, fill="black", font=title_font)

    origin = CHART_ORIGIN
    width = CHART_PLOT_WIDTH
    height = CHART_PLOT_HEIGHT
    x_positions = compute_x_positions(rows, origin_x=origin[0], width=width, use_actual_x_spacing=use_actual_x_spacing)
    x_ticks = []
    y_ticks = []
    for i in range(6):
        value = i / 5
        y = origin[1] - value * height
        y_ticks.append((y, f"{value * 100:.0f}%"))

    draw_axes(draw, origin, width, height, x_ticks=[], y_ticks=y_ticks, tick_font=tick_font)

    for idx, row in enumerate(rows):
        x = x_positions[idx]
        x_ticks.append((x, str(row["seq_length"])))
    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, tick_font=tick_font)

    label_offsets = (-30, -10, 12, 32, 52)
    for idx, key in enumerate(ratio_keys):
        color = PART_COLORS[color_key_for_metric(key)]
        points = []
        for row_idx, row in enumerate(rows):
            x = x_positions[row_idx]
            y = origin[1] - float(row[key]) * height
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=5)
        for point, row in zip(points, rows):
            draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=color)
            value_text = f"{float(row[key]) * 100:.1f}%"
            value_width, value_height = measure_text(draw, value_text, tick_font)
            text_y = point[1] + label_offsets[min(idx, len(label_offsets) - 1)] - value_height / 2
            draw.text((point[0] - value_width / 2, text_y), value_text, fill=color, font=tick_font)

    for idx, (key, label) in enumerate(zip(ratio_keys, legend_labels)):
        color = PART_COLORS[color_key_for_metric(key)]
        legend_y = 150 + idx * 36
        draw.line((1605, legend_y + 16, 1645, legend_y + 16), fill=color, width=5)
        draw.ellipse((1620 - 7, legend_y + 9, 1620 + 7, legend_y + 23), fill=color)
        draw.text((1665, legend_y), label, fill="black", font=legend_font)
    save_chart(image, output_path)


def save_step_time_chart(rows, output_path: Path, *, title: str, use_actual_x_spacing: bool = False):
    image = Image.new("RGB", (CHART_WIDTH, CHART_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(46)
    tick_font = load_font(22)
    value_font = load_font(18)
    title_width, _ = measure_text(draw, title, title_font)
    draw.text(((CHART_WIDTH - title_width) / 2, 28), title, fill="black", font=title_font)

    origin = CHART_ORIGIN
    width = CHART_PLOT_WIDTH
    height = CHART_PLOT_HEIGHT
    max_time = max(float(row["step_gpu_time_ms"]) for row in rows) or 1.0
    max_time *= 1.1
    x_positions = compute_x_positions(rows, origin_x=origin[0], width=width, use_actual_x_spacing=use_actual_x_spacing)
    x_ticks = []
    y_ticks = []
    for i in range(6):
        value = max_time * i / 5
        y = origin[1] - (value / max_time) * height
        y_ticks.append((y, f"{value:.0f}"))

    draw_axes(draw, origin, width, height, x_ticks=[], y_ticks=y_ticks, tick_font=tick_font)

    points = []
    for idx, row in enumerate(rows):
        x = x_positions[idx]
        y = origin[1] - (float(row["step_gpu_time_ms"]) / max_time) * height
        x_ticks.append((x, str(row["seq_length"])))
        points.append((x, y))

    draw_axes(draw, origin, width, height, x_ticks=x_ticks, y_ticks=y_ticks, tick_font=tick_font)
    if len(points) >= 2:
        draw.line(points, fill="#4c78a8", width=5)
    for point, row in zip(points, rows):
        draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill="#4c78a8")
        value_text = f"{float(row['step_gpu_time_ms']):.0f}"
        value_width, value_height = measure_text(draw, value_text, value_font)
        draw.text((point[0] - value_width / 2, point[1] - value_height - 12), value_text, fill="black", font=value_font)

    save_chart(image, output_path)


def empty_summary() -> dict[str, float]:
    return {
        "step_instances": 0.0,
        "step_union_instances": 0.0,
        "train_forward_backward_time_ms": 0.0,
        "train_optimizer_step_time_ms": 0.0,
        "step_gpu_time_ms": 0.0,
        "step_gpu_active_union_ms": 0.0,
        "transformer_block_time_ms": 0.0,
        "attention_time_ms": 0.0,
        "attention_compute_time_ms": 0.0,
        "attention_comm_time_ms": 0.0,
        "attention_forward_non_comm_union_ms": 0.0,
        "attention_forward_comm_union_ms": 0.0,
        "attention_forward_path_union_ms": 0.0,
        "attention_forward_overlap_ms": 0.0,
        "attention_forward_non_comm_only_ms": 0.0,
        "attention_forward_comm_only_ms": 0.0,
        "attention_backward_non_comm_union_ms": 0.0,
        "attention_backward_comm_union_ms": 0.0,
        "attention_backward_path_union_ms": 0.0,
        "attention_backward_overlap_ms": 0.0,
        "attention_backward_non_comm_only_ms": 0.0,
        "attention_backward_comm_only_ms": 0.0,
        "attention_non_comm_union_ms": 0.0,
        "attention_comm_union_ms": 0.0,
        "attention_path_union_ms": 0.0,
        "attention_overlap_ms": 0.0,
        "attention_non_comm_only_ms": 0.0,
        "attention_comm_only_ms": 0.0,
        "non_attention_only_ms": 0.0,
        "attention_other_time_ms": 0.0,
        "attention_path_total_forward_time_ms": 0.0,
        "ffn_time_ms": 0.0,
        "transformer_others_time_ms": 0.0,
        "embedding_time_ms": 0.0,
        "final_norm_time_ms": 0.0,
        "output_layer_time_ms": 0.0,
        "loss_time_ms": 0.0,
        "lm_head_or_postprocess_time_ms": 0.0,
        "non_transformer_forward_time_ms": 0.0,
        "total_backward_time_ms": 0.0,
        "attention_backward_time_ms": 0.0,
        "attention_compute_backward_time_ms": 0.0,
        "attention_comm_backward_time_ms": 0.0,
        "attention_other_backward_time_ms": 0.0,
        "attention_path_total_backward_time_ms": 0.0,
        "ffn_backward_time_ms": 0.0,
        "transformer_others_backward_time_ms": 0.0,
        "input_layernorm_backward_time_ms": 0.0,
        "pre_mlp_layernorm_backward_time_ms": 0.0,
        "self_attn_bda_backward_time_ms": 0.0,
        "mlp_bda_backward_time_ms": 0.0,
        "embedding_backward_time_ms": 0.0,
        "final_norm_backward_time_ms": 0.0,
        "output_layer_backward_time_ms": 0.0,
        "loss_backward_time_ms": 0.0,
        "non_transformer_backward_time_ms": 0.0,
        "backward_unattributed_time_ms": 0.0,
        "transformer_block_backward_time_ms": 0.0,
        "attention_total_time_ms": 0.0,
        "attention_compute_total_time_ms": 0.0,
        "attention_comm_total_time_ms": 0.0,
        "attention_other_total_time_ms": 0.0,
        "attention_path_total_time_ms": 0.0,
        "ffn_total_time_ms": 0.0,
        "transformer_others_total_time_ms": 0.0,
        "non_transformer_total_time_ms": 0.0,
        "unattributed_total_time_ms": 0.0,
        "transformer_block_total_time_ms": 0.0,
        "attention_ratio": 0.0,
        "attention_compute_ratio": 0.0,
        "attention_comm_ratio": 0.0,
        "attention_path_total_ratio": 0.0,
        "ffn_ratio": 0.0,
        "transformer_others_ratio": 0.0,
        "attention_backward_ratio": 0.0,
        "attention_compute_backward_ratio": 0.0,
        "attention_comm_backward_ratio": 0.0,
        "attention_path_total_backward_ratio": 0.0,
        "ffn_backward_ratio": 0.0,
        "transformer_others_backward_ratio": 0.0,
        "backward_unattributed_ratio": 0.0,
        "attention_total_ratio": 0.0,
        "attention_compute_total_ratio": 0.0,
        "attention_comm_total_ratio": 0.0,
        "attention_path_total_total_ratio": 0.0,
        "ffn_total_ratio": 0.0,
        "transformer_others_total_ratio": 0.0,
        "attention_global_ratio": 0.0,
        "attention_compute_global_ratio": 0.0,
        "attention_comm_global_ratio": 0.0,
        "attention_path_total_global_ratio": 0.0,
        "attention_compute_forward_proj_ms": 0.0,
        "attention_comm_related_forward_proj_ms": 0.0,
        "attention_compute_backward_proj_ms": 0.0,
        "attention_comm_related_backward_proj_ms": 0.0,
        "attention_compute_total_proj_ms": 0.0,
        "attention_comm_related_total_proj_ms": 0.0,
        "attention_compute_forward_proj_ratio": 0.0,
        "attention_comm_related_forward_proj_ratio": 0.0,
        "attention_compute_backward_proj_ratio": 0.0,
        "attention_comm_related_backward_proj_ratio": 0.0,
        "attention_compute_total_proj_ratio": 0.0,
        "attention_comm_related_total_proj_ratio": 0.0,
        "attention_compute_global_proj_ratio": 0.0,
        "attention_comm_related_global_proj_ratio": 0.0,
        "attention_forward_non_comm_only_ratio": 0.0,
        "attention_forward_comm_only_ratio": 0.0,
        "attention_forward_overlap_ratio": 0.0,
        "attention_backward_non_comm_only_ratio": 0.0,
        "attention_backward_comm_only_ratio": 0.0,
        "attention_backward_overlap_ratio": 0.0,
        "attention_non_comm_only_total_ratio": 0.0,
        "attention_comm_only_total_ratio": 0.0,
        "attention_overlap_total_ratio": 0.0,
        "attention_non_comm_inclusive_ratio": 0.0,
        "attention_comm_inclusive_ratio": 0.0,
        "attention_non_comm_only_ratio": 0.0,
        "attention_comm_only_ratio": 0.0,
        "attention_overlap_ratio": 0.0,
        "attention_path_ratio": 0.0,
        "non_attention_only_ratio": 0.0,
        "ffn_global_ratio": 0.0,
        "transformer_others_global_ratio": 0.0,
        "non_transformer_global_ratio": 0.0,
        "backward_unattributed_global_ratio": 0.0,
        "unattributed_global_ratio": 0.0,
    }


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chart_filename(output_prefix: str, prefix: str, stem: str) -> str:
    return f"{output_prefix}_{prefix}_{stem}.png"


def chart_output_path(
    output_root: Path,
    *,
    output_prefix: str,
    prefix: str,
    stem: str,
    attention_chart: bool,
) -> Path:
    if attention_chart:
        subdir = output_root / "attention"
    else:
        subdir_name = {"all": "all", "exp_line": "exp", "linear_line": "linear"}[prefix]
        subdir = output_root / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / chart_filename(output_prefix, prefix, stem)


def emit_chart_bundle(
    rows,
    output_root: Path,
    *,
    output_prefix: str,
    prefix: str,
    label: str,
    attention_mode: str,
):
    if not rows:
        return

    use_actual_x_spacing = prefix == "all"

    if attention_mode == "cp":
        forward_part_keys = (
            "attention_compute_forward_proj_ms",
            "attention_comm_related_forward_proj_ms",
            "ffn_time_ms",
            "transformer_others_time_ms",
        )
        forward_labels = ("Attention Compute", "Attention Comm/Overlap", "FFN", "Transformer Others")
        forward_ratio_keys = (
            "attention_compute_forward_proj_ratio",
            "attention_comm_related_forward_proj_ratio",
            "ffn_ratio",
            "transformer_others_ratio",
        )
        backward_part_keys = (
            "attention_compute_backward_proj_ms",
            "attention_comm_related_backward_proj_ms",
            "ffn_backward_time_ms",
            "transformer_others_backward_time_ms",
        )
        backward_labels = ("Attention Compute", "Attention Comm/Overlap", "FFN", "Transformer Others")
        backward_ratio_keys = (
            "attention_compute_backward_proj_ratio",
            "attention_comm_related_backward_proj_ratio",
            "ffn_backward_ratio",
            "transformer_others_backward_ratio",
        )
        total_part_keys = (
            "attention_compute_total_proj_ms",
            "attention_comm_related_total_proj_ms",
            "ffn_total_time_ms",
            "transformer_others_total_time_ms",
        )
        total_labels = ("Attention Compute", "Attention Comm/Overlap", "FFN", "Transformer Others")
        total_ratio_keys = (
            "attention_compute_total_proj_ratio",
            "attention_comm_related_total_proj_ratio",
            "ffn_total_ratio",
            "transformer_others_total_ratio",
        )
        global_part_keys = (
            "attention_compute_total_proj_ms",
            "attention_comm_related_total_proj_ms",
            "ffn_total_time_ms",
            "transformer_others_total_time_ms",
            "non_transformer_total_time_ms",
            "unattributed_total_time_ms",
        )
        global_labels = (
            "Attention Compute",
            "Attention Comm/Overlap",
            "FFN",
            "Transformer Others",
            "Non-Transformer",
            "Unattributed",
        )
        global_ratio_keys = (
            "attention_compute_global_proj_ratio",
            "attention_comm_related_global_proj_ratio",
            "ffn_global_ratio",
            "transformer_others_global_ratio",
            "non_transformer_global_ratio",
            "unattributed_global_ratio",
        )
    else:
        forward_part_keys = ("attention_time_ms", "ffn_time_ms", "transformer_others_time_ms")
        forward_labels = ("Attention", "FFN", "Transformer Others")
        forward_ratio_keys = ("attention_ratio", "ffn_ratio", "transformer_others_ratio")
        backward_part_keys = (
            "attention_backward_time_ms",
            "ffn_backward_time_ms",
            "transformer_others_backward_time_ms",
            "backward_unattributed_time_ms",
        )
        backward_labels = ("Attention", "FFN", "Transformer Others", "Backward Unattributed")
        backward_ratio_keys = (
            "attention_backward_ratio",
            "ffn_backward_ratio",
            "transformer_others_backward_ratio",
            "backward_unattributed_ratio",
        )
        total_part_keys = (
            "attention_total_time_ms",
            "ffn_total_time_ms",
            "transformer_others_total_time_ms",
        )
        total_labels = ("Attention", "FFN", "Transformer Others")
        total_ratio_keys = ("attention_total_ratio", "ffn_total_ratio", "transformer_others_total_ratio")
        global_part_keys = (
            "attention_total_time_ms",
            "ffn_total_time_ms",
            "transformer_others_total_time_ms",
            "non_transformer_total_time_ms",
            "backward_unattributed_time_ms",
            "unattributed_total_time_ms",
        )
        global_labels = (
            "Attention",
            "FFN",
            "Transformer Others",
            "Non-Transformer",
            "Backward Unattributed",
            "Unattributed",
        )
        global_ratio_keys = (
            "attention_global_ratio",
            "ffn_global_ratio",
            "transformer_others_global_ratio",
            "non_transformer_global_ratio",
            "backward_unattributed_global_ratio",
            "unattributed_global_ratio",
        )

    save_stacked_bar_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="forward_stacked",
            attention_chart=False,
        ),
        title="Forward GPU Breakdown",
        total_key="transformer_block_time_ms",
        part_keys=forward_part_keys,
        legend_labels=forward_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_ratio_line_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="forward_ratios",
            attention_chart=False,
        ),
        title="Forward GPU Ratios",
        ratio_keys=forward_ratio_keys,
        legend_labels=forward_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_stacked_bar_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="backward_stacked",
            attention_chart=False,
        ),
        title="Backward GPU Breakdown",
        total_key="transformer_block_backward_time_ms",
        part_keys=backward_part_keys,
        legend_labels=backward_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_ratio_line_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="backward_ratios",
            attention_chart=False,
        ),
        title="Backward GPU Ratios",
        ratio_keys=backward_ratio_keys,
        legend_labels=backward_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_stacked_bar_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="total_stacked",
            attention_chart=False,
        ),
        title="Total GPU Breakdown",
        total_key="transformer_block_total_time_ms",
        part_keys=total_part_keys,
        legend_labels=total_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_ratio_line_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="total_ratios",
            attention_chart=False,
        ),
        title="Total GPU Ratios",
        ratio_keys=total_ratio_keys,
        legend_labels=total_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_stacked_bar_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="global_stacked",
            attention_chart=False,
        ),
        title="Global GPU Breakdown",
        total_key="train_forward_backward_time_ms",
        part_keys=global_part_keys,
        legend_labels=global_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_ratio_line_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="global_ratios",
            attention_chart=False,
        ),
        title="Global GPU Ratios",
        ratio_keys=global_ratio_keys,
        legend_labels=global_labels,
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_step_time_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="step_time",
            attention_chart=False,
        ),
        title="Avg GPU Step Time",
        use_actual_x_spacing=use_actual_x_spacing,
    )


def emit_attention_accounting_bundle(
    rows,
    output_root: Path,
    *,
    output_prefix: str,
    prefix: str,
    attention_mode: str,
):
    if not rows:
        return
    if attention_mode != "cp":
        return
    use_actual_x_spacing = prefix == "all"

    save_stacked_bar_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="attention_path_stacked",
            attention_chart=True,
        ),
        title="Attention Path Breakdown",
        total_key="attention_path_union_ms",
        part_keys=(
            "attention_non_comm_only_ms",
            "attention_comm_only_ms",
            "attention_overlap_ms",
        ),
        legend_labels=("Non-Comm Only", "Comm Only", "Overlap"),
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_ratio_line_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="attention_path_ratios",
            attention_chart=True,
        ),
        title="Attention Path Ratios",
        ratio_keys=(
            "attention_non_comm_only_total_ratio",
            "attention_comm_only_total_ratio",
            "attention_overlap_total_ratio",
        ),
        legend_labels=("Non-Comm Only", "Comm Only", "Overlap"),
        use_actual_x_spacing=use_actual_x_spacing,
    )
    save_ratio_line_chart(
        rows,
        chart_output_path(
            output_root,
            output_prefix=output_prefix,
            prefix=prefix,
            stem="attention_global_ratios",
            attention_chart=True,
        ),
        title="Attention Active Ratios",
        ratio_keys=(
            "attention_non_comm_inclusive_ratio",
            "attention_comm_only_ratio",
            "attention_overlap_ratio",
            "attention_path_ratio",
        ),
        legend_labels=("Non-Comm", "Comm Only", "Overlap", "Path Total"),
        use_actual_x_spacing=use_actual_x_spacing,
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
            "in_exp_line": int(seq in EXP_LINE_SEQ_LENGTHS),
            "in_linear_line": int(seq in LINEAR_LINE_SEQ_LENGTHS),
            "is_boundary_probe": int(seq in BOUNDARY_SEQ_LENGTHS),
            "selected_gpus": metadata.get("selected_gpus", ""),
            "tp_size": metadata.get("tp_size", ""),
            "pp_size": metadata.get("pp_size", ""),
            "cp_size": metadata.get("cp_size", ""),
            "sequence_parallel": metadata.get("sequence_parallel", ""),
            "model_variant": metadata.get("model_variant", ""),
            "num_layers": metadata.get("num_layers", ""),
        }

        if status != "success" or not report_path.exists():
            row.update(empty_summary())
            rows.append(row)
            continue

        row.update(summarize_report(report_path))
        rows.append(row)

    csv_path = args.output_root / f"{args.output_prefix}_profile_summary.csv"
    write_summary_csv(csv_path, rows)

    boundary_rows = [row for row in rows if int(row["is_boundary_probe"]) == 1]
    if boundary_rows:
        write_summary_csv(
            args.output_root / f"{args.output_prefix}_boundary_summary.csv",
            boundary_rows,
        )

    successful_rows = [row for row in rows if row["status"] == "success"]
    if not successful_rows:
        return

    if args.attention_mode == "auto":
        attention_mode = "cp" if any(int(row.get("cp_size") or 0) > 1 for row in successful_rows) else "non_cp"
    else:
        attention_mode = args.attention_mode

    emit_chart_bundle(
        successful_rows,
        args.output_root,
        output_prefix=args.output_prefix,
        prefix="all",
        label=f"{args.label} All Successful Points",
        attention_mode=attention_mode,
    )
    emit_attention_accounting_bundle(
        successful_rows,
        args.output_root,
        output_prefix=args.output_prefix,
        prefix="all",
        attention_mode=attention_mode,
    )

    exp_rows = [row for row in successful_rows if int(row["in_exp_line"]) == 1]
    emit_chart_bundle(
        exp_rows,
        args.output_root,
        output_prefix=args.output_prefix,
        prefix="exp_line",
        label=f"{args.label} Exp Line",
        attention_mode=attention_mode,
    )
    emit_attention_accounting_bundle(
        exp_rows,
        args.output_root,
        output_prefix=args.output_prefix,
        prefix="exp_line",
        attention_mode=attention_mode,
    )

    linear_rows = [row for row in successful_rows if int(row["in_linear_line"]) == 1]
    emit_chart_bundle(
        linear_rows,
        args.output_root,
        output_prefix=args.output_prefix,
        prefix="linear_line",
        label=f"{args.label} Linear Line",
        attention_mode=attention_mode,
    )
    emit_attention_accounting_bundle(
        linear_rows,
        args.output_root,
        output_prefix=args.output_prefix,
        prefix="linear_line",
        attention_mode=attention_mode,
    )


if __name__ == "__main__":
    main()
