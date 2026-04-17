#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_base_module():
    base_path = Path(__file__).with_name("analyze_qwen3_0p6b_seq_nsys.py")
    spec = importlib.util.spec_from_file_location("base_qwen0p6b_pp4_nsys", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base analyzer from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_base_module()
    original_parse_args = module.parse_args
    output_root_holder = {}

    def parse_args():
        args = original_parse_args()
        if args.input_root == Path("runs/qwen0p6b_seq_study_pp4_nsys_0p6b_base"):
            args.input_root = Path("runs/qwen0p6b_seq_study_pp2_cp2_nsys_2layers")
        if args.output_root == Path("runs/qwen0p6b_seq_study_pp4_nsys_0p6b_base/analysis"):
            args.output_root = Path("runs/qwen0p6b_seq_study_pp2_cp2_nsys_2layers/analysis")
        if args.label == "Qwen3-0.6B PP4":
            args.label = "Qwen3-0.6B 2-layer PP2+CP2"
        output_root_holder["path"] = args.output_root
        return args

    module.parse_args = parse_args
    module.main()

    output_root = output_root_holder.get("path")
    if output_root is not None and output_root.exists():
        for path in output_root.iterdir():
            if "qwen3_0p6b_pp4_seq_" not in path.name:
                continue
            target = path.with_name(path.name.replace("qwen3_0p6b_pp4_seq_", "qwen3_0p6b_pp2_cp2_seq_"))
            path.rename(target)


if __name__ == "__main__":
    main()
