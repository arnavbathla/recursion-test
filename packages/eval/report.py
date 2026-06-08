"""Generate a human-readable Markdown eval report from eval/ablation JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from packages.common.storage import read_json, write_text


def _depth_table(depths: dict[str, dict]) -> str:
    header = (
        "| recursion_depth | exact_match | pixel_accuracy | shape_accuracy | mean_latency_ms | p50_ms | p95_ms | num_tasks |\n"
        "| --------------- | ----------- | -------------- | -------------- | --------------- | ------ | ------ | --------- |\n"
    )
    rows = []
    for d in sorted(depths, key=lambda x: int(x)):
        m = depths[d]
        rows.append(
            f"| {d} | {m['exact_match']:.3f} | {m['pixel_accuracy']:.3f} | "
            f"{m['shape_accuracy']:.3f} | {m['mean_latency_ms']:.1f} | "
            f"{m['p50_latency_ms']:.1f} | {m['p95_latency_ms']:.1f} | {m['num_tasks']} |"
        )
    return header + "\n".join(rows)


def _ablation_table(ablations: list[dict]) -> str:
    header = (
        "| model | recursion_depth | exact_match | pixel_accuracy | notes |\n"
        "| ----- | --------------- | ----------- | -------------- | ----- |\n"
    )
    rows = []
    for a in ablations:
        rows.append(
            f"| {a['model']} | {a['recursion_depth']} | {a['exact_match']:.3f} | "
            f"{a['pixel_accuracy']:.3f} | {a.get('notes','')} |"
        )
    return header + "\n".join(rows)


def generate_report(
    eval_json: dict[str, Any],
    out_path: str | Path,
    *,
    param_count: int | None = None,
    ablation_json: dict[str, Any] | None = None,
    baseline_eval_json: dict[str, Any] | None = None,
) -> Path:
    depths = eval_json.get("depths", {})
    exacts = [m["exact_match"] for m in depths.values()] or [0.0]
    best_exact = max(exacts)

    lines: list[str] = []
    lines.append(f"# Eval Report: {eval_json.get('model_id', 'model')}\n")
    lines.append("## Provenance\n")
    lines.append(f"- dataset_hash: `{eval_json.get('dataset_hash')}`")
    lines.append(f"- checkpoint_hash: `{eval_json.get('checkpoint_hash')}`")
    lines.append(f"- config_hash: `{eval_json.get('config_hash')}`")
    lines.append(f"- model parameter count: {param_count if param_count is not None else 'n/a'}")
    lines.append(f"- eval split: `{eval_json.get('split')}`")
    lines.append(f"- device: `{eval_json.get('device')}`\n")

    lines.append("## Accuracy / latency vs recursion depth\n")
    lines.append(_depth_table(depths) + "\n")

    if baseline_eval_json is not None:
        lines.append("## Baseline comparison (non-recursive)\n")
        lines.append(_depth_table(baseline_eval_json.get("depths", {})) + "\n")

    if ablation_json is not None:
        lines.append("## Ablations\n")
        lines.append(_ablation_table(ablation_json.get("ablations", [])) + "\n")

    lines.append("## Latency\n")
    lines.append(
        "Latency grows with recursion depth (each step reuses the same weights). "
        "See the mean/p50/p95 columns in the depth table above.\n"
    )

    lines.append("## Failure categories\n")
    lines.append(
        "- shape mismatch (predicted height/width != target) -> pixel_accuracy 0.0\n"
        "- correct shape but wrong cell colors -> partial pixel_accuracy, exact_match false\n"
        "- both correct -> exact_match true\n"
    )

    lines.append("## Claims supported by this run\n")
    lines.append(
        f"- The system runs real recursive inference at multiple depths and measures objective, "
        f"pixel-exact metrics. Best observed exact_match across depths: {best_exact:.3f}.\n"
    )
    lines.append("## Claims NOT supported\n")
    lines.append(
        "- This run does not claim SOTA, AGI, RSI, or solving all ARC tasks. Exact-match quality "
        "is bounded by the training budget used to produce this checkpoint.\n"
    )
    text = "\n".join(lines)
    return write_text(out_path, text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Markdown eval report")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--param-count", type=int, default=None)
    parser.add_argument("--ablation-json", default=None)
    parser.add_argument("--baseline-eval-json", default=None)
    args = parser.parse_args()

    eval_json = read_json(args.eval_json)
    ablation_json = read_json(args.ablation_json) if args.ablation_json else None
    baseline_json = read_json(args.baseline_eval_json) if args.baseline_eval_json else None
    out = generate_report(
        eval_json, args.out, param_count=args.param_count,
        ablation_json=ablation_json, baseline_eval_json=baseline_json,
    )
    print(f"OK: wrote report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
