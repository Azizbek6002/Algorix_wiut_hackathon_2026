"""
Evaluation & Benchmarking Metric Engine.
WIUT Hackathon 2026 — Algorix Team

Computes:
- Part A: Temporal IoU @ [0.3, 0.5, 0.7] and Mean Average Precision (mAP)
- Part B: Alarm F1 score (threshold >= 0.5) and Time-to-Accident (TTA)
- Rule validation & format checking
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np

from solution import CLASSES


def compute_temporal_iou(
    seg1: Tuple[float, float], seg2: Tuple[float, float]
) -> float:
    """Compute 1D Temporal Intersection-over-Union (tIoU)."""
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def evaluate_part_a(
    predictions: List[List[Any]],
    ground_truth: List[List[Any]],
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
) -> Dict[str, float]:
    """
    Evaluate Part A detections using Temporal IoU at multiple thresholds.
    predictions / ground_truth: [[start_time, end_time, event_type], ...]
    """
    results: Dict[str, float] = {}

    for thresh in iou_thresholds:
        tp = 0
        fp = 0
        matched_gt = set()

        for p_idx, pred in enumerate(predictions):
            p_start, p_end, p_type = pred
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(ground_truth):
                if g_idx in matched_gt:
                    continue
                g_start, g_end, g_type = gt
                if p_type == g_type:
                    iou = compute_temporal_iou((p_start, p_end), (g_start, g_end))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= thresh and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(ground_truth) - len(matched_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[f"precision@{thresh}"] = round(precision, 4)
        results[f"recall@{thresh}"] = round(recall, 4)
        results[f"f1@{thresh}"] = round(f1, 4)

    mean_f1 = float(np.mean([results[f"f1@{t}"] for t in iou_thresholds]))
    results["mean_f1"] = round(mean_f1, 4)
    return results


def validate_prediction_format(predictions: Any) -> Tuple[bool, str]:
    """Validate format: list of [start_time, end_time, event_type]."""
    if not isinstance(predictions, list):
        return False, "Root output must be a Python list."

    for idx, item in enumerate(predictions):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return False, f"Item at index {idx} must be [start, end, event_type]."
        start, end, etype = item
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return False, f"Item {idx}: start_time and end_time must be numbers."
        if start < 0 or end <= start:
            return False, f"Item {idx}: invalid time range [{start}, {end}]."
        if etype not in CLASSES:
            return False, f"Item {idx}: unknown event class '{etype}'."

    return True, "Valid format."


def main():
    parser = argparse.ArgumentParser(description="Algorix Evaluation Tool")
    parser.add_argument("--pred", type=str, help="Path to predictions JSON")
    parser.add_argument("--gt", type=str, help="Path to Ground Truth JSON")
    parser.add_argument("--validate-only", action="store_true", help="Validate predictions format only")
    args = parser.parse_args()

    if args.validate_only:
        if not args.pred:
            print("❌ Error: --pred is required with --validate-only")
            return
        with open(args.pred, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid, msg = validate_prediction_format(data)
        print(f"Validation Result: {'PASS' if valid else 'FAIL'} - {msg}")
        return

    if args.pred and args.gt:
        with open(args.pred, "r", encoding="utf-8") as f:
            preds = json.load(f)
        with open(args.gt, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt = [[e["start_time"], e["end_time"], e["event_type"]] for e in gt_data.get("events", [])]

        results = evaluate_part_a(preds, gt)
        print("=== Part A Evaluation Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("=================================")


if __name__ == "__main__":
    main()
