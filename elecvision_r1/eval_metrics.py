from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .reward import bbox_iou, normalize_box_record, parse_grounding_answer


def _read_json_or_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def _xywh_to_xyxy(box: Sequence[Any]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _records_from_coco(payload: Mapping[str, Any], *, prediction: bool) -> list[dict[str, Any]]:
    records = []
    for ann in payload.get("annotations", []):
        box = ann.get("bbox")
        if not box or len(box) != 4:
            continue
        records.append(
            {
                "image_id": str(ann.get("image_id")),
                "label": str(ann.get("category_id", ann.get("label"))),
                "bbox_2d": _xywh_to_xyxy(box),
                "score": float(ann.get("score", 1.0 if prediction else 0.0)),
            }
        )
    return records


def _records_from_rows(rows: Iterable[Any], *, prediction: bool) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        if isinstance(row, Mapping) and "annotations" in row and "images" in row:
            records.extend(_records_from_coco(row, prediction=prediction))
            continue
        if isinstance(row, Mapping) and {"image_id", "bbox"}.issubset(row.keys()):
            normalized = normalize_box_record(
                {
                    "label": row.get("category_id", row.get("label", "unknown")),
                    "bbox_2d": _xywh_to_xyxy(row["bbox"]),
                }
            )
            if normalized is not None:
                records.append(
                    {
                        "image_id": str(row["image_id"]),
                        "label": normalized["label"],
                        "bbox_2d": normalized["bbox_2d"],
                        "score": float(row.get("score", 1.0 if prediction else 0.0)),
                    }
                )
            continue
        image_id = str(row.get("image_id", row.get("id", idx))) if isinstance(row, Mapping) else str(idx)
        objects = []
        if isinstance(row, Mapping):
            objects = row.get("objects", row.get("detections", row.get("prediction", row.get("solution", []))))
        if isinstance(objects, str):
            objects = parse_grounding_answer(objects)
        for obj in objects if isinstance(objects, list) else []:
            normalized = normalize_box_record(obj)
            if normalized is None:
                continue
            records.append(
                {
                    "image_id": image_id,
                    "label": normalized["label"],
                    "bbox_2d": normalized["bbox_2d"],
                    "score": float(obj.get("score", 1.0 if prediction else 0.0)) if isinstance(obj, Mapping) else 1.0,
                }
            )
    return records


def load_detection_records(path: str | Path, *, prediction: bool) -> list[dict[str, Any]]:
    rows = _read_json_or_jsonl(path)
    if len(rows) == 1 and isinstance(rows[0], Mapping) and "annotations" in rows[0] and "images" in rows[0]:
        return _records_from_coco(rows[0], prediction=prediction)
    return _records_from_rows(rows, prediction=prediction)


def average_precision_at_iou(
    predictions: Sequence[Mapping[str, Any]],
    ground_truths: Sequence[Mapping[str, Any]],
    iou_threshold: float,
) -> float:
    gt_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for gt in ground_truths:
        gt_by_key[(str(gt["image_id"]), str(gt["label"]))].append(gt)
    n_gt = len(ground_truths)
    if n_gt == 0:
        return 0.0

    preds = sorted(
        predictions,
        key=lambda item: (-float(item.get("score", 1.0)), str(item.get("image_id")), str(item.get("label"))),
    )
    matched: set[tuple[str, str, int]] = set()
    tp = []
    fp = []
    for pred in preds:
        key = (str(pred["image_id"]), str(pred["label"]))
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt in enumerate(gt_by_key.get(key, [])):
            match_key = (key[0], key[1], gt_idx)
            if match_key in matched:
                continue
            iou = bbox_iou(pred["bbox_2d"], gt["bbox_2d"])
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched.add((key[0], key[1], best_idx))
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0
    cum_tp = []
    cum_fp = []
    t_total = 0.0
    f_total = 0.0
    for t, f in zip(tp, fp):
        t_total += t
        f_total += f
        cum_tp.append(t_total)
        cum_fp.append(f_total)

    precisions = [t / max(t + f, 1e-12) for t, f in zip(cum_tp, cum_fp)]
    recalls = [t / n_gt for t in cum_tp]
    ap = 0.0
    for recall_threshold in [i / 100 for i in range(101)]:
        candidates = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
        ap += max(candidates) if candidates else 0.0
    return ap / 101


def coco_style_map(
    predictions: Sequence[Mapping[str, Any]],
    ground_truths: Sequence[Mapping[str, Any]],
    iou_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    thresholds = list(iou_thresholds or [0.50 + 0.05 * i for i in range(10)])
    labels = sorted({str(gt["label"]) for gt in ground_truths})
    per_threshold = {}
    per_label = {label: [] for label in labels}
    for threshold in thresholds:
        aps = []
        for label in labels:
            label_preds = [p for p in predictions if str(p["label"]) == label]
            label_gts = [g for g in ground_truths if str(g["label"]) == label]
            ap = average_precision_at_iou(label_preds, label_gts, threshold)
            aps.append(ap)
            per_label[label].append(ap)
        per_threshold[f"{threshold:.2f}"] = sum(aps) / len(aps) if aps else 0.0
    label_ap = {label: sum(values) / len(values) if values else 0.0 for label, values in per_label.items()}
    return {
        "mAP": sum(per_threshold.values()) / len(per_threshold) if per_threshold else 0.0,
        "AP50": per_threshold.get("0.50", 0.0),
        "AP75": per_threshold.get("0.75", 0.0),
        "per_iou_threshold": per_threshold,
        "per_category_AP": label_ap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute COCO-style mAP for ElecVision-R1 predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON or JSONL")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON or JSONL")
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--summary-only", action="store_true", help="Print completion status without metric values.")
    args = parser.parse_args()
    predictions = load_detection_records(args.predictions, prediction=True)
    ground_truths = load_detection_records(args.ground_truth, prediction=False)
    metrics = coco_style_map(predictions, ground_truths)
    if args.summary_only:
        text = json.dumps(
            {"status": "completed", "metric_family": "COCO-style mAP", "values_displayed": False},
            ensure_ascii=False,
            indent=2,
        )
    else:
        text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
