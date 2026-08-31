from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


BoxRecord = Mapping[str, Any]


def _finite_float_list(values: Sequence[Any], length: int = 4) -> Optional[list[float]]:
    if len(values) != length:
        return None
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in nums):
        return None
    return nums


def _xywh_to_xyxy(values: Sequence[Any]) -> Optional[list[float]]:
    nums = _finite_float_list(values)
    if nums is None:
        return None
    x, y, w, h = nums
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _normalize_annotation_role(value: Any) -> str | None:
    if value is None:
        return None
    role = str(value).strip().lower().replace("-", "_")
    if not role:
        return None
    if role in {"equipment", "equip", "device", "component", "equipment_region"}:
        return "equipment"
    if role in {"defect", "fault", "damage", "anomaly", "defect_region"}:
        return "defect"
    return role


def normalize_box_record(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, Mapping):
        return None

    label = item.get("label", item.get("category", item.get("class", item.get("category_id"))))
    raw_box = item.get("bbox_2d", item.get("bbox", item.get("box")))
    if label is None or raw_box is None:
        return None

    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
        return None
    if len(raw_box) != 4:
        return None

    mode = str(item.get("bbox_mode", item.get("box_mode", "xyxy"))).lower()
    if mode == "xywh":
        box = _xywh_to_xyxy(raw_box)
    else:
        box = _finite_float_list(raw_box)
    if box is None:
        return None

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None

    score = item.get("score", item.get("confidence", 1.0))
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 1.0
    if not math.isfinite(score_value):
        score_value = 1.0

    record = {
        "label": str(label).strip(),
        "bbox_2d": [x1, y1, x2, y2],
        "score": score_value,
    }
    role = _normalize_annotation_role(
        item.get("role", item.get("type", item.get("target_type", item.get("annotation_set"))))
    )
    if role is not None:
        record["role"] = role
    return record


def _extract_json_payload(text: str) -> Any:
    raw = text.strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        raw = answer_match.group(1).strip()

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        raw = fenced_blocks[-1].strip()

    return json.loads(raw)


_ROLE_CONTAINER_KEYS = {
    "equipment": (
        "equipment",
        "equipments",
        "equipment_predictions",
        "equipment_annotations",
        "equipment_ground_truths",
        "equipment_boxes",
    ),
    "defect": (
        "defect",
        "defects",
        "defect_predictions",
        "defect_annotations",
        "defect_ground_truths",
        "defect_boxes",
    ),
}


def _records_from_payload(payload: Any, role: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        direct_record = normalize_box_record(payload)
        if direct_record is not None:
            if role is not None and "role" not in direct_record:
                direct_record["role"] = role
            return [direct_record]

        records: list[dict[str, Any]] = []
        for channel, keys in _ROLE_CONTAINER_KEYS.items():
            for key in keys:
                if key in payload:
                    records.extend(_records_from_payload(payload[key], role=channel))

        for key in ("objects", "detections", "answer", "annotations", "ground_truth", "ground_truths", "target"):
            if key in payload:
                records.extend(_records_from_payload(payload[key], role=role))
        return records

    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_records_from_payload(item, role=role))
        return records

    return []


def parse_grounding_answer(answer: Any) -> list[dict[str, Any]]:
    if isinstance(answer, str):
        try:
            parsed = _extract_json_payload(answer)
        except json.JSONDecodeError:
            return []
    else:
        parsed = answer

    return [record for record in _records_from_payload(parsed) if record["label"]]


def load_allowed_labels(labels: Any = None) -> set[str] | None:
    if labels is None or labels == "":
        return None
    if isinstance(labels, (set, tuple, list)):
        values = labels
    else:
        path = Path(str(labels))
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            values = [line.strip() for line in raw.splitlines()]
        else:
            values = payload.keys() if isinstance(payload, Mapping) else payload

    normalized = {str(item).strip() for item in values if str(item).strip()}
    return normalized or None


def filter_allowed_labels(records: Sequence[BoxRecord], allowed_labels: set[str] | None) -> list[BoxRecord]:
    if allowed_labels is None:
        return list(records)
    return [record for record in records if str(record["label"]) in allowed_labels]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _group_by_label(records: Sequence[BoxRecord]) -> dict[str, list[BoxRecord]]:
    grouped: dict[str, list[BoxRecord]] = {}
    for record in records:
        grouped.setdefault(str(record["label"]), []).append(record)
    return grouped


def _annotation_role(record: BoxRecord, equipment_labels: set[str] | None, defect_labels: set[str] | None) -> str | None:
    explicit = _normalize_annotation_role(
        record.get("role", record.get("type", record.get("target_type", record.get("annotation_set"))))
    )
    if explicit is not None:
        return explicit

    label = str(record["label"])
    in_equipment = equipment_labels is not None and label in equipment_labels
    in_defect = defect_labels is not None and label in defect_labels
    if in_equipment and not in_defect:
        return "equipment"
    if in_defect and not in_equipment:
        return "defect"
    return None


def _group_by_role_and_label(
    records: Sequence[BoxRecord],
    equipment_labels: set[str] | None,
    defect_labels: set[str] | None,
) -> dict[tuple[str, str], list[BoxRecord]]:
    grouped: dict[tuple[str, str], list[BoxRecord]] = {}
    for record in records:
        role = _annotation_role(record, equipment_labels, defect_labels) or "unscoped"
        grouped.setdefault((role, str(record["label"])), []).append(record)
    return grouped


def _uses_annotation_sets(
    predictions: Sequence[BoxRecord],
    ground_truths: Sequence[BoxRecord],
    equipment_labels: set[str] | None,
    defect_labels: set[str] | None,
) -> bool:
    if equipment_labels is not None or defect_labels is not None:
        return True
    return any(_annotation_role(record, None, None) is not None for record in [*predictions, *ground_truths])


def average_precision(
    predictions: Sequence[BoxRecord],
    ground_truths: Sequence[BoxRecord],
    iou_threshold: float,
) -> float:
    if not ground_truths:
        return 0.0

    ordered_predictions = sorted(
        enumerate(predictions),
        key=lambda item: (-float(item[1].get("score", 1.0)), item[0]),
    )
    matched: set[int] = set()
    tp: list[float] = []
    fp: list[float] = []

    for _, pred in ordered_predictions:
        best_iou = 0.0
        best_index = -1
        for gt_index, gt in enumerate(ground_truths):
            if gt_index in matched:
                continue
            iou = bbox_iou(pred["bbox_2d"], gt["bbox_2d"])
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index

        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0

    cum_tp: list[float] = []
    cum_fp: list[float] = []
    t_total = 0.0
    f_total = 0.0
    for t, f in zip(tp, fp):
        t_total += t
        f_total += f
        cum_tp.append(t_total)
        cum_fp.append(f_total)

    precisions = [t / max(t + f, 1e-12) for t, f in zip(cum_tp, cum_fp)]
    recalls = [t / len(ground_truths) for t in cum_tp]

    interpolated = 0.0
    for recall_threshold in [i / 100 for i in range(101)]:
        candidates = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
        interpolated += max(candidates) if candidates else 0.0
    return interpolated / 101.0


def mean_average_precision(
    predictions: Sequence[BoxRecord],
    ground_truths: Sequence[BoxRecord],
    iou_thresholds: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> float:
    if not predictions and not ground_truths:
        return 1.0
    if not ground_truths:
        return 0.0

    thresholds = list(iou_thresholds or [0.5])
    pred_by_label = _group_by_label(predictions)
    gt_by_label = _group_by_label(ground_truths)
    label_set = list(labels) if labels is not None else sorted(gt_by_label)
    if not label_set:
        return 0.0

    values: list[float] = []
    for threshold in thresholds:
        for label in label_set:
            values.append(
                average_precision(
                    pred_by_label.get(str(label), []),
                    gt_by_label.get(str(label), []),
                    threshold,
                )
            )
    return sum(values) / len(values) if values else 0.0


def mean_average_precision_by_annotation_set(
    predictions: Sequence[BoxRecord],
    ground_truths: Sequence[BoxRecord],
    iou_thresholds: Sequence[float] | None = None,
    *,
    equipment_labels: set[str] | None = None,
    defect_labels: set[str] | None = None,
) -> float:
    if not predictions and not ground_truths:
        return 1.0
    if not ground_truths:
        return 0.0

    thresholds = list(iou_thresholds or [0.5])
    pred_by_key = _group_by_role_and_label(predictions, equipment_labels, defect_labels)
    gt_by_key = _group_by_role_and_label(ground_truths, equipment_labels, defect_labels)
    key_set = sorted(gt_by_key)
    if not key_set:
        return 0.0

    values: list[float] = []
    for threshold in thresholds:
        for key in key_set:
            values.append(
                average_precision(
                    pred_by_key.get(key, []),
                    gt_by_key.get(key, []),
                    threshold,
                )
            )
    return sum(values) / len(values) if values else 0.0


def prediction_penalty(predictions: Sequence[BoxRecord], ground_truths: Sequence[BoxRecord]) -> float:
    if not predictions and not ground_truths:
        return 1.0
    if not predictions:
        return 0.0
    if not ground_truths:
        return 0.0
    return min(1.0, len(ground_truths) / len(predictions))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _expand_solutions(solution: Any, n_items: int) -> list[Any]:
    if isinstance(solution, list) and len(solution) == n_items and not all(isinstance(x, Mapping) for x in solution):
        return list(solution)
    return [solution] * n_items


def _payload_has_valid_detection_format(payload: Any) -> bool:
    if isinstance(payload, list):
        return all(_payload_has_valid_detection_format(item) for item in payload)
    if isinstance(payload, Mapping):
        if normalize_box_record(payload) is not None:
            return True
        container_keys = [
            *[key for keys in _ROLE_CONTAINER_KEYS.values() for key in keys],
            "objects",
            "detections",
            "answer",
            "annotations",
            "ground_truth",
            "ground_truths",
            "target",
        ]
        present_keys = [key for key in container_keys if key in payload]
        if not present_keys:
            return False
        return all(_payload_has_valid_detection_format(payload[key]) for key in present_keys)
    return False


@dataclass
class GroundingReward:
    iou_threshold: float = 0.5
    map_weight: float = 1.0
    length_penalty: bool = True
    round_digits: int = 4
    allowed_labels: set[str] | None = None
    equipment_labels: set[str] | None = None
    defect_labels: set[str] | None = None

    def score_records(self, predictions: Sequence[BoxRecord], ground_truths: Sequence[BoxRecord]) -> float:
        predictions = filter_allowed_labels(predictions, self.allowed_labels)
        ground_truths = filter_allowed_labels(ground_truths, self.allowed_labels)
        if _uses_annotation_sets(predictions, ground_truths, self.equipment_labels, self.defect_labels):
            map_score = mean_average_precision_by_annotation_set(
                predictions,
                ground_truths,
                [self.iou_threshold],
                equipment_labels=self.equipment_labels,
                defect_labels=self.defect_labels,
            )
        else:
            map_score = mean_average_precision(predictions, ground_truths, [self.iou_threshold])
        alpha = prediction_penalty(predictions, ground_truths) if self.length_penalty else 1.0
        score = self.map_weight * alpha * map_score
        return max(0.0, min(1.0, score))

    def score_text(self, completion: Any, solution: Any) -> float:
        predictions = parse_grounding_answer(completion)
        ground_truths = parse_grounding_answer(solution)
        return round(self.score_records(predictions, ground_truths), self.round_digits)

    def __call__(self, completions: Sequence[Any], solution: Any = None, **kwargs: Any) -> list[float]:
        target = solution
        if target is None:
            target = kwargs.get("ground_truth", kwargs.get("ground_truths", kwargs.get("target")))
        if target is None:
            return [0.0] * len(completions)

        threshold = float(kwargs.get("iou_thr", self.iou_threshold))
        labels = load_allowed_labels(kwargs.get("allowed_labels"))
        equipment_labels = load_allowed_labels(kwargs.get("equipment_labels"))
        defect_labels = load_allowed_labels(kwargs.get("defect_labels"))
        scorer = GroundingReward(
            iou_threshold=threshold,
            map_weight=float(kwargs.get("map_weight", kwargs.get("accuracy_weight", self.map_weight))),
            length_penalty=_bool_value(kwargs.get("length_penalty", self.length_penalty)),
            round_digits=int(kwargs.get("round_digits", self.round_digits)),
            allowed_labels=labels,
            equipment_labels=equipment_labels if equipment_labels is not None else self.equipment_labels,
            defect_labels=defect_labels if defect_labels is not None else self.defect_labels,
        )

        solutions: Iterable[Any] = _expand_solutions(target, len(completions))
        return [scorer.score_text(completion, item) for completion, item in zip(completions, solutions)]


@dataclass
class FormatReward:
    require_answer_tags: bool = True
    allowed_labels: set[str] | None = None
    round_digits: int = 4

    def score_text(self, completion: Any) -> float:
        raw = str(completion or "").strip()
        if self.require_answer_tags and not re.search(r"<answer>.*?</answer>", raw, flags=re.DOTALL | re.IGNORECASE):
            return 0.0
        try:
            parsed = _extract_json_payload(raw)
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(parsed, (list, Mapping)):
            return 0.0
        if not _payload_has_valid_detection_format(parsed):
            return 0.0
        records = _records_from_payload(parsed)
        if self.allowed_labels is not None:
            if len(filter_allowed_labels(records, self.allowed_labels)) != len(records):
                return 0.0
        return 1.0

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        labels = load_allowed_labels(kwargs.get("allowed_labels"))
        scorer = FormatReward(
            require_answer_tags=_bool_value(kwargs.get("require_answer_tags", self.require_answer_tags)),
            allowed_labels=labels,
            round_digits=int(kwargs.get("round_digits", self.round_digits)),
        )
        return [round(scorer.score_text(completion), scorer.round_digits) for completion in completions]
