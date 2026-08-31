from __future__ import annotations

from collections.abc import Mapping, Sequence


def _format_label_map(title: str, labels: Mapping[str, str] | Sequence[str] | None) -> list[str]:
    if not labels:
        return []
    lines = [f"{title}:"]
    if isinstance(labels, Mapping):
        for code, name in labels.items():
            lines.append(f"- {code}: {name}")
    else:
        for code in labels:
            lines.append(f"- {code}")
    return lines


def build_grounding_prompt(
    *,
    task_instruction: str = "Identify equipment regions and defect regions in the image.",
    equipment_labels: Mapping[str, str] | Sequence[str] | None = None,
    defect_labels: Mapping[str, str] | Sequence[str] | None = None,
    rag_context: str = "",
) -> str:
    lines = [
        "You are a power equipment fault inspector.",
        task_instruction.strip(),
        "Return a JSON-formatted response enclosed within <answer> tags.",
        'Each item must contain "bbox_2d" as [x1, y1, x2, y2] pixel coordinates and "label" as a predefined label code.',
        "Do not use labels outside the provided taxonomy.",
    ]
    lines.extend(_format_label_map("Equipment labels", equipment_labels))
    lines.extend(_format_label_map("Defect labels", defect_labels))
    if rag_context.strip():
        lines.extend(["Retrieved path evidence:", rag_context.strip()])
        lines.append("Use the retrieved evidence as reference material for diagnostic reasoning.")
    lines.append('Output format: <answer>[{"bbox_2d":[x1,y1,x2,y2],"label":"label_code"}]</answer>')
    return "\n".join(line for line in lines if line)
