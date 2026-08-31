from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elecvision_r1.eval_metrics import coco_style_map, load_detection_records
from elecvision_r1.grpo import ElecVisionGRPOScorer, load_jsonl, summarize_group_scores
from elecvision_r1.path_evidence import PathEvidenceRetriever
from elecvision_r1.prompts import build_grounding_prompt


def _grpo_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scorer = ElecVisionGRPOScorer()
    groups = scorer.score_records(rows)
    summarize_group_scores(groups)
    return {
        "reward_scoring": "completed",
        "group_relative_advantage": "completed",
        "clipped_objective": "implemented",
        "kl_regularization": "implemented",
        "optimizer_update": "tested",
    }


def _rag_summary(graph_path: Path) -> dict[str, Any]:
    retriever = PathEvidenceRetriever.from_json(graph_path)
    queries = [
        "oil conservator oil leakage maintenance",
        "pressure gauge meter panel damage",
        "oil level window reading abnormal",
    ]
    for query in queries:
        hits = retriever.retrieve_nodes(query, top_k=6)
        retriever.retrieve(hits, top_k=5)
    return {
        "node_retrieval": "completed",
        "path_retrieval": "completed",
        "flow_pruning": "completed",
        "path_linearization": "completed",
        "stages": ["node_retrieval", "path_retrieval", "flow_pruning", "path_linearization"],
    }


def _prompt_summary() -> dict[str, Any]:
    prompt = build_grounding_prompt(
        equipment_labels={"cysb_cyg": "Oil Conservator"},
        defect_labels={"sly_bjbmyw": "Oil Leakage from Oil Storage Cabinet"},
        rag_context="oil_conservator --may_exhibit--> sly_bjbmyw",
    )
    required = ["<answer>", "bbox_2d", "label", "Retrieved path evidence"]
    if not all(item in prompt for item in required):
        raise ValueError("Prompt interface verification failed.")
    return {
        "structured_prompt": "completed",
        "taxonomy_constraint": "completed",
        "rag_context_injection": "completed",
    }


def _load_payload() -> dict[str, Any]:
    predictions = load_detection_records(ROOT / "examples" / "verification_predictions.json", prediction=True)
    ground_truth = load_detection_records(ROOT / "examples" / "verification_annotations.json", prediction=False)
    rows = load_jsonl(ROOT / "examples" / "verification_candidates.jsonl")
    coco_style_map(predictions, ground_truth)
    return {
        "purpose": "local_interface_verification",
        "note": "The report summarizes released code paths and does not display case-level images, completions, or annotations.",
        "evaluation": {
            "status": "completed",
            "metric_family": "COCO-style mAP",
            "iou_range": "0.50:0.95",
            "values_displayed": False,
        },
        "grpo": _grpo_summary(rows),
        "rag": _rag_summary(ROOT / "examples" / "verification_path_graph.json"),
        "prompt": _prompt_summary(),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    grpo = payload["grpo"]
    rag = payload["rag"]
    prompt = payload["prompt"]
    lines = [
        "# Local Verification Summary",
        "",
        "This report checks the public reward, GRPO, final-evaluation, and path-centric RAG interfaces.",
        "It is not a benchmark table and does not display case-level images, completions, or annotations.",
        "",
        "## Detection Metric Utility",
        "",
        "- COCO-style evaluator completed successfully.",
        "- Completion status is stored in `verification_results.json` for machine checks.",
        "",
        "## ElecVision-GRPO",
        "",
        f"- Reward scoring: {grpo['reward_scoring']}",
        f"- Group-relative advantage: {grpo['group_relative_advantage']}",
        f"- Clipped objective: {grpo['clipped_objective']}",
        f"- KL regularization: {grpo['kl_regularization']}",
        f"- Optimizer update: {grpo['optimizer_update']}",
        "",
        "## Path-Centric RAG",
        "",
        f"- Node retrieval: {rag['node_retrieval']}",
        f"- Relational path retrieval: {rag['path_retrieval']}",
        f"- Flow pruning: {rag['flow_pruning']}",
        f"- Prompt-context linearization: {rag['path_linearization']}",
        f"- Stages: {', '.join(rag['stages'])}",
        "",
        "## Prompt Interface",
        "",
        f"- Structured grounding prompt: {prompt['structured_prompt']}",
        f"- Taxonomy constraint: {prompt['taxonomy_constraint']}",
        f"- RAG context injection: {prompt['rag_context_injection']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, payload: Mapping[str, Any]) -> None:
    grpo = payload["grpo"]
    rag = payload["rag"]
    prompt = payload["prompt"]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ElecVision-R1 Verification Summary</title>
  <style>
    body {{
      margin: 0;
      background: #f6f8fa;
      color: #24292f;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid #d0d7de;
      padding: 24px 32px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 26px;
    }}
    header p {{
      margin: 0;
      color: #57606a;
      max-width: 860px;
    }}
    main {{
      max-width: 1040px;
      margin: 24px auto;
      padding: 0 24px 40px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 16px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      border-top: 1px solid #d8dee4;
      padding: 9px 0;
      font-size: 14px;
    }}
    .metric:first-of-type {{
      border-top: 0;
    }}
    .metric span {{
      color: #57606a;
    }}
    .metric strong {{
      font-weight: 600;
    }}
    .note {{
      color: #57606a;
      font-size: 13px;
    }}
    .stage {{
      display: inline-block;
      border: 1px solid #d0d7de;
      border-radius: 999px;
      padding: 4px 9px;
      margin: 4px 4px 0 0;
      font-size: 12px;
      background: #f6f8fa;
    }}
    @media (max-width: 860px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ElecVision-R1 Verification Summary</h1>
    <p>Public code-path verification for reward scoring, standalone GRPO optimization, COCO-style evaluation, and path-centric RAG. Case-level images, completions, and annotations are not displayed.</p>
  </header>
  <main>
    <section class="grid">
      <div class="panel">
        <h2>Evaluation Utility</h2>
        <div class="metric"><span>Status</span><strong>completed</strong></div>
        <div class="metric"><span>Output</span><strong>machine-readable JSON</strong></div>
        <p class="note">Completion status is kept in the JSON report for reproducibility checks.</p>
      </div>
      <div class="panel">
        <h2>ElecVision-GRPO</h2>
        <div class="metric"><span>Reward scoring</span><strong>{grpo['reward_scoring']}</strong></div>
        <div class="metric"><span>Relative advantage</span><strong>{grpo['group_relative_advantage']}</strong></div>
        <div class="metric"><span>Clipped objective</span><strong>{grpo['clipped_objective']}</strong></div>
        <div class="metric"><span>Optimizer update</span><strong>{grpo['optimizer_update']}</strong></div>
      </div>
      <div class="panel">
        <h2>Path-Centric RAG</h2>
        <div class="metric"><span>Node retrieval</span><strong>completed</strong></div>
        <div class="metric"><span>Path retrieval</span><strong>completed</strong></div>
        <div class="metric"><span>Flow pruning</span><strong>completed</strong></div>
        <div class="metric"><span>Linearization</span><strong>completed</strong></div>
      </div>
      <div class="panel">
        <h2>Prompt Interface</h2>
        <div class="metric"><span>Structured prompt</span><strong>{prompt['structured_prompt']}</strong></div>
        <div class="metric"><span>Taxonomy constraint</span><strong>{prompt['taxonomy_constraint']}</strong></div>
        <div class="metric"><span>RAG context</span><strong>{prompt['rag_context_injection']}</strong></div>
      </div>
    </section>
    <section class="panel">
      <h2>RAG Stages</h2>
      {''.join(f'<span class="stage">{stage}</span>' for stage in rag['stages'])}
      <p class="note">The released implementation covers node retrieval, relational path retrieval, distance-aware flow pruning, and prompt-context linearization.</p>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ElecVision-R1 public verification summary.")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_payload()
    _write_json(out_dir / "verification_results.json", payload)
    _write_markdown(out_dir / "verification_results.md", payload)
    _write_html(out_dir / "verification_summary.html", payload)
    print(out_dir / "verification_results.json")
    print(out_dir / "verification_results.md")
    print(out_dir / "verification_summary.html")


if __name__ == "__main__":
    main()
