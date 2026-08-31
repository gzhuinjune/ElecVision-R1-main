from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .reward import FormatReward, GroundingReward


@dataclass
class GRPOConfig:
    format_weight: float = 1.0
    accuracy_weight: float = 1.0
    advantage_eps: float = 1e-6
    iou_threshold: float = 0.5
    require_answer_tags: bool = True
    equipment_labels: Sequence[str] | None = None
    defect_labels: Sequence[str] | None = None


@dataclass
class CandidateScore:
    completion: str
    format_reward: float
    accuracy_reward: float
    total_reward: float
    advantage: float = 0.0


@dataclass
class GroupScore:
    record_id: str
    prompt: str
    scores: list[CandidateScore]
    reward_mean: float
    reward_std: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    variance = sum((item - mu) ** 2 for item in values) / len(values)
    return math.sqrt(max(0.0, variance))


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    mu = _mean(rewards)
    sigma = _std(rewards)
    if sigma < eps:
        return [0.0 for _ in rewards]
    return [(reward - mu) / (sigma + eps) for reward in rewards]


class ElecVisionGRPOScorer:
    def __init__(self, config: GRPOConfig | None = None):
        self.config = config or GRPOConfig()
        self.format_reward = FormatReward(require_answer_tags=self.config.require_answer_tags)
        equipment_labels = set(self.config.equipment_labels) if self.config.equipment_labels is not None else None
        defect_labels = set(self.config.defect_labels) if self.config.defect_labels is not None else None
        self.grounding_reward = GroundingReward(
            iou_threshold=self.config.iou_threshold,
            equipment_labels=equipment_labels,
            defect_labels=defect_labels,
        )

    def score_group(
        self,
        *,
        record_id: str,
        prompt: str,
        completions: Sequence[Any],
        solution: Any,
    ) -> GroupScore:
        format_scores = self.format_reward(completions)
        accuracy_scores = self.grounding_reward(
            completions,
            solution=solution,
            equipment_labels=self.config.equipment_labels,
            defect_labels=self.config.defect_labels,
        )
        total_rewards = [
            self.config.format_weight * fmt + self.config.accuracy_weight * acc
            for fmt, acc in zip(format_scores, accuracy_scores)
        ]
        advantages = group_relative_advantages(total_rewards, self.config.advantage_eps)

        scores: list[CandidateScore] = []
        for completion, fmt, acc, total, advantage in zip(
            completions,
            format_scores,
            accuracy_scores,
            total_rewards,
            advantages,
        ):
            scores.append(
                CandidateScore(
                    completion=str(completion),
                    format_reward=round(float(fmt), 4),
                    accuracy_reward=round(float(acc), 4),
                    total_reward=round(float(total), 4),
                    advantage=round(float(advantage), 4),
                )
            )

        return GroupScore(
            record_id=str(record_id),
            prompt=str(prompt),
            scores=scores,
            reward_mean=round(_mean(total_rewards), 4),
            reward_std=round(_std(total_rewards), 4),
        )

    def score_records(self, rows: Iterable[Mapping[str, Any]]) -> list[GroupScore]:
        results: list[GroupScore] = []
        for idx, row in enumerate(rows):
            completions = row.get("completions", row.get("candidates", row.get("responses", [])))
            if not isinstance(completions, list):
                completions = [completions]
            solution = row.get("solution", row.get("ground_truth", row.get("target")))
            results.append(
                self.score_group(
                    record_id=str(row.get("id", row.get("record_id", idx))),
                    prompt=str(row.get("prompt", row.get("instruction", ""))),
                    completions=completions,
                    solution=solution,
                )
            )
        return results


def load_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(payload)
    return rows


def save_group_scores(groups: Sequence[GroupScore], path: str | Path) -> None:
    payload = [asdict(group) for group in groups]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_group_scores(groups: Sequence[GroupScore]) -> dict[str, Any]:
    all_rewards = [score.total_reward for group in groups for score in group.scores]
    all_accuracy = [score.accuracy_reward for group in groups for score in group.scores]
    best_rewards = [max((score.total_reward for score in group.scores), default=0.0) for group in groups]
    return {
        "group_count": len(groups),
        "candidate_count": len(all_rewards),
        "mean_total_reward": round(_mean(all_rewards), 4),
        "mean_accuracy_reward": round(_mean(all_accuracy), 4),
        "mean_best_reward": round(_mean(best_rewards), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ElecVision-GRPO candidate groups.")
    parser.add_argument("--input", required=True, help="JSONL file containing prompts, candidate completions, and solutions.")
    parser.add_argument("--output", required=True, help="Path for the scored JSON report.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--format-weight", type=float, default=1.0)
    parser.add_argument("--accuracy-weight", type=float, default=1.0)
    args = parser.parse_args()

    config = GRPOConfig(
        iou_threshold=args.iou_threshold,
        format_weight=args.format_weight,
        accuracy_weight=args.accuracy_weight,
    )
    scorer = ElecVisionGRPOScorer(config)
    groups = scorer.score_records(load_jsonl(args.input))
    save_group_scores(groups, args.output)
    print(json.dumps(summarize_group_scores(groups), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
