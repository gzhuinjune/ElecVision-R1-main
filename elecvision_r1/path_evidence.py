from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class NodeHit:
    node_id: str
    score: float
    text: str


@dataclass(frozen=True)
class PathEvidence:
    path: tuple[str, ...]
    score: float
    text: str


class HashingTextEncoder:
    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)


class SentenceTransformerEncoder:
    def __init__(self, model_name_or_path: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


class DenseNodeIndexer:
    def __init__(self, nodes: Mapping[str, Mapping[str, Any]], encoder: Any | None = None):
        self.nodes = nodes
        self.encoder = encoder or HashingTextEncoder()
        self.node_vectors = {node_id: self._node_vector(node_id, data) for node_id, data in nodes.items()}

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        min_score: float = 0.0,
        query_embedding: Sequence[float] | None = None,
    ) -> list[NodeHit]:
        query_vector = _l2_normalize(query_embedding) if query_embedding is not None else self.encoder.encode(query)
        hits: list[NodeHit] = []
        for node_id, node_vector in self.node_vectors.items():
            score = _cosine(query_vector, node_vector)
            if score >= min_score:
                hits.append(NodeHit(node_id=node_id, score=score, text=self._node_text(node_id, self.nodes[node_id])))
        hits.sort(key=lambda item: (item.score, item.node_id), reverse=True)
        return hits[:top_k]

    def _node_vector(self, node_id: str, data: Mapping[str, Any]) -> list[float]:
        embedding = data.get("embedding")
        if isinstance(embedding, Sequence) and not isinstance(embedding, (str, bytes)):
            try:
                return _l2_normalize([float(value) for value in embedding])
            except (TypeError, ValueError):
                embedding = None
        return self.encoder.encode(self._node_text(node_id, data))

    @staticmethod
    def _node_text(node_id: str, data: Mapping[str, Any]) -> str:
        fields = [
            node_id,
            node_id.replace("_", " "),
            data.get("name", ""),
            data.get("type", data.get("entity_type", "")),
            data.get("description", ""),
        ]
        aliases = data.get("aliases", [])
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
            fields.extend(str(alias) for alias in aliases)
        return " ".join(str(field) for field in fields if str(field).strip())


class PathEvidenceRetriever:
    def __init__(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        *,
        encoder: Any | None = None,
    ):
        self.nodes = {str(k): dict(v) for k, v in nodes.items()}
        self.edges = [dict(edge) for edge in edges]
        self.adj: dict[str, list[tuple[str, dict[str, Any]]]] = {node: [] for node in self.nodes}
        for edge in self.edges:
            src = str(edge["source"])
            dst = str(edge["target"])
            data = dict(edge)
            data.setdefault("relation", data.get("keywords", "related_to"))
            data.setdefault("weight", 1.0)
            self.adj.setdefault(src, []).append((dst, data))
            if not data.get("directed", False):
                reverse = dict(data)
                reverse["source"], reverse["target"] = dst, src
                reverse["relation"] = data.get("reverse_relation", f"reverse_of:{data.get('relation', 'related_to')}")
                self.adj.setdefault(dst, []).append((src, reverse))
        self.indexer = DenseNodeIndexer(self.nodes, encoder=encoder)

    @classmethod
    def from_json(cls, path: str | Path, *, encoder: Any | None = None) -> "PathEvidenceRetriever":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = payload.get("nodes", {})
        if isinstance(nodes, list):
            nodes = {str(node["id"]): node for node in nodes}
        return cls(nodes=nodes, edges=payload.get("edges", []), encoder=encoder)

    def retrieve_nodes(
        self,
        query: str,
        *,
        top_k: int = 6,
        min_score: float = 0.0,
        query_embedding: Sequence[float] | None = None,
    ) -> list[NodeHit]:
        return self.indexer.search(query, top_k=top_k, min_score=min_score, query_embedding=query_embedding)

    def seeds_from_query(self, query: str, *, top_k: int = 6) -> list[str]:
        return [hit.node_id for hit in self.retrieve_nodes(query, top_k=top_k)]

    def retrieve(
        self,
        seed_entities: Iterable[str | NodeHit],
        *,
        max_hops: int = 3,
        top_k: int = 5,
        alpha: float = 0.8,
        threshold: float = 0.05,
    ) -> list[PathEvidence]:
        seed_scores = self._seed_scores(seed_entities)
        seeds = [node_id for node_id in seed_scores if node_id in self.adj]
        scored: list[PathEvidence] = []
        for i, src in enumerate(seeds):
            for dst in seeds[i + 1 :]:
                base_flow = math.sqrt(max(seed_scores[src], 0.0) * max(seed_scores[dst], 0.0))
                for path, edge_datas, resources in self._flow_paths(
                    src,
                    dst,
                    max_hops=max_hops,
                    alpha=alpha,
                    threshold=threshold,
                    initial_flow=base_flow,
                ):
                    score = self._path_score(resources)
                    if score >= threshold:
                        scored.append(PathEvidence(tuple(path), score, self.linearize(path, edge_datas)))
        scored.sort(key=lambda item: (item.score, -len(item.path), item.path), reverse=True)
        return _dedupe_paths(scored, top_k=top_k)

    def retrieve_for_query(self, query: str, **kwargs: Any) -> list[PathEvidence]:
        seed_top_k = int(kwargs.pop("seed_top_k", 6))
        min_score = float(kwargs.pop("node_min_score", 0.0))
        query_embedding = kwargs.pop("query_embedding", None)
        hits = self.retrieve_nodes(query, top_k=seed_top_k, min_score=min_score, query_embedding=query_embedding)
        return self.retrieve(hits, **kwargs)

    @staticmethod
    def to_prompt_context(evidence: Sequence[PathEvidence]) -> str:
        if not evidence:
            return "No path evidence was retrieved."
        return "\n".join(f"[Path {idx}; score={item.score:.3f}] {item.text}" for idx, item in enumerate(evidence, 1))

    def _flow_paths(
        self,
        src: str,
        dst: str,
        *,
        max_hops: int,
        alpha: float,
        threshold: float,
        initial_flow: float,
    ):
        stack: list[tuple[str, list[str], list[dict[str, Any]], list[float]]] = [(src, [src], [], [max(initial_flow, 0.0)])]
        while stack:
            node, path, edge_datas, resources = stack.pop()
            if node == dst and edge_datas:
                yield path, edge_datas, resources
                continue
            if len(edge_datas) >= max_hops:
                continue
            hop = len(edge_datas)
            candidates = [(nxt, edge_data) for nxt, edge_data in self.adj.get(node, []) if nxt not in path]
            total_weight = sum(max(float(edge.get("weight", 1.0)), 0.0) for _, edge in candidates)
            if total_weight <= 0:
                continue
            for nxt, edge_data in candidates:
                if nxt in path:
                    continue
                edge_weight = max(float(edge_data.get("weight", 1.0)), 0.0)
                direction_factor = 0.5 if str(edge_data.get("relation", "")).startswith("reverse_of:") else 1.0
                propagated = resources[-1] * (edge_weight / total_weight) * direction_factor * (alpha**hop)
                if propagated < threshold:
                    continue
                stack.append((nxt, path + [nxt], edge_datas + [edge_data], resources + [propagated]))

    @staticmethod
    def _path_score(resources: Sequence[float]) -> float:
        if len(resources) <= 1:
            return 0.0
        path_resources = resources[1:]
        return max(0.0, min(1.0, sum(path_resources) / len(path_resources)))

    def _seed_scores(self, seed_entities: Iterable[str | NodeHit]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item in seed_entities:
            if isinstance(item, NodeHit):
                node_id = item.node_id
                score = item.score
            else:
                node_id = str(item)
                score = 1.0
            if node_id in self.nodes:
                scores[node_id] = max(scores.get(node_id, 0.0), float(score))
        return scores

    def linearize(self, path: Sequence[str], edge_datas: Sequence[Mapping[str, Any]]) -> str:
        parts: list[str] = []
        for idx, node in enumerate(path):
            node_data = self.nodes.get(node, {})
            desc = node_data.get("description", "")
            node_type = node_data.get("type", node_data.get("entity_type", "entity"))
            if idx == 0:
                parts.append(f"{node} ({node_type}: {desc})")
            else:
                edge = edge_datas[idx - 1]
                relation = edge.get("relation", edge.get("keywords", "related_to"))
                parts.append(f"--{relation}--> {node} ({node_type}: {desc})")
        return " ".join(parts)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens


def _l2_normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0:
        return [0.0 for _ in values]
    return [float(value) / norm for value in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        size = min(len(a), len(b))
        a = a[:size]
        b = b[:size]
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _dedupe_paths(paths: Sequence[PathEvidence], *, top_k: int) -> list[PathEvidence]:
    kept: list[PathEvidence] = []
    seen: set[tuple[str, ...]] = set()
    for item in paths:
        if item.path in seen:
            continue
        seen.add(item.path)
        kept.append(item)
        if len(kept) >= top_k:
            break
    return kept
