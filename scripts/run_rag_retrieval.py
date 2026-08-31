from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elecvision_r1.path_evidence import PathEvidenceRetriever, SentenceTransformerEncoder


def _load_embedding(path: str) -> list[float] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("embedding", payload.get("query_embedding"))
    if not isinstance(payload, list):
        raise ValueError("Query embedding must be a JSON list or an object with an embedding field.")
    return [float(value) for value in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run path-centric graph retrieval for ElecVision-R1.")
    parser.add_argument("--graph", default="examples/verification_path_graph.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-top-k", type=int, default=6)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--encoder-model", default="", help="Optional sentence-transformers model for dense node retrieval.")
    parser.add_argument("--query-embedding", default="", help="Optional JSON vector for query-side dense retrieval.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    encoder = SentenceTransformerEncoder(args.encoder_model) if args.encoder_model else None
    query_embedding = _load_embedding(args.query_embedding)
    retriever = PathEvidenceRetriever.from_json(ROOT / args.graph, encoder=encoder)
    hits = retriever.retrieve_nodes(args.query, top_k=args.seed_top_k, query_embedding=query_embedding)
    evidence = retriever.retrieve(hits, top_k=args.top_k, max_hops=args.max_hops, threshold=args.threshold)
    payload = {
        "node_hits": [hit.__dict__ for hit in hits],
        "paths": [item.__dict__ | {"path": list(item.path)} for item in evidence],
        "prompt_context": retriever.to_prompt_context(evidence),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
