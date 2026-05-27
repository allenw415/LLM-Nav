from __future__ import annotations

import argparse

from ._common import PROJECT_ROOT, ensure_project_root_on_path, render_json, resolve_project_path, write_text_if_requested

ensure_project_root_on_path()

from st_nav import MemoryImageRetriever, load_dotenv  # noqa: E402
from st_nav_data.memory_localization import DEFAULT_SIGLIP2_MODEL  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve memory evidence images from one target room using passage-image or semantic-text queries."
    )
    parser.add_argument("--target-room-id", required=True, help='Target memory room, e.g. "Room 8".')
    parser.add_argument("--query-images", nargs="*", help="Passage image paths to use as visual queries.")
    parser.add_argument("--semantic-query", help="Neutral passage semantic query text for text-to-image retrieval.")
    parser.add_argument("--mode", choices=["image", "semantic"], default="image")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-k-per-query", type=int, default=4)
    parser.add_argument("--require-existing-images", action="store_true")
    parser.add_argument("--index-path", default="artifacts/memory_localization/floor0_siglip2_images.npz")
    parser.add_argument("--metadata-path", default="artifacts/memory_localization/floor0_siglip2_images.metadata.json")
    parser.add_argument("--faiss-path", default="artifacts/memory_localization/floor0_siglip2_images.faiss")
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--output-path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    retriever = MemoryImageRetriever(
        index_path=resolve_project_path(args.index_path),
        metadata_path=resolve_project_path(args.metadata_path),
        faiss_path=resolve_project_path(args.faiss_path),
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        use_faiss=not args.no_faiss,
        project_root=PROJECT_ROOT,
    )

    if args.mode == "semantic":
        if not args.semantic_query:
            raise SystemExit("--semantic-query is required when --mode semantic.")
        memories = retriever.retrieve_room_memories_for_text_queries(
            args.target_room_id,
            [args.semantic_query],
            top_k_per_query=args.top_k_per_query,
            max_memories=args.top_k,
            require_existing_images=args.require_existing_images,
        )
        query_payload = {"mode": "semantic", "semantic_query": args.semantic_query}
    else:
        query_images = [resolve_project_path(path) for path in args.query_images or []]
        if not query_images:
            raise SystemExit("--query-images is required when --mode image.")
        memories = retriever.retrieve_room_memories_for_query_images(
            args.target_room_id,
            query_images,
            passage_labels=[f"query_{index}" for index, _ in enumerate(query_images)],
            top_k_per_query=args.top_k_per_query,
            max_memories=args.top_k,
            require_existing_images=args.require_existing_images,
        )
        query_payload = {"mode": "image", "query_images": [str(path) for path in query_images]}

    payload = {
        "target_room_id": args.target_room_id,
        "query": query_payload,
        "top_k": args.top_k,
        "result_count": len(memories),
        "memories": memories,
    }
    output = render_json(payload)
    write_text_if_requested(output, args.output_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
