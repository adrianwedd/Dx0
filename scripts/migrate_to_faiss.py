"""Convert existing embedding vectors to a FAISS index."""

from __future__ import annotations

import argparse
import numpy as np

try:
    import faiss
except Exception as exc:  # pragma: no cover - faiss not installed
    raise RuntimeError("faiss library is required") from exc


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate embeddings to FAISS")
    parser.add_argument("--input", required=True, help="Path to .npy embeddings")
    parser.add_argument(
        "--output", default="index.faiss", help="Destination FAISS index file"
    )
    parsed = parser.parse_args(args)

    embeddings = np.load(parsed.input).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, parsed.output)


if __name__ == "__main__":  # pragma: no cover
    main()

