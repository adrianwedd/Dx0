"""Benchmark LLM chat latency."""
from __future__ import annotations

import argparse
import time
from typing import List

from sdb.llm_client import HFLocalClient, OllamaClient, OpenAIClient


CLIENTS = {
    "openai": OpenAIClient,
    "ollama": OllamaClient,
    "hf-local": HFLocalClient,
}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM latency")
    parser.add_argument("provider", choices=CLIENTS.keys(), help="LLM provider")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--prompt", default="Hello", help="Prompt text")
    parser.add_argument("--runs", type=int, default=5, help="Number of requests")
    parser.add_argument("--model-path", help="Local model path for hf-local")
    args = parser.parse_args(argv)

    if args.provider == "hf-local":
        if not args.model_path:
            raise SystemExit("--model-path required for hf-local provider")
        client = HFLocalClient(args.model_path)
    else:
        client = CLIENTS[args.provider]()

    messages = [{"role": "user", "content": args.prompt}]
    start = time.perf_counter()
    for _ in range(args.runs):
        client.chat(messages, model=args.model)
    duration = time.perf_counter() - start

    print(f"avg_llm_latency={duration/args.runs:.4f}s over {args.runs} runs")


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
