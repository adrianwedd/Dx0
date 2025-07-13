#!/usr/bin/env python
"""Fine-tune a language model on the prepared NEJM dataset."""
from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def train(model_name: str, data_path: str, output_dir: str) -> None:
    dataset = load_dataset("json", data_files=data_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = dataset.map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(output_dir=output_dir, num_train_epochs=1)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()
    trainer.save_model(output_dir)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local model")
    parser.add_argument("model", help="Base model name or path")
    parser.add_argument("data", help="JSONL file from prepare_finetune.py")
    parser.add_argument("output", help="Directory to save the fine-tuned model")
    parsed = parser.parse_args(args)
    train(parsed.model, parsed.data, parsed.output)


if __name__ == "__main__":  # pragma: no cover
    main()
