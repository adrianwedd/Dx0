# Fine-Tuning a Domain-Specific Model

This guide outlines how to train a Hugging Face model on the NEJM case corpus
and integrate it with Dx0.

1. **Prepare the dataset**
   Use `scripts/prepare_finetune.py` to convert the JSON cases into a JSONL file
   suitable for language model training:

   ```bash
   python scripts/prepare_finetune.py data/sdbench/cases nejm_cases.jsonl
   ```

2. **Train the model**
   Run `scripts/train_domain_model.py` with a base model and the prepared
   dataset. The example below performs one epoch of causal language modelling:

   ```bash
   python scripts/train_domain_model.py mistral-base nejm_cases.jsonl models/nejm
   ```

   The script uses `transformers.Trainer` and saves the fine-tuned weights to the
   specified output directory.

3. **Configure Dx0**
   Start the CLI with `--llm-provider hf-local` and pass the model path via
   `--hf-model` or set `hf_model` in the YAML configuration file:

   ```yaml
   hf_model: models/nejm
   ```

   The `HFLocalClient` loads the model using `transformers.pipeline` and can be
   selected just like the OpenAI or Ollama providers.
