#!/bin/bash
#SBATCH --job-name=batch_eval
#SBATCH --array=1-10
#SBATCH --output=logs/batch_eval_%a.out

# Example Slurm submission script
python scripts/slurm_batch_eval.py \
  --start 1 --end 10 \
  --concurrency 4 \
  --output results.csv
