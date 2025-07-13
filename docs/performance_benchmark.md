# Performance Benchmarking

This guide explains how to collect baseline latency metrics for retrieval,
LLM calls, and the in-memory cache. The `run_benchmarks.py` script executes
short load tests and writes the results to a CSV file that can be tracked
over time.

```bash
python scripts/run_benchmarks.py data/sdbench/cases/cases.json \
    --provider openai --model gpt-3.5-turbo \
    --output results/performance_baseline.csv
```

The output CSV contains average latencies in seconds:

```
retrieval_avg_s,llm_avg_s,cache_cold_avg_s,cache_warm_avg_s
0.1234,0.4500,0.1500,0.0100
```

Import `performance-dashboard.json` into Grafana to visualize these metrics
alongside Prometheus counters. By saving the CSV in version control, future
runs can detect regressions when the numbers drift upward.
