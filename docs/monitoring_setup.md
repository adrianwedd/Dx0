# Monitoring with Prometheus and Grafana

This guide explains how to collect metrics from Dx0 and visualize them with Grafana.

## Starting the Metrics Server

Launch the application or CLI with the metrics server enabled. Metrics are exported on port `8000` by default. The helper `start_metrics_server()` in `sdb.metrics` starts the HTTP endpoint:

```bash
python cli.py --metrics-port 8000 ...
```

## Prometheus Configuration

Use the sample `prometheus.yml` to scrape the metrics endpoint:

```yaml
scrape_configs:
  - job_name: 'dx0'
    static_configs:
      - targets: ['localhost:8000']
```

Run Prometheus pointing to this file:

```bash
prometheus --config.file docs/prometheus.yml
```

## Running Grafana

Start Grafana locally (for example using Docker):

```bash
docker run -p 3000:3000 grafana/grafana
```

Add Prometheus as a data source at `http://localhost:9090` and import the dashboards from the `docs/` folder:

- `grafana-dashboard.json` for general metrics
- `budget-dashboard.json` for budget tracking

The screenshots below show the two dashboards after importing.

![Metrics Dashboard](images/grafana_metrics.png.b64)
![Budget Dashboard](images/grafana_budget.png.b64)

