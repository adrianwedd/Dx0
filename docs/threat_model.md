# Threat Model Overview

This document summarizes key security considerations for Dx0.

## Authentication

User accounts for the web UI are defined in `sdb/ui/users.yml`. Passwords are stored as bcrypt hashes. The path can be overridden with the `UI_USERS_FILE` environment variable. Session tokens for the CLI are saved to `~/.dx0/token.json` with permissions `0600`.

**Threats**: credential theft, weak passwords, unauthorized access.

**Mitigations**:
- Use strong, unique passwords when adding entries to `users.yml`.
- Protect the token file with correct file permissions and avoid committing it to version control.
- Limit network exposure of the UI and use HTTPS in production deployments.

## Data Storage

Case data and evaluation metrics can be stored in JSON or SQLite databases. Example paths include `cases.db` and `results.db`. Some scripts support uploading data to S3 via DVC.

**Threats**: unauthorized access or tampering with case data; leakage of PHI if real patient cases are used.

**Mitigations**:
- Restrict file permissions on local databases.
- Use access controls on any remote storage such as S3 buckets.
- Ensure only de‑identified cases are stored in shared locations.

## Logging

Structured logs are emitted via `structlog` and may include panel actions and metrics. Metrics can also be exported to Prometheus.

**Threats**: log injection or exposure of sensitive information in logs.

**Mitigations**:
- Sanitize user input before logging.
- Limit log retention and restrict access to log storage.

