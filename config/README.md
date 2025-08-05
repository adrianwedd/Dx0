# Configuration Profiles

This directory contains environment-specific configuration files for the SDBench application.

## Files

- `development.yml` - Settings for local development
- `staging.yml` - Settings for staging/testing environment
- `production.yml` - Settings for production deployment
- `local.yml` - Local overrides (gitignored, optional)

## Usage

### Loading Configuration

The application automatically loads configuration in this order:

1. Default values from `sdb/config.py`
2. Environment-specific YAML file (if `SDBENCH_ENV` is set)
3. `config/local.yml` (if it exists)
4. Environment variables (highest priority)

### Environment Variable

Set `SDBENCH_ENV` to automatically load the corresponding configuration:

```bash
export SDBENCH_ENV=development  # loads config/development.yml
export SDBENCH_ENV=staging      # loads config/staging.yml  
export SDBENCH_ENV=production   # loads config/production.yml
```

### Manual Configuration Loading

You can also manually specify a configuration file:

```bash
python -c "from sdb.config import load_settings; settings = load_settings('config/development.yml')"
```

## Security Notes

### Development
- Uses default/weak secrets (acceptable for local development)
- Enables tracing and debug features
- Permissive rate limiting

### Staging
- Similar to production but with relaxed security for testing
- Requires some environment variables (secrets)
- Moderate rate limiting

### Production
- **ALL sensitive values must be set via environment variables**
- Strong security defaults
- Aggressive rate limiting
- Requires proper secret management

## Required Environment Variables by Environment

### Development
- None (all have defaults)

### Staging
- `UI_SECRET_KEY` - JWT signing key
- `OPENAI_API_KEY` - OpenAI API access
- `SENTRY_DSN` - (optional) Error reporting

### Production
- `UI_SECRET_KEY` - **CRITICAL** Strong random JWT signing key
- `OPENAI_API_KEY` - OpenAI API access
- `SDB_CASE_DB_SQLITE` - Database file path
- `SESSIONS_DB` - Session database path
- `SENTRY_DSN` - Error reporting
- `SDB_TRACING_HOST` - Tracing service host

## Example Environment Variables

```bash
# Production example
export SDBENCH_ENV=production
export UI_SECRET_KEY="your-256-bit-secret-key-here"
export OPENAI_API_KEY="sk-..."
export SDB_CASE_DB_SQLITE="/app/data/cases.db"
export SESSIONS_DB="/app/data/sessions.db"
export SENTRY_DSN="https://...@ingest.sentry.io/..."
export SDB_TRACING_HOST="jaeger.company.com"
```

## Creating Local Overrides

Create `config/local.yml` to override any settings locally:

```yaml
# config/local.yml
openai_model: "gpt-3.5-turbo"  # Use cheaper model for testing
ui_budget_limit: 50.0          # Lower budget for testing
```

This file is gitignored and won't be committed to version control.