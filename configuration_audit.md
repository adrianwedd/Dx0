# Configuration Audit Report

## Executive Summary
The codebase has a **partially centralized configuration system** with most hardcoded values already moved to `sdb/config.py`. However, several areas need attention for complete configuration centralization.

## Current State: ✅ GOOD
- **Configuration Framework**: Pydantic-based config in `sdb/config.py`
- **Environment Variables**: 13+ environment variables already supported
- **Default Values**: Sensible defaults provided for most settings

## Hardcoded Values Found

### 🟡 MINOR ISSUES (Already in config.py as defaults)
- `sdb/config.py:13`: `openai_model: str = "gpt-4"` ✅
- `sdb/config.py:15`: `ollama_base_url: HttpUrl = "http://localhost:11434"` ✅
- `sdb/config.py:27`: `tracing_host: str = "localhost"` ✅

### 🟠 MODERATE ISSUES (Need centralization)
- `sdb/decision.py:133`: `model: str = "gpt-4"` - Should use settings.openai_model
- `sdb/judge.py:21`: `model: str = "gpt-4"` - Should use settings.openai_model

### 🟢 GOOD PRACTICES (Already using environment variables)
- UI Budget Limit: `UI_BUDGET_LIMIT`
- Sentry DSN: `SENTRY_DSN`
- UI Secret Key: `UI_SECRET_KEY`
- Session Settings: `UI_TOKEN_TTL`, `SESSIONS_DB`
- Rate Limiting: `MESSAGE_RATE_LIMIT`, `MESSAGE_RATE_WINDOW`
- Metrics Port: `SDB_METRICS_PORT`
- OpenAI API Key: `OPENAI_API_KEY`

## Database Configuration: ✅ EXCELLENT
- All database paths are configurable
- SQLite connections use environment variables or config settings
- No hardcoded database URLs found

## API Endpoints: ✅ GOOD
- API versioning consistent (`/api/v1/`)
- No hardcoded external service URLs
- Static file serving properly configured

## Recommendations

### High Priority (Fix Now)
1. **Model Name Centralization**: Replace hardcoded "gpt-4" in decision.py and judge.py
2. **Add Missing Config Options**: Add any missing configuration parameters

### Medium Priority (Next Phase)
1. **Environment Profiles**: Create dev/staging/prod configuration templates
2. **Configuration Validation**: Add more comprehensive validation

### Low Priority (Future)
1. **Configuration Hot Reload**: Consider adding runtime configuration updates

## Risk Assessment: 🟢 LOW
- Most configuration is already centralized
- Only 2 hardcoded model references need fixing
- Strong foundation for environment-specific deployments

## Next Steps
1. ✅ Update decision.py and judge.py to use settings.openai_model
2. ✅ Add any missing configuration options to sdb/config.py
3. ✅ Create environment profile templates