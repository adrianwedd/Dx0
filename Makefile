.PHONY: install-dev
install-dev:
	python -m pip install -r requirements-dev.txt

.PHONY: docs-api
docs-api:
	python scripts/export_openapi.py --info

.PHONY: docs-api-validate
docs-api-validate:
	python scripts/export_openapi.py --validate

.PHONY: docs-api-json
docs-api-json:
	python scripts/export_openapi.py --format json

.PHONY: docs-api-yaml
docs-api-yaml:
	python scripts/export_openapi.py --format yaml
