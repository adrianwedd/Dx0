# Dependency Security Report

Date: 2025-07-13

The project dependencies were audited using `pip-audit`:

```
$ pip-audit -r requirements-dev.txt
```

## Findings

- **torch 2.7.1** – vulnerability [GHSA-887c-mr87-cxwp](https://github.com/advisories/GHSA-887c-mr87-cxwp) (aliases CVE-2025-3730). No patched release is currently listed.

## Mitigation

Monitor upstream releases of PyTorch and upgrade when a version addressing GHSA-887c-mr87-cxwp becomes available. Until then ensure that untrusted input is not passed to vulnerable APIs.

