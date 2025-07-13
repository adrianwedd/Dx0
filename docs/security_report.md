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

---

The project was also scanned with `safety` using the locked dependencies:

```
$ safety check -r requirements.lock
```

### Additional Findings

- **pyjwt 2.8.0** – vulnerability [CVE-2024-53861](https://data.safetycli.com/v/74429/97c) allows issuer claim bypass due to partial string comparison.

### Mitigation

Upgrade `pyjwt` to version 2.10.1 or later which resolves the partial comparison issue.

