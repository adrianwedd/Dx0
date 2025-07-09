# Dependency Management

This project pins exact versions for all development packages. When a new dependency
is added or an update is required, regenerate the lock file and audit the
resulting environment.

```bash
pip install pip-tools pip-audit
pip-compile --output-file=requirements.lock requirements-dev.txt
pip install -r requirements-dev.txt
pip-audit -r requirements.lock
```

`pip-compile` ensures deterministic installs while `pip-audit` checks for
known vulnerabilities. Commit both `requirements-dev.txt` and
`requirements.lock` after running these commands.
