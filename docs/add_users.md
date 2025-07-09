# Adding UI Users

The Physician UI validates logins against a YAML file containing bcrypt password hashes. By default the file is `sdb/ui/users.yml`.

## Generating a Hash

Run the snippet below and copy the printed hash into the YAML file.

```bash
python - <<'PY'
import bcrypt, getpass
pw = getpass.getpass("Password: ")
print(bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode())
PY
```

## Updating `users.yml`

Edit the `users` mapping in `sdb/ui/users.yml` to add a new entry:

```yaml
users:
  physician: "$2b$12$existinghash..."
  newuser: "$2b$12$generatedhash..."
```

## Custom Credential Paths

`sdb/ui/app.py` loads credentials from the path specified by the `UI_USERS_FILE` environment variable. Set this variable to use a different YAML file when deploying the UI.
