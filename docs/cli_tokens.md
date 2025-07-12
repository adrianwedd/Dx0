# CLI Session Tokens

The `dx0 login` command stores your API access and refresh tokens in
`~/.dx0/token.json` by default. The file permissions are set so only your
account can read it (`0600`). Avoid checking this file into version control or
sharing it with other users. If you need to use a different location, pass the
`--token-file` option when logging in.

Tokens are short lived. The helper functions in `sdb.token` automatically refresh
the access token using the stored refresh token when necessary.
