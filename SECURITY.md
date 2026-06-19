# Security

## Reporting a vulnerability

Please report security issues privately to the maintainers rather than opening a
public issue. Include a description, affected version/commit, and reproduction
steps. We aim to acknowledge reports within a few business days.

## Threat model

Picochat ships two local HTTP servers, and both can consume real resources:

- `pico web` (`picochat.web`) — the dashboard API. It can **launch training
  subprocesses, download datasets, and write files** under the working
  directory. Treat it as a code-execution surface.
- `pico serve` (`picochat.serve`) — an OpenAI-compatible inference server. It
  spends compute and exposes the trained model.

### Default posture: loopback only

Both servers bind `127.0.0.1` by default. On loopback they are reachable only
from the local machine and run without a token for zero-friction local use.

### Exposure requires a token

If you bind a non-loopback address (e.g. `--host 0.0.0.0`):

- `pico web` mints an auth token automatically (or honors one you supply) and
  refuses to serve the `/api/*` surface without it. The token is printed at
  startup and is carried in the URL as `?token=…`; the UI stores it and strips
  it from the address bar. Programmatic clients send it as the
  `X-Picochat-Token` header or `Authorization: Bearer <token>`.
- `pico serve` mints a bearer API key automatically (or honors `--api-key`) and
  rejects `/v1/*` requests without `Authorization: Bearer <key>`.

### CSRF / DNS-rebinding

State-changing requests to `pico web` are rejected when the `Origin` header does
not match the request `Host`. This blocks a malicious web page from driving the
localhost API in a logged-in browser, and mitigates DNS-rebinding.

### What is still out of scope

- No multi-user accounts, roles, or per-user audit log yet (single trusted
  operator model). Do not expose either server to untrusted networks.
- Tokens are bearer credentials with no rotation/expiry; treat them as secrets.
- The dashboard does not sandbox the training commands it launches.
