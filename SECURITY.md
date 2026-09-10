# Security

## Reporting a vulnerability

Please report it privately through GitHub: **Security → Report a
vulnerability** on this repository. Do not open a public issue for anything
that could expose a user's API key, their machine, or other players' data.

## What this project touches, and how each is contained

| Surface | Exposure | Containment |
|---|---|---|
| HenrikDev API key | Sent to `api.henrikdev.xyz` over verified TLS | Read from `.env`, which is gitignored. Never logged or printed; `valwr.check` reports only its length. |
| Riot local client | Lockfile password, `127.0.0.1` only | Read-only: GET requests, no writes, no agent selection, no memory access. TLS verification is off for this loopback connection alone, because the client's certificate is self-signed. |
| Collected match data | Other players' match histories | Stays in `data/`, gitignored. No route or command looks up an arbitrary player. |
| Live dashboard | Serves the current match, including other players' gamertags | Binds `127.0.0.1` by default. `LocalOnly` refuses requests addressed to a domain name (DNS rebinding) and websockets opened from any other origin. The page escapes every rendered value and sends a Content-Security-Policy that allows no external script, image, font or connection, and no framing. |
| Phone mode (`phone.bat`, `--host 0.0.0.0`) | The same match data, to your local network | Opt-in, and it says so when it starts. Anything on that network can read the page while it runs, so use it on a network you control. |
| Model files (`models/*.joblib`) | Pickle, which runs code when loaded | Only ever loaded from your own `models/` directory, which you create by training. **Never load a model file someone else sent you.** None is distributed with this repository. |
| Agent and map art | Downloaded from `valorant-api.com` | Filenames are validated before anything is written, so a malformed response cannot write outside the art directories. |
| Public demo (GitHub Pages) | A static copy of the dashboard page | Built from invented players only (`valwr/dash/demo.py`); no database, game client or API is involved. Its content policy allows no connection at all. |
| CI | GitHub Actions | Read-only token, no secrets, and never calls an external API. |

## Not in scope

This is a local tool with no hosted service, accounts or remote endpoint.
Nothing in it is meant to be exposed to the internet, and a deployment that
does so is outside what this project supports.
