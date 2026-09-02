# qwen-agent

Local Grok-style coding agent TUI. **Your laptop holds the CLI, git repos, and session logs. A friend’s GPU host only runs Qwen3.8-27B over an OpenAI-compatible `/v1` API.**

No project files are copied to the server. Each turn the CLI sends the last ~30 chat messages (plus tool results you already approved). That is enough; you do **not** need a session-cache container on the GPU host.

Linux and WSL (Ubuntu) only. Not a Windows-native app.

## Split of responsibilities

| Where | What |
|-------|------|
| **Your machine** | `qwen-agent` TUI, git working tree, `~/.qwen-agent/config.toml`, `~/.qwen-agent/sessions/` |
| **GPU host (`portal8k`)** | Ollama / llama.cpp / vLLM serving `qwen3.8:27b` on Tailscale |
| **Not used** | Remote session store, remote checkouts, `10.40.0.10` LAN URLs |

The remote API is stateless. A Docker container on the server is only useful to *run the model* (Ollama/vLLM), not to cache your chats.

## Friends (clients)

Inside **Ubuntu or WSL** (not PowerShell):

```bash
# from a git checkout
bash install.sh

# or from the network (set the repo first)
export QWEN_AGENT_REPO=https://github.com/<ORG>/<REPO>.git
curl -fsSL https://raw.githubusercontent.com/<ORG>/<REPO>/main/install.sh | bash
```

Redownload / repair anytime: run the same command again. `~/.qwen-agent/config.toml` and `sessions/` are kept; app + venv are replaced.

```bash
export PATH="$HOME/.local/bin:$PATH"
nano ~/.qwen-agent/config.toml   # Tailscale base_url
qwen-agent doctor
cd ~/src/my-repo
qwen-agent
```

Env overrides (optional): `QWEN_BASE_URL`, `QWEN_API_KEY`, `QWEN_MODEL`.

Default config (`~/.qwen-agent/config.toml`):

```toml
[api]
base_url = "http://100.64.11.62:11434/v1"
api_key = "dummy"
model = "qwen3.8:27b"
timeout_s = 300

[ui]
vim_mode = false

[safety]
auto_approve_read = true
```

## WSL

```powershell
wsl --install -d Ubuntu
wsl
```

Then `install.sh` inside that distro. Clone and work under `~/src/...`, **not** `/mnt/c` (that mount is slow). Tools (`bash`, git, tests) run in WSL, so they are real Linux.

## GPU host (friend who runs Qwen)

Bind the API to all interfaces or the Tailscale IP — **not** `127.0.0.1` and **not only** `10.40.0.10`.

Ollama:

```bash
# e.g. systemd drop-in
Environment=OLLAMA_HOST=0.0.0.0:11434
ollama pull qwen3.8:27b    # redownload weights if needed
```

llama.cpp is often `:8080/v1`. Point clients at `http://<tailscale-ip>:8080/v1`.

Tailscale ACL: allow each friend’s user onto that port, e.g. `portal8k:11434` (and `:8080` if needed). Do **not** put friend repos on this host.

Optional: run Ollama/vLLM in Docker on the GPU host. That container is the **model runtime**, not a place to store other people’s sessions.

## Keys

- Enter — send
- Esc then Enter — newline
- Ctrl+C — cancel in-flight; twice to quit
- Ctrl+D or `/quit` — exit
- `/help` `/clear` `/cwd` `/model` `/doctor`
- write / edit / bash: `[y]` once, `[a]` always this session, `[n]` skip

## Layout

```
qwen-agent/
  README.md
  install.sh
  requirements.txt
  pyproject.toml
  .env.example
  src/qwen_agent/
```

Installed copy lives in `~/.qwen-agent/app` with a venv at `~/.qwen-agent/venv`. The `qwen-agent` script is symlinked to `~/.local/bin/qwen-agent`.
