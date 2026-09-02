#!/usr/bin/env bash
# Idempotent install / redownload for qwen-agent (Linux and WSL only).
set -euo pipefail

PREFIX="${HOME}/.qwen-agent"
APP_DIR="${PREFIX}/app"
VENV_DIR="${PREFIX}/venv"
CONFIG="${PREFIX}/config.toml"
LOCAL_BIN="${HOME}/.local/bin"
DEFAULT_REPO="${QWEN_AGENT_REPO:-}"
DEFAULT_REF="${QWEN_AGENT_REF:-main}"

die() { echo "qwen-agent install: $*" >&2; exit 1; }
info() { echo "qwen-agent install: $*"; }

# --- platform ---
uname_s="$(uname -s 2>/dev/null || echo unknown)"
if [[ "${OS:-}" == "Windows_NT" ]] || [[ "${uname_s}" == MINGW* ]] || [[ "${uname_s}" == MSYS* ]] || [[ "${uname_s}" == CYGWIN* ]]; then
  die "native Windows is not supported. Run this inside WSL: wsl.exe"
fi
if [[ "${uname_s}" != "Linux" ]]; then
  die "Linux or WSL required (uname -s = Linux). Got: ${uname_s}"
fi

in_wsl=0
if grep -qi microsoft /proc/version 2>/dev/null || [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  in_wsl=1
fi

# --- locate sources ---
SCRIPT_PATH=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
fi

SRC_DIR=""
CLEANUP_TMP=""
if [[ -n "${SCRIPT_PATH}" && -d "$(dirname "${SCRIPT_PATH}")/src/qwen_agent" ]]; then
  SRC_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  info "installing from local tree: ${SRC_DIR}"
elif [[ -n "${DEFAULT_REPO}" ]]; then
  CLEANUP_TMP="$(mktemp -d)"
  info "cloning ${DEFAULT_REPO} (${DEFAULT_REF})"
  git clone --depth 1 --branch "${DEFAULT_REF}" "${DEFAULT_REPO}" "${CLEANUP_TMP}/repo"
  SRC_DIR="${CLEANUP_TMP}/repo"
else
  cat >&2 <<'EOF'
qwen-agent install: no sources found.

From a checkout:
  bash install.sh

From the network (set the git URL first):
  export QWEN_AGENT_REPO=https://github.com/<ORG>/<REPO>.git
  curl -fsSL https://raw.githubusercontent.com/<ORG>/<REPO>/main/install.sh | bash

Native Windows is not supported. Use WSL: wsl.exe
EOF
  exit 1
fi

# --- packages ---
need_cmds=()
need_pkgs=()
have() { command -v "$1" >/dev/null 2>&1; }

pick_python() {
  local c
  for c in python3.14 python3.13 python3.12 python3.11 python3; do
    if have "$c"; then
      if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        echo "$c"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(pick_python || true)"
if [[ -z "${PY}" ]]; then
  need_pkgs+=(python3 python3-venv python3-pip)
fi
have git || { need_cmds+=(git); need_pkgs+=(git); }
have curl || { need_cmds+=(curl); need_pkgs+=(curl); }
want_rg=0
have rg || want_rg=1

ensure_venv_pkg() {
  if [[ -n "${PY}" ]] && ! "${PY}" -c 'import venv, ensurepip' 2>/dev/null; then
    need_pkgs+=(python3-venv python3-pip)
  fi
}
ensure_venv_pkg

apt_install() {
  local pkgs=("$@")
  ((${#pkgs[@]})) || return 0
  if ! have apt-get; then
    return 1
  fi
  if have sudo && [[ -t 0 || -t 1 ]]; then
    info "installing packages: ${pkgs[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${pkgs[@]}"
    return 0
  fi
  return 1
}

if ((${#need_pkgs[@]})); then
  mapfile -t need_pkgs < <(printf '%s\n' "${need_pkgs[@]}" | awk 'NF && !seen[$0]++')
  if ! apt_install "${need_pkgs[@]}"; then
    die "missing ${need_pkgs[*]}. Install with: sudo apt-get install -y ${need_pkgs[*]}"
  fi
fi
if ((want_rg)); then
  apt_install ripgrep || info "ripgrep not installed (optional). grep tool will use a Python fallback."
fi

PY="$(pick_python)" || die "Python 3.11+ is required"

# --- preserve config, wipe app + venv ---
mkdir -p "${PREFIX}/sessions" "${LOCAL_BIN}"
if [[ ! -f "${CONFIG}" ]]; then
  cat > "${CONFIG}" <<'TOML'
[api]
base_url = "http://100.64.11.62:11434/v1"
api_key = "dummy"
model = "qwen3.8:27b"
timeout_s = 300

[ui]
vim_mode = false

[safety]
auto_approve_read = true
# bash/write/edit require y/n unless always-allow this session
TOML
  info "wrote ${CONFIG}"
else
  info "keeping existing ${CONFIG}"
fi

info "replacing ${APP_DIR} and ${VENV_DIR}"
rm -rf "${APP_DIR}" "${VENV_DIR}"
mkdir -p "${APP_DIR}"

# copy project files (not .git, venv)
if have rsync; then
  rsync -a --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.egg-info/' \
    "${SRC_DIR}/" "${APP_DIR}/"
else
  tar -C "${SRC_DIR}" --exclude '.git' --exclude '.venv' --exclude 'venv' -cf - . \
    | tar -C "${APP_DIR}" -xf -
fi

"${PY}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${APP_DIR}"

ln -sfn "${VENV_DIR}/bin/qwen-agent" "${LOCAL_BIN}/qwen-agent"

if [[ -n "${CLEANUP_TMP}" ]]; then
  rm -rf "${CLEANUP_TMP}"
fi

cat <<EOF

Installed.

Next steps:
  1. Ensure ~/.local/bin is on PATH (this shell):
       export PATH="\$HOME/.local/bin:\$PATH"
     Add that line to ~/.bashrc if it is missing.
  2. Edit the API endpoint if needed:
       nano ${CONFIG}
  3. Check connectivity:
       qwen-agent doctor
  4. Work in a Linux/WSL repo (prefer \$HOME/src/..., not /mnt/c/...):
       cd ~/src/my-repo
       qwen-agent

Redownload / repair: run this installer again. config.toml is kept.
EOF

if ((in_wsl)); then
  cat <<'EOF'

WSL: run from a WSL shell, not PowerShell. Clone and edit under /home/..., not /mnt/c (I/O is slow there).
EOF
fi
