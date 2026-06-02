#!/usr/bin/env bash
set -euo pipefail

# Start AVO in interactive mode and inject the main-agent prompt through tmux.
# AVO reads TASK/hooks from TARGET/.n3 because --working-dir points at TARGET.

AVO_ROOT="${AVO_ROOT:-/home/xutingz/gitsrc/avo}"
AVO_ENV="${AVO_ENV:-${AVO_ROOT}/.venv}"
TARGET="${TARGET:-/home/xutingz/gitsrc/DeepGEMM}"
PROMPT_FILE="${PROMPT_FILE:-${TARGET}/.n3/PROMPT.md}"
AVO_MODEL_PRESET="${AVO_MODEL_PRESET:-gpt55}"
if [[ -z "${AVO_MODEL:-}" ]]; then
  case "${AVO_MODEL_PRESET}" in
    gpt55|gpt-5.5|codex55|codex-5.5)
      AVO_MODEL="nvinf/azure/openai/gpt-5.5"
      ;;
    claude46|claude-4.6|opus46|opus-4.6)
      AVO_MODEL="nvinf/aws/anthropic/bedrock-claude-opus-4-6"
      ;;
    claude47|claude-4.7|opus47|opus-4.7)
      AVO_MODEL="nvinf/aws/anthropic/bedrock-claude-opus-4-7"
      ;;
    claude48|claude-4.8|opus48|opus-4.8)
      AVO_MODEL="nvinf/aws/anthropic/bedrock-claude-opus-4-8"
      ;;
    *)
      echo "Unknown AVO_MODEL_PRESET: ${AVO_MODEL_PRESET}" >&2
      echo "Use gpt55, claude46, claude47, claude48, or set AVO_MODEL to a full model id." >&2
      exit 1
      ;;
  esac
fi
SESSION="${SESSION:-avo-deepgemm-g1g2}"
LOG="${LOG:-${TARGET}/avo/avo-deepgemm-g1g2.log}"
ENABLE_BACKGROUND_AGENT="${ENABLE_BACKGROUND_AGENT:-true}"
BG_AGENT_LOGS="${BG_AGENT_LOGS:-true}"

if [[ ! -x "${AVO_ENV}/bin/n3" ]]; then
  echo "n3 not found or not executable: ${AVO_ENV}/bin/n3" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

if [[ ("${AVO_MODEL}" == nvinf/* || "${AVO_MODEL}" == azure/openai/*) && -z "${NVINF_ACCESS_TOKEN:-}" ]]; then
  OPENCODE_CONFIG="${OPENCODE_CONFIG:-${HOME}/.config/opencode/opencode.json}"
  if [[ -f "${OPENCODE_CONFIG}" ]]; then
    eval "$("${AVO_ENV}/bin/python" - "${OPENCODE_CONFIG}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
opts = cfg.get("provider", {}).get("nvinf", {}).get("options", {})
api_key = opts.get("apiKey")
base_url = opts.get("baseURL")
if api_key:
    print("export NVINF_ACCESS_TOKEN=" + shlex.quote(api_key))
    print("export NVINF_API_KEY=${NVINF_ACCESS_TOKEN}")
if base_url:
    print("export NVINF_BASE_URL=" + shlex.quote(base_url))
PY
    )"
  fi
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG}")"

tmux new-session -d -s "${SESSION}" "
  cd \"${AVO_ROOT}\" &&
  export SHELL=/bin/bash ENABLE_BACKGROUND_AGENT=\"${ENABLE_BACKGROUND_AGENT}\" BG_AGENT_LOGS=\"${BG_AGENT_LOGS}\" &&
  \"${AVO_ENV}/bin/n3\" --model \"${AVO_MODEL}\" --working-dir \"${TARGET}\"
"

tmux pipe-pane -o -t "${SESSION}" "cat >> '${LOG}'"
sleep "${AVO_PROMPT_DELAY:-5}"
tmux load-buffer "${PROMPT_FILE}"
tmux paste-buffer -t "${SESSION}"
tmux send-keys -t "${SESSION}" C-m

cat <<EOF
Started AVO session: ${SESSION}
Target repo: ${TARGET}
Prompt file: ${PROMPT_FILE}
Model preset: ${AVO_MODEL_PRESET}
Model: ${AVO_MODEL}
Log file: ${LOG}

Monitor:
  tmux capture-pane -pt ${SESSION} -S -80
  tail -f ${LOG}
  tmux attach -t ${SESSION}
EOF
