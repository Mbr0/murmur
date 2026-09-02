#!/usr/bin/env bash
# Run the Murmur integration suite.
#
# The integration tests live in tests/integration/ and are discovered
# separately from the unit suite, so `python -m unittest discover -s tests`
# stays fast and needs no models, no binaries and no network.
#
# Every test skips with an explicit reason when the runtime it needs is
# absent, so this script is safe to run on a fresh checkout: it will just
# report four skips. To exercise them for real, build the binaries
#
#   bash scripts/tools/fetch_whispercpp.sh   # vendor/whispercpp/whisper-server
#   bash scripts/tools/fetch_llama.sh        # vendor/llamacpp/llama-server
#
# and install the models from Settings -> Speech engine (or with
# engines.model_store.ModelStore().download(<model id>)).
#
# Environment:
#   MURMUR_WHISPER_SERVER   path to whisper-server; set from vendor/ when unset
#   MURMUR_LLAMA_SERVER     path to llama-server;   set from vendor/ when unset
#   MURMUR_PYTHON           interpreter to use; defaults to venv/bin/python
#
# Any extra arguments are passed straight to `python -m unittest` (e.g. -v).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Point the engines at the binaries this checkout built, unless the caller
# already chose one. An absent binary is left unset on purpose: the tests turn
# that into a skip with a build instruction, which is a better message than a
# path that does not exist.
if [ -z "${MURMUR_WHISPER_SERVER:-}" ] && [ -x "${REPO_ROOT}/vendor/whispercpp/whisper-server" ]; then
  export MURMUR_WHISPER_SERVER="${REPO_ROOT}/vendor/whispercpp/whisper-server"
fi
if [ -z "${MURMUR_LLAMA_SERVER:-}" ] && [ -x "${REPO_ROOT}/vendor/llamacpp/llama-server" ]; then
  export MURMUR_LLAMA_SERVER="${REPO_ROOT}/vendor/llamacpp/llama-server"
fi

PYTHON="${MURMUR_PYTHON:-}"
if [ -z "${PYTHON}" ]; then
  if [ -x "${REPO_ROOT}/venv/bin/python" ]; then
    PYTHON="${REPO_ROOT}/venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "Python:         ${PYTHON}"
echo "whisper-server: ${MURMUR_WHISPER_SERVER:-<not built, whisper.cpp tests will skip>}"
echo "llama-server:   ${MURMUR_LLAMA_SERVER:-<not built, cleanup tests will skip>}"
echo

cd "${REPO_ROOT}"
exec "${PYTHON}" -m unittest discover -s tests/integration -p "test_*.py" "$@"
