#!/usr/bin/env bash
# Build the bundled llama.cpp `llama-server` binary into vendor/llamacpp/.
#
# Decision D3: local cleanup runs on a ~3B GGUF served by a bundled
# llama-server child process, spoken to over the OpenAI-compatible chat route.
# This script produces that binary.
#
# The pinned tag must stay in step with LLAMA_CPP_TAG in cleanup/llama_server.py.
#
# Usage:  bash scripts/tools/fetch_llama.sh
# Output: <repo>/vendor/llamacpp/llama-server  (path printed on success)

set -euo pipefail

LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-v0.3.0}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${REPO_ROOT}/vendor/llamacpp"
DEST_BIN="${DEST_DIR}/llama-server"

for tool in git cmake; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: ${tool} is required but not on PATH" >&2
    exit 1
  fi
done

WORK_DIR="$(mktemp -d -t murmur-llamacpp)"
cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

echo "==> cloning llama.cpp ${LLAMA_CPP_TAG}" >&2
git clone --depth 1 --branch "${LLAMA_CPP_TAG}" "${LLAMA_CPP_REPO}" "${WORK_DIR}/llama.cpp" >&2

# LLAMA_BUILD_UI=OFF keeps the embedded Web UI (and its Node toolchain) out of
# the build. The llama-ui target still exists and links, it just carries no
# assets — which is what we want, since the app never opens that UI and
# `--no-webui` disables the route anyway.
CMAKE_ARGS=(
  -S "${WORK_DIR}/llama.cpp"
  -B "${WORK_DIR}/build"
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_UI=OFF
  -DBUILD_SHARED_LIBS=OFF
)

if [ "$(uname -m)" = "arm64" ]; then
  echo "==> arm64 detected, enabling Metal" >&2
  CMAKE_ARGS+=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
else
  CMAKE_ARGS+=(-DGGML_METAL=OFF)
fi

echo "==> configuring" >&2
cmake "${CMAKE_ARGS[@]}" >&2

echo "==> building target llama-server" >&2
cmake --build "${WORK_DIR}/build" --config Release --target llama-server -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" >&2

BUILT_BIN=""
for candidate in \
  "${WORK_DIR}/build/bin/llama-server" \
  "${WORK_DIR}/build/bin/Release/llama-server" \
  "${WORK_DIR}/build/tools/server/llama-server"; do
  if [ -f "${candidate}" ]; then
    BUILT_BIN="${candidate}"
    break
  fi
done

if [ -z "${BUILT_BIN}" ]; then
  echo "error: llama-server was not produced by the build" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
cp "${BUILT_BIN}" "${DEST_BIN}"
chmod +x "${DEST_BIN}"

echo "${DEST_BIN}"
