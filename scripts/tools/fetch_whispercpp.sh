#!/usr/bin/env bash
# Build the bundled whisper.cpp `whisper-server` binary into vendor/whispercpp/.
#
# Decision D2: Murmur talks HTTP to a bundled whisper-server child process rather
# than linking a Python binding. This script produces that binary.
#
# The pinned tag must stay in step with WHISPER_CPP_TAG in engines/whispercpp.py.
#
# Usage:  bash scripts/tools/fetch_whispercpp.sh
# Output: <repo>/vendor/whispercpp/whisper-server  (path printed on success)

set -euo pipefail

WHISPER_CPP_TAG="${WHISPER_CPP_TAG:-v1.7.5}"
WHISPER_CPP_REPO="${WHISPER_CPP_REPO:-https://github.com/ggml-org/whisper.cpp.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${REPO_ROOT}/vendor/whispercpp"
DEST_BIN="${DEST_DIR}/whisper-server"

for tool in git cmake; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: ${tool} is required but not on PATH" >&2
    exit 1
  fi
done

WORK_DIR="$(mktemp -d -t murmur-whispercpp)"
cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

echo "==> cloning whisper.cpp ${WHISPER_CPP_TAG}" >&2
git clone --depth 1 --branch "${WHISPER_CPP_TAG}" "${WHISPER_CPP_REPO}" "${WORK_DIR}/whisper.cpp" >&2

CMAKE_ARGS=(
  -S "${WORK_DIR}/whisper.cpp"
  -B "${WORK_DIR}/build"
  -DCMAKE_BUILD_TYPE=Release
  -DWHISPER_BUILD_TESTS=OFF
  -DWHISPER_BUILD_EXAMPLES=ON
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

echo "==> building target whisper-server" >&2
cmake --build "${WORK_DIR}/build" --config Release --target whisper-server -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" >&2

BUILT_BIN=""
for candidate in \
  "${WORK_DIR}/build/bin/whisper-server" \
  "${WORK_DIR}/build/bin/Release/whisper-server" \
  "${WORK_DIR}/build/examples/server/whisper-server"; do
  if [ -f "${candidate}" ]; then
    BUILT_BIN="${candidate}"
    break
  fi
done

if [ -z "${BUILT_BIN}" ]; then
  echo "error: whisper-server was not produced by the build" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
cp "${BUILT_BIN}" "${DEST_BIN}"
chmod +x "${DEST_BIN}"

echo "${DEST_BIN}"
