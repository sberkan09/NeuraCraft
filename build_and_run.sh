#!/usr/bin/env bash
# Builds NeuraCraft (incrementally) and runs the demo binary.
# Any arguments are passed to the binary, e.g.:
#   ./build_and_run.sh            # xor + spiral + sine demos
#   ./build_and_run.sh spiral
#   ./build_and_run.sh mnist data/mnist 5
#   ./build_and_run.sh --test     # build and run the test suite instead
set -euo pipefail

cd "$(dirname "$0")"

BUILD_DIRECTORY="./build"
BINARY_PATH="./bin/NeuraCraft"

if command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
fi

cmake -S . -B "$BUILD_DIRECTORY" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIRECTORY" --parallel "$JOBS"

if [ "${1:-}" = "--test" ]; then
    ctest --test-dir "$BUILD_DIRECTORY" --output-on-failure
    exit $?
fi

exec "$BINARY_PATH" "$@"
