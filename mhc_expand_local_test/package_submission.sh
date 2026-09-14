#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../B_easy" && pwd)"
OUTPUT_PATH="${1:-${SCRIPT_DIR}/mhc_expand_submission_$(date +%Y%m%d_%H%M%S).zip}"

if ! command -v zip >/dev/null && ! command -v python3 >/dev/null; then
    echo "ERROR: either zip or python3 is required." >&2
    exit 2
fi

if [[ -e "${OUTPUT_PATH}" ]]; then
    echo "ERROR: refusing to overwrite existing file: ${OUTPUT_PATH}" >&2
    exit 2
fi

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf -- "${STAGING_DIR}"' EXIT

mkdir -p "${STAGING_DIR}/code/op_host" "${STAGING_DIR}/code/op_kernel"
cp -- "${PROJECT_DIR}/CMakeLists.txt" "${STAGING_DIR}/code/"
cp -- "${PROJECT_DIR}/op_host/CMakeLists.txt" \
    "${PROJECT_DIR}/op_host/mhc_expand.cpp" \
    "${STAGING_DIR}/code/op_host/"
cp -- "${PROJECT_DIR}/op_kernel/CMakeLists.txt" \
    "${PROJECT_DIR}/op_kernel/mhc_expand.cpp" \
    "${PROJECT_DIR}/op_kernel/mhc_expand_tiling.h" \
    "${PROJECT_DIR}/op_kernel/tiling_key_mhc_expand.h" \
    "${STAGING_DIR}/code/op_kernel/"

OUTPUT_DIR="$(dirname -- "${OUTPUT_PATH}")"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"
OUTPUT_FILE="${OUTPUT_DIR}/$(basename -- "${OUTPUT_PATH}")"

(
    cd -- "${STAGING_DIR}"
    if command -v zip >/dev/null; then
        zip -qr "${OUTPUT_FILE}" code
    else
        python3 -m zipfile -c "${OUTPUT_FILE}" code
    fi
)

echo "Submission archive created: ${OUTPUT_FILE}"
echo "Archive contents:"
if command -v unzip >/dev/null; then
    unzip -l "${OUTPUT_FILE}"
else
    python3 -m zipfile -l "${OUTPUT_FILE}"
fi
