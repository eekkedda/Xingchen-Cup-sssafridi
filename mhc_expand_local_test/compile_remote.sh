#!/usr/bin/env bash
# CANN-version-agnostic build script for remote 910B machines (CANN 9.x).
# Usage: bash compile_remote.sh [BUILD_DIR]   (from mhc_expand_local_test/)
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../B_easy" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/build_remote}"

# Locate CANN regardless of conda layout.
if [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    for setup_file in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/cann/set_env.sh" \
        "${HOME}/Ascend/ascend-toolkit/set_env.sh" \
        "${HOME}/Ascend/cann/set_env.sh" \
        "${CONDA_PREFIX:-/nonexistent}/set_env.sh" \
        "${CONDA_PREFIX:-/nonexistent}/Ascend/cann/set_env.sh"; do
        if [[ -f "${setup_file}" ]]; then
            set +u; source "${setup_file}"; set -u; break
        fi
    done
fi
if [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" && -n "${ASCEND_HOME_PATH:-}" ]]; then
    export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
fi
if [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    echo "ERROR: CANN environment not found (source its set_env.sh first)." >&2
    exit 2
fi

CANN_CMAKE_DIR="${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
if [[ ! -d "${CANN_CMAKE_DIR}" ]]; then
    echo "ERROR: ASC CMake not found under ${ASCEND_CANN_PACKAGE_PATH}" >&2
    exit 2
fi

version_text=""
for version_file in \
    "${ASCEND_CANN_PACKAGE_PATH}/version.info" \
    "${ASCEND_CANN_PACKAGE_PATH}/version.cfg" \
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info"; do
    if [[ -f "${version_file}" ]]; then
        version_text="$(tr '\n' ' ' < "${version_file}")"; break
    fi
done
echo "CANN path : ${ASCEND_CANN_PACKAGE_PATH}"
[[ -z "${version_text}" ]] || echo "CANN info : ${version_text}"
echo "NOTE: competition judge uses CANN 8.5.0; this build is for LOCAL real-HW"
echo "      experiments only. Final submissions must still pass the 8.5 judge."

export ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH:-${ASCEND_CANN_PACKAGE_PATH}}"
export CMAKE_PREFIX_PATH="${CANN_CMAKE_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

command -v cmake >/dev/null || { echo "ERROR: cmake >= 3.16 required." >&2; exit 2; }

mkdir -p "${BUILD_DIR}"
{
    cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
        -DASC_DIR="${CANN_CMAKE_DIR}" \
        -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
        -DASCEND_COMPUTE_UNIT=ascend910b
    cmake --build "${BUILD_DIR}" --target binary --parallel "$(nproc)"
    cmake --build "${BUILD_DIR}" --target install --parallel "$(nproc)"
} 2>&1 | tee "${BUILD_DIR}/build.log"
echo "Remote build finished: ${BUILD_DIR}"
