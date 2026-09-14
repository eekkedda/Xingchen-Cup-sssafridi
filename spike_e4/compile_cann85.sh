#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../B_easy" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/build_cann85}"

if [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    for setup_file in \
        "/usr/local/Ascend/cann/set_env.sh" \
        "${HOME}/Ascend/cann/set_env.sh" \
        "${CONDA_PREFIX:-/nonexistent}/set_env.sh" \
        "${CONDA_PREFIX:-/nonexistent}/Ascend/cann/set_env.sh"; do
        if [[ -f "${setup_file}" ]]; then
            # shellcheck disable=SC1090
            # CANN's environment script reads unset variables directly.
            set +u
            source "${setup_file}"
            set -u
            break
        fi
    done
fi

if [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" && -n "${ASCEND_HOME_PATH:-}" ]]; then
    export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
elif [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" && -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    export ASCEND_CANN_PACKAGE_PATH="${ASCEND_TOOLKIT_HOME}"
elif [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" && -n "${ASCEND_OPP_PATH:-}" ]]; then
    export ASCEND_CANN_PACKAGE_PATH="$(dirname -- "${ASCEND_OPP_PATH}")"
fi

if [[ -z "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    echo "ERROR: CANN environment is not active." >&2
    echo "Install/activate CANN 8.5.0, then source its set_env.sh." >&2
    exit 2
fi

command -v cmake >/dev/null || {
    echo "ERROR: cmake >= 3.16 is required." >&2
    exit 2
}

CANN_CMAKE_DIR="${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
if [[ ! -d "${CANN_CMAKE_DIR}" ]]; then
    echo "ERROR: ASC CMake package was not found under:" >&2
    echo "       ${CANN_CMAKE_DIR}" >&2
    echo "Check that ASCEND_CANN_PACKAGE_PATH points to the CANN 8.5 root." >&2
    exit 2
fi

version_text=""
for version_file in \
    "${ASCEND_CANN_PACKAGE_PATH}/version.info" \
    "${ASCEND_CANN_PACKAGE_PATH}/version.cfg" \
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info"; do
    if [[ -f "${version_file}" ]]; then
        version_text="$(tr '\n' ' ' < "${version_file}")"
        break
    fi
done
if [[ -n "${version_text}" && "${version_text}" != *"8.5.0"* ]]; then
    echo "ERROR: the active CANN installation does not report version 8.5.0:" >&2
    echo "       ${version_text}" >&2
    exit 2
fi

export ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH:-${ASCEND_CANN_PACKAGE_PATH}}"
export CMAKE_PREFIX_PATH="${CANN_CMAKE_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# A Conda host compiler links generated helper libraries against Conda's
# libstdc++. Keep that runtime ahead of the system copy when using this setup.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DEVLIB_DIR="${ASCEND_CANN_PACKAGE_PATH}/$(uname -m)-linux/devlib"
if [[ -d "${DEVLIB_DIR}" ]]; then
    # The stub directory contains duplicate CANN libraries. Append it so the
    # real toolkit libraries initialized by set_env.sh keep precedence.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${DEVLIB_DIR}"
fi

# A fully user-local Conda setup may not have libc development headers under
# /usr/include. Bisheng's precompile pass still needs its bundled C/C++ headers.
HCC_DIR="${ASCEND_CANN_PACKAGE_PATH}/tools/hcc"
if [[ ! -f /usr/include/stdio.h && -d "${HCC_DIR}/sysroot/usr/include" ]]; then
    HCC_CXX_DIR="${HCC_DIR}/aarch64-target-linux-gnu/include/c++/7.3.0"
    export CPATH="${HCC_CXX_DIR}:${HCC_CXX_DIR}/aarch64-target-linux-gnu:${HCC_DIR}/sysroot/usr/include${CPATH:+:${CPATH}}"
fi
mkdir -p "${BUILD_DIR}"

echo "CANN path : ${ASCEND_CANN_PACKAGE_PATH}"
[[ -z "${version_text}" ]] || echo "CANN info : ${version_text}"
echo "Project   : ${PROJECT_DIR}"
echo "Build dir : ${BUILD_DIR}"
echo "Target    : ascend910b / binary"

{
    cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
        -DASC_DIR="${CANN_CMAKE_DIR}" \
        -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
        -DASCEND_COMPUTE_UNIT=ascend910b
    cmake --build "${BUILD_DIR}" --target binary --parallel "$(nproc)"
    cmake --build "${BUILD_DIR}" --target install --parallel "$(nproc)"
} 2>&1 | tee "${BUILD_DIR}/build.log"

{
    echo "CANN 8.5 binary build and install targets succeeded."
    echo "Build log: ${BUILD_DIR}/build.log"
} | tee -a "${BUILD_DIR}/build.log"
