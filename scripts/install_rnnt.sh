# #!/usr/bin/env bash
# set -euo pipefail

# export CUDA_HOME=/usr/local/cuda-12.1 # change to your path
# export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
# # export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
# # export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# # export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
# # export PATH=${CUDA_HOME}/bin:${PATH}
# export CMAKE_CUDA_ARCHITECTURES=86


# # # Safely append to LIBRARY_PATH and LD_LIBRARY_PATH if they exist
# # export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
# # export LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH:-}

# export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS:-}"
# # export CMAKE_CUDA_ARCHITECTURES=75


# if [ $# -eq 0 ]; then
#     with_openmp=ON
# elif [ $# -eq 1 ]; then
#     with_openmp=$1
# elif [ $# -gt 1 ]; then
#     echo "Usage: $0 [with_openmp| ON or OFF]"
#     exit 1;
# fi
# unames="$(uname -s)"
# if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
#     echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
#     exit 0
# fi


# if ! python -c "import packaging.version" &> /dev/null; then
#     python3 -m pip install packaging
# fi
# # TODO(kamo): Consider clang case
# # Note: Requires gcc>=4.9.2 to build extensions with pytorch>=1.0
# if python3 -c 'import torch as t;assert t.__version__[0] == "2"' &> /dev/null; then \
#     python3 -c "from packaging.version import parse as V;assert V('$(gcc -dumpversion)') >= V('4.9.2'), 'Requires gcc>=4.9.2'"; \
# fi
# cd extras
# rm -rf warp-transducer
# git clone --single-branch --branch update_torch2.1 https://github.com/b-flo/warp-transducer.git

# (
#     set -euo pipefail
#     cd warp-transducer

#     mkdir build
#     (
#         set -euo pipefail
#         cd build && cmake -DWITH_OMP="${with_openmp}" .. && make
#     )

#     (
#         set -euo pipefail
#         cd pytorch_binding && python3 -m pip install -e .
#     )
# )

#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 1) Thiết lập các biến môi trường CUDA
# =============================================================================
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS:-}"
export LDFLAGS="-L${CUDA_HOME}/lib64 ${LDFLAGS:-}"

# =============================================================================
# 2) Xử lý tham số đầu vào (bật/tắt OpenMP)
# =============================================================================
if [ $# -eq 0 ]; then
    with_openmp=ON
elif [ $# -eq 1 ]; then
    with_openmp=$1
else
    echo "Usage: $0 [with_openmp| ON or OFF]"
    exit 1
fi

# =============================================================================
# 3) Kiểm tra hệ điều hành (Linux / macOS)
# =============================================================================
unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work on ${unames}. Exit."
    exit 0
fi

# =============================================================================
# 4) Đảm bảo đã cài gói 'packaging' (cần cho so sánh phiên bản)
# =============================================================================
if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi

# =============================================================================
# 5) Kiểm tra GCC >= 4.9.2 nếu đang dùng PyTorch 2.x
# =============================================================================
if python3 -c 'import torch as t; assert t.__version__[0] == "2"' &> /dev/null; then
    python3 -c "from packaging.version import parse as V; \
                assert V('$(gcc -dumpversion)') >= V('4.9.2'), 'Requires gcc>=4.9.2'"
fi

# =============================================================================
# 6) Tải mã nguồn warp-transducer (branch update_torch2.1) & build
# =============================================================================
cd extras
rm -rf warp-transducer
git clone --single-branch --branch update_torch2.1 \
    https://github.com/b-flo/warp-transducer.git

(
    set -euo pipefail
    cd warp-transducer

    mkdir build
    (
        set -euo pipefail
        cd build
        cmake -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DWITH_OMP="${with_openmp}" .. 
        make
    )

    (
        set -euo pipefail
        cd pytorch_binding
        python3 -m pip install -e .
    )
)

