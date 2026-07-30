FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
COPY scripts/install_deps.sh /tmp/install_deps.sh
RUN sh /tmp/install_deps.sh

# 2. Install "TheRock" ROCm SDK (Tarball Method)
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
# We pass ARGs to the script via ENV or rely on defaults. 
# But let's be explicit and export them for the RUN command.
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
  export ROCM_MAJOR_VER=$ROCM_MAJOR_VER && \
  export GFX=$GFX && \
  /tmp/install_rocm_sdk.sh

# 4. Python Venv Setup
RUN /usr/bin/python3.12 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONNOUSERSITE=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 5. Install PyTorch (TheRock Nightly)
# Pin to known good version
ARG TORCH_ROCM_VERSION=2.13.0a0+rocm7.14.0a20260608
RUN python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ \
  --pre "torch==${TORCH_ROCM_VERSION}" torch torchaudio torchvision && \
  (find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
  rm -rf /root/.cache/pip

WORKDIR /opt

COPY scripts/patch_aiter_headers.py /opt/patch_aiter_headers.py
RUN python -m pip install --upgrade cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11 numba scipy

# AWQ support: conch-triton-kernels for AWQMarlin-on-ROCm (PR #36505).
RUN python -m pip install conch-triton-kernels==1.3

# Flash-Attention & AITER
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" \
    VLLM_USE_TRITON_AWQ=1
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH"

RUN git clone https://github.com/ROCm/flash-attention.git && \
  cd flash-attention && \
  git checkout main_perf && \
  git submodule update --init third_party/aiter && \
  cd third_party/aiter && \
  git submodule update --init 3rdparty/composable_kernel && \
  export CK_DIR="$(pwd)/3rdparty/composable_kernel" && \
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist -v . && \
  python -m pip install --force-reinstall /tmp/dist/amd_aiter*.whl && \
  python /opt/patch_aiter_headers.py && \
  cd /opt/flash-attention && \
  python -c "import re; f=open('setup.py','r'); t=f.read(); f.close(); t=re.sub(r'subprocess\.run\([\s\S]*?third_party/aiter[\s\S]*?check=True,\s*\)', 'pass # patched', t); f=open('setup.py','w'); f.write(t)" && \
  pip install --no-build-isolation --no-deps . && \
  cd /opt && rm -rf /opt/flash-attention /opt/patch_aiter_headers.py && \
  (find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
  rm -rf /root/.cache/pip

# Fix Fedora lib vs lib64 split: setup.py install writes to lib/, pip to lib64/.
# flash-attention's find_packages() may install a partial aiter copy into lib/.
# Merge any straggler files from lib/ into lib64/ so Python finds everything.
# When lib64 is a symlink to lib (Fedora's default venv layout), the two
# site-packages dirs resolve to the same path — skip the merge, since the
# cp-into-self then rm-rf would delete the entire aiter package.
RUN lib_sp=/opt/venv/lib/python3.12/site-packages; \
  lib64_sp=/opt/venv/lib64/python3.12/site-packages; \
  if [ "$(readlink -f "$lib_sp")" != "$(readlink -f "$lib64_sp")" ] && \
  [ -d "$lib_sp/aiter" ]; then \
  cp -rn "$lib_sp/aiter/"* "$lib64_sp/aiter/" 2>/dev/null || true; \
  rm -rf "$lib_sp/aiter"; \
  fi

# 6. Clone vLLM
# Track HEAD for latest fixes. Override with --build-arg VLLM_COMMIT=<sha>.
ARG VLLM_COMMIT=
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm
RUN if [ -n "$VLLM_COMMIT" ]; then \
  echo "Pinning vLLM to commit $VLLM_COMMIT" && git checkout "$VLLM_COMMIT"; \
  else echo "Tracking vLLM HEAD: $(git rev-parse --short HEAD)"; fi

# --- PATCHING ---
COPY scripts/patch_strix.py /opt/vllm/patch_strix.py
RUN python /opt/vllm/patch_strix.py

# --- FP8 (W8A8) Strix Halo Triton kernels (EXPERIMENTAL / RFC — see issue #67) ---
# Custom FP8 kernels by @leonyurko for gfx1151 (which has no native FP8). The kernel
# modules live on PYTHONPATH (/opt/fp8); patch_fp8_kernels.py routes vLLM's
# compressed-tensors W8A8-FP8 scaled-mm path through the fused Triton dequant-GEMM
# (fp8_triton.fp8_gemm). Kept separate from patch_strix.py so it stays independent
# of the is_integrated memory work. Serve FP8 models with VLLM_ROCM_USE_AITER=0 and
# --enforce-eager (see the kernel repo's serve scripts).
# https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support
ARG FP8_KERNELS_REF=50424f5525b8382353551e3301d0da56eca0be2b
RUN git clone https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support.git /opt/fp8 && \
  cd /opt/fp8 && git checkout "$FP8_KERNELS_REF"
COPY scripts/patch_fp8_kernels.py /opt/vllm/patch_fp8_kernels.py
RUN python /opt/vllm/patch_fp8_kernels.py
ENV PYTHONPATH=/opt/fp8

# 7. Build vLLM (Wheel Method) with CLANG Host Compiler
ENV ROCM_HOME="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV VLLM_TARGET_DEVICE="rocm"
ENV PYTORCH_ROCM_ARCH="gfx1151"
ENV HIP_ARCHITECTURES="gfx1151"          
ENV AMDGPU_TARGETS="gfx1151"              
ENV MAX_JOBS="4"

# --- CRITICAL FIX FOR SEGFAULT ---
# We force the Host Compiler (CC/CXX) to be the ROCm Clang, not Fedora GCC.
# This aligns the ABI of the compiled vLLM extensions with PyTorch.
ENV CC="/opt/rocm/llvm/bin/clang"
ENV CXX="/opt/rocm/llvm/bin/clang++"

# Recent vLLM main ships PyO3/Rust extension modules (vllm/_rust_*.so) for the
# tool-call/reasoning parsers and tokenizer helpers. Building the wheel now
# requires a Rust toolchain plus the setuptools-rust backend, otherwise the
# build fails in metadata prep with "ModuleNotFoundError: No module named
# 'setuptools_rust'". Installed right before the vLLM build so the expensive
# flash-attention/AITER layers above stay cacheable.
RUN dnf install -y rust cargo && dnf clean all && rm -rf /var/cache/dnf/* && \
  python -m pip install "setuptools-rust>=1.9.0" && rm -rf /root/.cache/pip

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  echo "Compiling with Bitcode: $HIP_DEVICE_LIB_PATH" && \
  export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151" && \   
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist -v . && \
  python -m pip install /tmp/dist/*.whl && \
  rm -rf /tmp/dist && \
  (find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
  rm -rf /root/.cache/pip

RUN python -m pip install ray

# --- bitsandbytes ---
# Official bitsandbytes repo (bitsandbytes-foundation) with proper gfx1151 support.
WORKDIR /opt
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /opt/bitsandbytes
WORKDIR /opt/bitsandbytes
RUN cmake -S . \
    -DCOMPUTE_BACKEND=hip \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBNB_ROCM_ARCH="gfx1151" \
    -DGPU_TARGETS="gfx1151" \
    && \
  make -j$(nproc) && \
  python -m pip install --no-cache-dir . --no-build-isolation --no-deps && \
  (find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
  rm -rf /root/.cache/pip

# 8. Final Cleanup & Runtime
WORKDIR /opt
RUN (find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
  find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
  rm -rf /root/.cache/pip || true && \
  dnf clean all && rm -rf /var/cache/dnf/*

COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
COPY scripts/start_vllm.py /opt/start-vllm
COPY scripts/start_vllm_cluster.py /opt/start-vllm-cluster
COPY scripts/measure_bandwidth.sh /opt/measure_bandwidth.sh
COPY scripts/cluster_manager.py /opt/cluster_manager.py
COPY scripts/models.py /opt/models.py

COPY benchmarks/max_context_results.json /opt/max_context_results.json
COPY benchmarks/bench_utils.py /opt/bench_utils.py
COPY benchmarks/run_vllm_bench.py /opt/run_vllm_bench.py
COPY benchmarks/vllm_cluster_bench.py /opt/vllm_cluster_bench.py
COPY benchmarks/find_max_context.py /opt/find_max_context.py
COPY rdma_cluster/compare_eth_vs_rdma.sh /opt/compare_eth_vs_rdma.sh
COPY scripts/configure_cluster.sh /opt/configure_cluster.sh
RUN chmod +x /opt/configure_cluster.sh

RUN chmod +x /opt/start-vllm /opt/start-vllm-cluster /opt/vllm_cluster_bench.py /opt/compare_eth_vs_rdma.sh /opt/find_max_context.py /opt/run_vllm_bench.py && \
  ln -s /opt/start-vllm /usr/local/bin/start-vllm && \
  ln -s /opt/start-vllm-cluster /usr/local/bin/start-vllm-cluster && \
  chmod 0644 /etc/profile.d/*.sh /opt/max_context_results.json /opt/models.py
RUN chmod 0644 /etc/profile.d/*.sh
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh


CMD ["/bin/bash"]
