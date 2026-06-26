#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# run_lora.sh  –  Install deps with uv (optional), run lora2.py
#
# Usage:
#   source /path/to/your/venv/bin/activate
#   bash run_lora.sh
#
# If a venv is already active ($VIRTUAL_ENV), it is used as-is — nothing
# is recreated. Dependencies are installed with `uv pip install` into that
# environment unless you set LORA_SKIP_INSTALL=1 (e.g. after first run).
#
# If no venv is active: activates ./venv when it exists, otherwise creates
# ./venv once with `uv venv`.
#
# Requires: uv on PATH — https://docs.astral.sh/uv/
# ══════════════════════════════════════════════════════════════════

set -e

VENV_DIR="venv"
PYTHON="python3"
SCRIPT="lora2.py"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ══════════════════════════════════════════════════════════════════
# 0. Sanity checks
# ══════════════════════════════════════════════════════════════════
info "Checking prerequisites..."

command -v uv &>/dev/null || error "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
command -v $PYTHON &>/dev/null || error "python3 not found. Install it first."

[[ -f "$SCRIPT" ]] || error "$SCRIPT not found in $(pwd). Place this script next to $SCRIPT."

# ── CUDA/GPU check (before venv — uses system tools only) ─────────
if command -v nvidia-smi &>/dev/null; then
    info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warning "No GPU detected. Training will be extremely slow on CPU."
fi

# ══════════════════════════════════════════════════════════════════
# 1. Select virtual environment (never replace an existing one)
# ══════════════════════════════════════════════════════════════════
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "Using your active venv (no create/switch): $VIRTUAL_ENV"
elif [[ -d "$VENV_DIR" ]]; then
    info "Activating existing ./$VENV_DIR ..."
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    info "venv: $VIRTUAL_ENV"
else
    info "No venv active and ./$VENV_DIR missing — creating once with uv ..."
    uv venv "$VENV_DIR" --python "$PYTHON"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    info "venv: $VIRTUAL_ENV"
fi

command -v python &>/dev/null || error "'python' not found inside the venv."

# ── Python version check (interpreter from the venv) ──────────────
PY_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
PY_VERSION="$PY_MAJOR.$PY_MINOR"
info "Python version: $PY_VERSION ($(command -v python))"

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 9 ) ]]; then
    error "Python 3.9+ is required (found $PY_VERSION). Please upgrade Python."
fi

BNB_SUPPORTED=true
if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    warning "Python $PY_VERSION detected."
    warning "bitsandbytes >=0.43.1 requires Python >=3.10 — will install 0.42.0 instead."
    warning "GPU 4-bit training still works; CPU-safe import is NOT available on this version."
    warning "Upgrade to Python 3.10+ when possible to remove this limitation."
    BNB_SUPPORTED=false
fi

# ══════════════════════════════════════════════════════════════════
# 2–3. Install dependencies with uv (into the active venv), optional
# ══════════════════════════════════════════════════════════════════
if [[ "${LORA_SKIP_INSTALL:-}" == "1" ]]; then
    info "LORA_SKIP_INSTALL=1 — skipping uv pip install (using packages already in this venv)."
else
    # Install certifi first so SSL env vars can be set before Hub downloads
    info "Installing / upgrading certifi (SSL certificates)..."
    uv pip install --upgrade certifi --quiet

    # ── Point all SSL/TLS to the venv's own certifi bundle ────────
    CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
    export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
    export SSL_CERT_FILE="$CA_BUNDLE"
    export CURL_CA_BUNDLE="$CA_BUNDLE"
    info "SSL CA bundle set to: $CA_BUNDLE"

    info "Installing requirements with uv into this venv..."

    # ── Detect CUDA ─────────────────────────────────────────────
    CUDA_AVAILABLE=false
    CUDA_MAJOR=0

    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_AVAILABLE=true
        info "nvcc found – CUDA $CUDA_VER detected."
    elif command -v nvidia-smi &>/dev/null; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        info "nvidia-smi found (driver $DRIVER_VER) – treating as CUDA 12."
        CUDA_MAJOR=12
        CUDA_AVAILABLE=true
    else
        warning "No CUDA toolchain found – installing CPU-only PyTorch wheel."
    fi

    # ── PyTorch ───────────────────────────────────────────────────
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        if [[ "$CUDA_MAJOR" -ge 12 ]]; then
            info "Installing torch with CUDA 12.1 support..."
            uv pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu121 --quiet
        elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
            info "Installing torch with CUDA 11.8 support..."
            uv pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu118 --quiet
        else
            warning "Older CUDA (<11). Installing default torch – GPU support not guaranteed."
            uv pip install torch torchvision torchaudio --quiet
        fi
    else
        info "Installing CPU-only torch..."
        uv pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu --quiet
    fi

    # ── bitsandbytes ────────────────────────────────────────────
    if [[ "$BNB_SUPPORTED" == true ]]; then
        info "Installing bitsandbytes >=0.43.1 ..."
        uv pip install "bitsandbytes>=0.43.1" --quiet
        if [[ "$CUDA_AVAILABLE" == false ]]; then
            warning "No GPU found. 4-bit quantisation is not supported on CPU."
            warning "Set load_in_4bit=False in lora2.py when running without a GPU."
        fi
    else
        info "Installing bitsandbytes==0.42.0 (Python 3.9 compatible)..."
        uv pip install "bitsandbytes==0.42.0" --quiet
        if [[ "$CUDA_AVAILABLE" == false ]]; then
            warning "No GPU + Python 3.9: bitsandbytes 0.42.0 will crash on import without CUDA."
            warning "Please upgrade to Python 3.10+ or run on a machine with a GPU."
        fi
    fi

    # trl>=0.22: SFTConfig + processing_class (see lora2.py)
    uv pip install \
        "transformers>=4.40.0" \
        "peft>=0.10.0" \
        "trl>=0.22.0" \
        "accelerate>=0.29.0" \
        "datasets>=2.18.0" \
        pandas \
        --quiet

    info "All requirements installed."
fi

# SSL env for lora2.py when install was skipped (certifi already in venv)
if [[ -z "${REQUESTS_CA_BUNDLE:-}" ]]; then
    CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
    export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
    export SSL_CERT_FILE="$CA_BUNDLE"
    export CURL_CA_BUNDLE="$CA_BUNDLE"
    info "SSL CA bundle set to: $CA_BUNDLE"
fi

# ── Verify imports ────────────────────────────────────────────────
info "Verifying imports..."
python - <<'EOF'
import torch, transformers, peft, trl, bitsandbytes, datasets, pandas
print(f"  torch          {torch.__version__}")
print(f"  transformers   {transformers.__version__}")
print(f"  peft           {peft.__version__}")
print(f"  trl            {trl.__version__}")
print(f"  bitsandbytes   {bitsandbytes.__version__}")
print(f"  datasets       {datasets.__version__}")
print(f"  pandas         {pandas.__version__}")
EOF

# ══════════════════════════════════════════════════════════════════
# 4. Run lora2.py
#    SSL env vars are exported above so Hub downloads use the venv certifi bundle.
# ══════════════════════════════════════════════════════════════════
info "Starting training: python $SCRIPT"
echo "──────────────────────────────────────────────────────────────"

python "$SCRIPT"

echo "──────────────────────────────────────────────────────────────"
info "Training complete. Adapters saved to ./qwen-lora-adapters"