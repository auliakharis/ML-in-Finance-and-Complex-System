#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# run_lora2.sh  –  Create venv, install dependencies, run lora2.py
# Usage: bash run_lora2.sh
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

command -v $PYTHON &>/dev/null || error "python3 not found. Install it first."
command -v pip3    &>/dev/null || error "pip3 not found. Install it first."

[[ -f "$SCRIPT" ]] || error "$SCRIPT not found in $(pwd). Place this script next to $SCRIPT."

# ── Python version check ───────────────────────────────────────────
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
PY_VERSION="$PY_MAJOR.$PY_MINOR"
info "Python version: $PY_VERSION"

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

# ── CUDA/GPU check ────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warning "No GPU detected. Training will be extremely slow on CPU."
fi

# ══════════════════════════════════════════════════════════════════
# 1. Create virtual environment
# ══════════════════════════════════════════════════════════════════
if [[ -d "$VENV_DIR" ]]; then
    warning "Virtual environment '$VENV_DIR' already exists – skipping creation."
else
    info "Creating virtual environment in ./$VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

source "$VENV_DIR/bin/activate"
info "Activated venv: $(which python)"

# ══════════════════════════════════════════════════════════════════
# 2. Upgrade pip + install certifi early
# ══════════════════════════════════════════════════════════════════
info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install certifi first so SSL env vars can be set before anything
# tries to make a network request (including other pip installs)
info "Installing certifi (SSL certificates)..."
pip install --upgrade certifi --quiet

# ── Point all SSL/TLS to the venv's own certifi bundle ────────────
# This fixes the 'could not find CA bundle' error that occurs when
# the venv path contains spaces or special characters (e.g. on macOS).
CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export CURL_CA_BUNDLE="$CA_BUNDLE"
info "SSL CA bundle set to: $CA_BUNDLE"

# ══════════════════════════════════════════════════════════════════
# 3. Install requirements
# ══════════════════════════════════════════════════════════════════
info "Installing requirements..."

# ── Detect CUDA ───────────────────────────────────────────────────
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
    warning "No CUDA toolchain found – installing CPU-only packages."
fi

# ── PyTorch ───────────────────────────────────────────────────────
if [[ "$CUDA_AVAILABLE" == true ]]; then
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
        info "Installing torch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121 --quiet
    elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
        info "Installing torch with CUDA 11.8 support..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 --quiet
    else
        warning "Older CUDA (<11). Installing default torch – GPU support not guaranteed."
        pip install torch torchvision torchaudio --quiet
    fi
else
    info "Installing CPU-only torch..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ── bitsandbytes ──────────────────────────────────────────────────
if [[ "$BNB_SUPPORTED" == true ]]; then
    info "Installing bitsandbytes >=0.43.1 ..."
    pip install "bitsandbytes>=0.43.1" --quiet
    if [[ "$CUDA_AVAILABLE" == false ]]; then
        warning "No GPU found. 4-bit quantisation is not supported on CPU."
        warning "Set load_in_4bit=False in lora2.py when running without a GPU."
    fi
else
    info "Installing bitsandbytes==0.42.0 (Python 3.9 compatible)..."
    pip install "bitsandbytes==0.42.0" --quiet
    if [[ "$CUDA_AVAILABLE" == false ]]; then
        warning "No GPU + Python 3.9: bitsandbytes 0.42.0 will crash on import without CUDA."
        warning "Please upgrade to Python 3.10+ or run on a machine with a GPU."
    fi
fi

# ── Remaining core packages ───────────────────────────────────────
pip install \
    "transformers>=4.40.0" \
    "peft>=0.10.0" \
    "trl>=0.8.0" \
    "accelerate>=0.29.0" \
    "datasets>=2.18.0" \
    pandas \
    --quiet

info "All requirements installed."

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
#    SSL env vars are already exported from step 2, so the model
#    download inside lora2.py will use the correct CA bundle.
# ══════════════════════════════════════════════════════════════════
info "Starting training: python $SCRIPT"
echo "──────────────────────────────────────────────────────────────"

python "$SCRIPT"

echo "──────────────────────────────────────────────────────────────"
info "Training complete. Adapters saved to ./qwen-lora-adapters"