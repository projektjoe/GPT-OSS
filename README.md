# GPT-OSS: Implementation from Scratch in Python

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational, from-scratch implementation of OpenAI's GPT-OSS model in Python. This project demonstrates how large language models work under the hood.
Check out the blog post at [ProjektJoe](https://www.projektjoe.com/blog/gptoss)

## Overview

This repository contains a complete implementation of the GPT-OSS transformer architecture, in Python, including:

- **Custom BFloat16 implementation** in C++ for numerical precision
- **Mixture of Experts (MoE)**
- **Rotary Position Embeddings (RoPE)** with NTK-aware scaling
- **Qrouped Query Attention** with attention sinks and sliding window
- **Functional** SwiGLU, RMSNorm, Softmax, Linear Layer


## Features

- **Educational Focus**: Clear, commented code designed for learning
- **Numerical Accuracy**: Matches PyTorch reference implementation
- **Comprehensive Tests**: Token-by-token validation against reference model
- **Modular Design**: Easy to understand and modify
- **Flexible Installation**: Core functionality without PyTorch dependency

## Quick Start

### Prerequisites

- Ubuntu 22.04 or Ubuntu 24.04

### Installation

> 📚 **Detailed installation guide:** See [INSTALL.md](INSTALL.md) for comprehensive installation instructions and troubleshooting.

1. **Clone the repository**
    ```bash
    git clone https://github.com/projektjoe/gptoss.git
    cd gptoss
    ```

2. **Install system dependencies**
    ```bash
    sudo apt update
    sudo apt install -y \
        python3-dev \
        libopenblas-dev \
        build-essential \
        libdnnl-dev \
        cmake
    ```

3. **Set up Python environment and install**


#### Mode A: Basic Installation (without PyTorch support - default)
1. Install UV
    ```bash
    # Install uv (fast Python package installer)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Restart your terminal
3. Create venv and install the project
    ```bash
    # Create and activate virtual environment
    uv venv .venv
    source .venv/bin/activate
    
    # Install package (this will automatically build C++ extensions)
    uv pip install -e .
    ```
4. Download the model weights of GPTOSS-20B and place them in the root folder
You can download the model weights from the [Hugging Face Hub](https://huggingface.co/openai/gpt-oss-20b) or directly from Hugging Face CLI:
    ```bash
    hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
    ```

6. Run the main script
    ```bash
    python main.py
    ```


#### Mode B: Installation with PyTorch Support (optional)
If you want to use the PyTorch layers to match the official OpenAI implementation for exact numerical accuracy:


1. Do all of the steps above
2. Install torch, then rerun the install for the project in --no-build-isolation mode
    ```bash
    # Method 1: Install torch first, then use no-build-isolation
    uv pip install torch
    uv pip install scikit_build_core
    uv pip install --no-build-isolation -e .
    
    # Method 2: Set environment variable to disable build isolation
    export UV_NO_BUILD_ISOLATION=1
    uv pip install -e ".[torch]"
    ```
3. Run the main script, which will now use torch linear layer instead of ours.
    ```bash
    python main.py
    ```
4. You could also run the test, which verifies the numerical consistency between our implementation and official OpenAI implementation via Torch.
   ```bash
   python test/test.py
   ```
   The test could be ran in two modes. by setting the VERIFY_LAYER_BY_LAYER = True, we will feed the output from official implementation to our next layer to isolate the testing layer by layer.
   if we set VERIFY_LAYER_BY_LAYER = False, we will test the entire model. If there are any errors, they will propagate to layers that come after.



The test suite performs token-by-token comparison with PyTorch's reference implementation, validating:
- Embedding lookup
- RMSNorm computations
- QKV projections
- RoPE application
- Attention mechanisms
- MoE routing and expert computation
- Final logits

## Architecture

### Overview

GPT-OSS is a 20 billion parameter transformer language model featuring:
- **Architecture**: Decoder-only transformer
- **Layers**: 36 transformer blocks
- **Hidden Size**: 2880
- **Attention**: Grouped-query attention with sliding window
- **FFN**: Mixture of 32 experts with top-4 routing

### High-Level Flow

```
Input Token
    ↓
Embedding (vocab_size → hidden_size)
    ↓
┌─────────────────────────────────────┐
│  Transformer Block (×36)            │
│  ┌───────────────────────────────┐  │
│  │ Attention                     │  │
│  │  • RMSNorm                    │  │
│  │  • QKV Projection             │  │
│  │  • RoPE                       │  │
│  │  • Scaled Dot-Product         │  │
│  │  • Output Projection          │  │
│  │  • Residual Connection        │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Mixture of Experts            │  │
│  │  • RMSNorm                    │  │
│  │  • Expert Routing (top-4)    │  │
│  │  • Expert Computation         │  │
│  │  • Weighted Combination       │  │
│  │  • Residual Connection        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
Unembedding (hidden_size → vocab_size)
    ↓
Logits
```



## Project Structure

```
gptoss/
├── main.py                 # Main model implementation and generation
├── load.py                 # Checkpoint loading and MXFP4 dequantization
├── dtypes/                 # Custom data type implementations
│   ├── bfloat16.cpp        # BFloat16 array operations
│   ├── bfloat16.hpp        # BFloat16 header
│   ├── linear.cpp          # Optimized linear layers (oneDNN)
│   ├── linear_torch.cpp    # Optional PyTorch-based linear layer
│   └── CMakeLists.txt      # Build configuration
├── test/
│   └── test.py             # Validation tests vs reference
├── official_implementation.py  # PyTorch reference (for testing)
├── pyproject.toml          # Project metadata and dependencies
└── README.md              # This file
```

## Testing

The project includes tests that validate numerical correctness.

To run the tests
```bash
python3 test/test.py
```

Output:
```bash
# Example test output
[OK] block[0].attn.norm token 0 passed.
[OK] qkv layer 0 token 0 passed.
[OK] rope q layer 0 token 0 passed.
[OK] rope k layer 0 token 0 passed.
[OK] att layer 0 token 0 passed.
[OK] linear & residual layer 0 token 0 passed.
[OK] gate layer 0 token 0 passed.
[OK] moe layer 0 token 0 passed.
```

## 🤝 Contributing

Contributions are welcome!.

Areas for contribution:
- Performance optimizations
- Additional documentation and tutorials
- Support for other platforms (macOS, Windows)
- Jupyter notebook tutorials
- Visualization tools



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for releasing GPT-OSS

