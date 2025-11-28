
# SilicoPepSeek

This repository implements the **SilicoPepSeek** model architecture, along with benchmark comparisons using DeepSeek and GPT-based approaches. The codebase provides the foundational model definitions and utility scripts required to instantiate the architectures for peptide-receptor sequence analysis.

## Repository Structure

The repository is organized as follows:

```text
.
├── LICENSE                                 # GNU Affero General Public License v3.0
├── SilicoPepSeek/
│   ├── code/
│   │   ├── SilicoPepSeek_model.py          # Main model architecture (MLA, MoE, Block definitions)
│   │   └── SilicoPepSeek_utils.py          # Tokenization and data handling utilities
│   └── model/
│       └── [Model Checkpoints]             # Pre-trained weights (e.g., fold0_best_model.pth)
└── benchmark_models/
    ├── DeepSeek-based_model/
    │   ├── DeepSeek_MLA_MOE_model.py       # DeepSeek-based variant architecture
    │   └── DeepSeek_MLA_MOE_utils.py       # Utilities for DeepSeek variant
    └── GPT-based_model/
        ├── pepGPT_model.py                 # GPT-based variant architecture (includes RNN/LSTM baselines)
        └── pepGPT_utils.py                 # Utilities for GPT variant
```

## Included Content

This release focuses on providing the core model architectures and essential utilities necessary for understanding and instantiating the models. To maintain a lightweight and focused codebase, we have included only the files strictly required for model definition and inference structure. Auxiliary files, such as those used for large-scale training loops, extensive testing suites, and raw data generation pipelines, are not included in this distribution.



## Setup and Installation

### Prerequisites

The codebase relies on PyTorch and several specific libraries for attention and mixture-of-experts layers. Ensure you have a Python environment set up with the following dependencies:

- **Python** (>= 3.8)
- **PyTorch** (with CUDA support recommended)
- **NumPy**
- **Einops**
- **rotary-embedding-torch**
- **st-moe-pytorch**



### Installation

You can install the necessary dependencies using pip:

```
pip install torch numpy einops rotary-embedding-torch st-moe-pytorch openpyxl
```



## Usage

The core model is defined in `SilicoPepSeek/code/SilicoPepSeek_model.py`. The architecture utilizes Multi-Head Latent Attention (MLA) and Mixture-of-Experts (MoE) blocks.

To instantiate the model in your own script:

```
import torch
from SilicoPepSeek.code.SilicoPepSeek_model import BigramLanguageModel

# Model parameters are currently defined within the model file 
# (n_embd=64, n_layer=4, etc.)
vocab_size = 100 # Example vocab size, adjust based on your token_dict
model = BigramLanguageModel(vocab_size)

# Move to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example forward pass
# Create dummy input (Batch Size, Sequence Length)
dummy_input = torch.randint(0, vocab_size, (1, 100)).to(device)
logits, loss = model(dummy_input)

print(f"Logits shape: {logits.shape}")
```

For benchmark models (DeepSeek-based or GPT-based), refer to the respective files in the `benchmark_models/` directory, which follow a similar usage pattern.



## License

This project is licensed under the **GNU Affero General Public License v3.0**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.