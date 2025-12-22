# Installation Guide

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended for deep learning)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/kairos-therapeutics.git
cd kairos-therapeutics
```

### 2. Set Up Environment

#### Using the setup script (Linux/Mac):

```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

#### Manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### 3. Configure Environment Variables

Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- OpenAI API key (if using GPT models)
- Weights & Biases API key (for experiment tracking)
- Database credentials

### 4. Download Pre-trained Models (Optional)

```bash
# TODO: Add instructions for downloading pre-trained models
```

### 5. Verify Installation

```bash
# Run tests
pytest tests/

# Check imports
python -c "from kairos import drug_discovery, medical_imaging, clinical_analysis, genomics; print('Success!')"
```

## GPU Setup

### CUDA Installation

For GPU acceleration, install CUDA toolkit:

1. Download CUDA from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Troubleshooting

### RDKit Installation Issues

If RDKit installation fails:

```bash
conda install -c conda-forge rdkit
```

### Memory Issues

For large-scale analyses, increase available memory or use batch processing.

## Next Steps

- Review the [Usage Guide](usage.md)
- Explore [Example Notebooks](../notebooks/)
- Read the [API Documentation](api.md)
