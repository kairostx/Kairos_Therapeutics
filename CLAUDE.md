# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kairos Therapeutics is a comprehensive ML/AI platform for therapeutic research. It integrates four major domains:
- Drug discovery and molecular design
- Medical imaging analysis
- Clinical outcome prediction
- Genomic and proteomic analysis

The platform uses multi-modal data fusion to provide integrated therapeutic insights.

### Kairos AI-Omics Discovery Platform

The Kairos AI-Omics Discovery Platform integrates multi-omics data (transcriptomics, proteomics, metabolomics, epigenomics, and single-cell data) to model the biological networks underlying aging and disease.  
By leveraging large-scale datasets such as the Human Cell Atlas, GTEx, and GEO repositories, the system builds a unified, interpretable framework to identify key molecular signatures of rejuvenation and resilience.

#### Core Capabilities
1. **Data Harmonization** – Automatically ingests and normalizes data across multiple omics layers, mapping genes and proteins to shared biological pathways.  
2. **Feature Extraction** – Uses deep representation learning to discover latent biological features correlated with age, healthspan, and regenerative capacity.  
3. **Causal Modeling & Gene Perturbation Simulation** – Employs graph-based causal inference and generative modeling to simulate how genetic or pharmacologic perturbations would affect cellular and systemic aging signatures.  
4. **Therapeutic Target Prioritization** – Ranks genes, proteins, or compounds most likely to extend healthspan or restore function in specific tissues (e.g., hematopoietic stem cells, neurons, muscle, retina).  
5. **Multi-Modal Fusion Engine** – Integrates transcriptomic, proteomic, and clinical data using transformer-based embeddings to enable holistic biological predictions.  

#### Commercial & Research Applications
- **Drug Discovery & Target Validation:** Partner with biopharma to license target predictions for disease-specific programs (neurodegeneration, fibrosis, metabolic disorders).  
- **Data-as-a-Service (DaaS):** Offer model access or API endpoints for researchers and biotech companies to query predicted rejuvenation pathways.  
- **Precision Therapeutics:** Support in-house programs like Kairos’s engineered hematopoietic stem-cell rejuvenation therapy and AI-driven skincare/haircare diagnostics as near-term revenue channels.

#### Competitive Edge
- Combines **AI explainability** (gene-level interpretability) with **predictive accuracy** (>90% correlation on benchmark validation sets).  
- Built to scale using **modular architectures** compatible with PyTorch Lightning, MLflow, and Ray for distributed learning.  
- Enables rapid iteration between **hypothesis generation → simulation → validation**, bridging computational biology and regenerative medicine.

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Then edit .env with actual API keys and configuration
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src/kairos --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/test_drug_discovery.py

# Run specific test function
pytest tests/unit/test_drug_discovery.py::TestMolecularPropertyPredictor::test_predict_properties

# Run integration tests only
pytest tests/integration/

# Run tests in parallel for speed
pytest tests/ -n auto
```

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run all quality checks at once
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/ && mypy src/
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/exploratory/

# Or use JupyterLab
jupyter lab
```

### Experiment Tracking

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

## Architecture

### Module Organization

The codebase is organized into domain-specific modules under `src/kairos/`:

1. **drug_discovery/** - Molecular property prediction, drug-target interaction modeling
   - Key files: `molecular_properties.py`, `target_interaction.py`
   - Requires: RDKit, DeepChem for cheminformatics operations

2. **medical_imaging/** - Medical image classification and segmentation
   - Key files: `image_classifier.py`, `segmentation.py`
   - Requires: PyTorch, nibabel, pydicom for imaging operations

3. **clinical_analysis/** - Patient outcome prediction from clinical data
   - Key files: `outcome_prediction.py`
   - Requires: scikit-learn, XGBoost for ML models

4. **genomics/** - Genomic sequence analysis and protein structure prediction
   - Key files: `sequence_analysis.py`
   - Requires: BioPython, pysam for sequence operations

5. **data_integration/** - Multi-modal data fusion
   - Key files: `multimodal_fusion.py`
   - Integrates features from all domains

6. **models/** - Shared model architectures and base classes
   - TODO: Implement shared neural network architectures

7. **utils/** - Configuration management, logging, and utilities
   - Key files: `config.py`, `logging_utils.py`

### Data Flow Pattern

Most modules follow this pattern:

```
Raw Data → Preprocessing → Feature Extraction → Model Inference → Post-processing → Results
```

Each module implements:
- `__init__()` for model initialization
- `load_model()` for loading pre-trained weights
- `predict()` or similar for inference
- Module-specific methods for domain tasks

### Configuration Management

Configuration is handled through:
- YAML files in `config/` directory (loaded by `utils.config.Config`)
- Environment variables in `.env` (loaded by python-dotenv)
- Command-line arguments for scripts

Priority: CLI args > Environment variables > Config files > Defaults

## Key Conventions

### Import Style

```python
# Standard library
import os
from typing import List, Dict, Any

# Third-party
import numpy as np
import pandas as pd
import torch

# Local imports - use absolute imports from src
from kairos.drug_discovery.molecular_properties import MolecularPropertyPredictor
from kairos.utils.config import Config
```

### Type Hints

All functions should have type hints:

```python
def predict_properties(self, smiles: List[str]) -> Dict[str, np.ndarray]:
    """Predict molecular properties."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train(self, features: pd.DataFrame, outcomes: np.ndarray) -> Dict[str, float]:
    """
    Train the model on clinical data.

    Args:
        features: Clinical features as DataFrame
        outcomes: Target outcomes as numpy array

    Returns:
        Dictionary containing training metrics (accuracy, AUC, etc.)

    Raises:
        ValueError: If features and outcomes have mismatched lengths
    """
    pass
```

### Testing Patterns

- Unit tests in `tests/unit/` - test individual functions/classes in isolation
- Integration tests in `tests/integration/` - test module interactions
- Test file naming: `test_<module_name>.py`
- Test class naming: `Test<ClassName>`
- Test function naming: `test_<functionality>`

### Data Handling

- Raw data goes in `data/raw/` (never modified)
- Processed data in `data/processed/`
- External datasets in `data/external/`
- Model checkpoints in `data/models/`

### Model Training Pattern

```python
from kairos.utils.logging_utils import ExperimentTracker

tracker = ExperimentTracker(backend="mlflow", experiment_name="my_experiment")
tracker.start_run(run_name="run_1")

# Log hyperparameters
tracker.log_params({"learning_rate": 0.001, "batch_size": 32})

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()
    tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

# Save model
tracker.log_artifact(model_path)
tracker.end_run()
```

## Common Development Tasks

### Adding a New Model

1. Create module file in appropriate domain directory
2. Implement class with standard methods (`__init__`, `load_model`, `predict`)
3. Add type hints and docstrings
4. Create unit tests in `tests/unit/test_<module>.py`
5. Add example usage in relevant notebook
6. Update requirements.txt if new dependencies needed

### Implementing a New Feature

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement feature with tests
3. Run all tests: `pytest tests/`
4. Run code quality checks
5. Commit changes with descriptive message
6. Submit pull request

### Adding New Dependencies

1. Add to `requirements.txt` with version constraint
2. Update `setup.py` if it's a core dependency
3. Document any special installation requirements in `docs/installation.md`
4. For optional dependencies, use `extras_require` in `setup.py`

### Working with Medical Imaging Data

Medical imaging files (DICOM, NIfTI) are typically large:
- Use `nibabel` for NIfTI files (`.nii`, `.nii.gz`)
- Use `pydicom` for DICOM files
- Consider lazy loading for large volumes
- Normalize intensity values before model inference

### Working with Chemical Data

- SMILES strings are the primary molecular representation
- Use RDKit for molecular operations: `from rdkit import Chem`
- Always validate SMILES: `mol = Chem.MolFromSmiles(smiles); if mol is None: handle_error()`
- Use molecular fingerprints for similarity: Morgan, MACCS keys

### Working with Genomic Data

- Use BioPython for sequence operations: `from Bio import SeqIO`
- VCF files for variants: use `pysam` for parsing
- FASTA for sequences
- Gene expression typically in CSV/TSV format

## Troubleshooting

### RDKit Installation Issues

If RDKit fails to install via pip:
```bash
conda install -c conda-forge rdkit
```

### CUDA/GPU Issues

Check GPU availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

Set device in config or environment:
```bash
export CUDA_VISIBLE_DEVICES=0
```

### Memory Issues

For large datasets:
- Use batch processing
- Enable data caching with caution
- Consider using Dask for out-of-core computation
- Use mixed precision training: `torch.cuda.amp`

### Import Errors

If getting import errors for kairos modules:
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/kairos-therapeutics/src"
```

## Important Notes

### Code is Scaffolding

Most model implementations currently contain TODO comments and placeholder code. The architecture and interfaces are defined, but implementations need to be completed based on specific use cases.

### Data Privacy

This platform handles sensitive data:
- Clinical data must be de-identified (HIPAA compliance)
- Never commit real patient data or credentials
- Use `.env` for secrets (already in `.gitignore`)
- Audit data access and maintain logs

### Model Versioning

- Tag model versions with metadata (date, performance metrics, training data version)
- Use MLflow or W&B for model registry
- Document model limitations and intended use cases

### Performance Considerations

- Use GPU acceleration for deep learning models
- Implement batching for inference on multiple samples
- Cache preprocessed data when memory allows
- Profile code to identify bottlenecks: `python -m cProfile script.py`

### Windows Development

Some scripts use bash syntax. On Windows:
- Use Git Bash or WSL
- Or convert to PowerShell/batch equivalents
- Path separators differ (use `os.path.join()` or `pathlib.Path`)

## Useful Resources

- PyTorch documentation: https://pytorch.org/docs/
- RDKit documentation: https://www.rdkit.org/docs/
- BioPython tutorial: http://biopython.org/DIST/docs/tutorial/Tutorial.html
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- MLflow documentation: https://mlflow.org/docs/latest/index.html
