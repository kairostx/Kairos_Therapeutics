"""
Unit tests for drug discovery module.
"""

import pytest
import numpy as np
from kairos.drug_discovery.molecular_properties import MolecularPropertyPredictor
from kairos.drug_discovery.target_interaction import DrugTargetInteractionModel


class TestMolecularPropertyPredictor:
    """Test molecular property prediction."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MolecularPropertyPredictor()
        assert predictor is not None

    def test_predict_properties(self):
        """Test property prediction."""
        predictor = MolecularPropertyPredictor()
        smiles = ['CC(C)Cc1ccc(cc1)C(C)C(O)=O']

        properties = predictor.predict_properties(smiles)

        assert isinstance(properties, dict)
        assert 'logP' in properties
        assert 'QED' in properties


class TestDrugTargetInteractionModel:
    """Test drug-target interaction modeling."""

    def test_initialization(self):
        """Test model initialization."""
        model = DrugTargetInteractionModel()
        assert model is not None
        assert model.model_type == "graph_neural_network"

    def test_predict_interaction(self):
        """Test interaction prediction."""
        model = DrugTargetInteractionModel()
        affinity, confidence = model.predict_interaction("SMILES_STRING", "PROTEIN_SEQ")

        assert isinstance(affinity, float)
        assert isinstance(confidence, float)
