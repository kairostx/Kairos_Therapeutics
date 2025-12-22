#!/bin/bash

# Run tests for Kairos Therapeutics

echo "Running tests..."

# Run unit tests with coverage
pytest tests/unit --cov=src/kairos --cov-report=html --cov-report=term

# Run integration tests
pytest tests/integration -v

echo "Tests complete! Coverage report available in htmlcov/index.html"
