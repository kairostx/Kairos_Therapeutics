#!/bin/bash

# Setup script for Kairos Therapeutics development environment

echo "Setting up Kairos Therapeutics development environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please update with your API keys and configuration."
fi

# Create data directories
mkdir -p data/{raw,processed,external,models}
mkdir -p logs

echo "Setup complete! Activate the environment with: source venv/bin/activate"
