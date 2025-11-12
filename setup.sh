#!/bin/bash

# Setup virtual environment and install dependencies
# Run this script from the project root directory

echo "🚀 Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "📝 To activate the virtual environment, run:"
echo "   source venv/bin/activate"
echo ""
echo "🏃 To run the program:"
echo "   python Distribution.py /path/to/images/directory"
