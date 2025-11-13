# #!/bin/bash

# # Setup virtual environment and install dependencies
# # Run this script from the project root directory

# echo "🚀 Setting up Python virtual environment..."

# # Create virtual environment
# python3 -m venv venv

# # Activate virtual environment
# source venv/bin/activate

# # Upgrade pip
# echo "📦 Upgrading pip..."
# pip install --upgrade pip

# # Install dependencies
# echo "📚 Installing dependencies from requirements.txt..."
# pip install -r requirements.txt

# echo "✅ Setup complete!"
# echo "📝 To activate the virtual environment, run:"
# echo "   source venv/bin/activate"
# echo ""
# echo "🏃 To run the program:"
# echo "   python Distribution.py /path/to/images/directory"



#!/bin/bash

# ============================================================================
# Virtual Environment Setup Script for Leaffliction Project
# ============================================================================
# This script creates a Python virtual environment and installs all required
# packages for the Image Transformation Program using PlantCV
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print banner
echo "============================================================================"
echo "  Virtual Environment Setup for Leaffliction Image Transformation"
echo "============================================================================"
echo ""

# Step 1: Check Python installation
print_info "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Found: $PYTHON_VERSION"
else
    print_error "Python3 is not installed!"
    print_info "Install Python3 with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Step 2: Check if venv module is available
print_info "Checking python3-venv module..."
if python3 -m venv --help >/dev/null 2>&1; then
    print_success "python3-venv is available"
else
    print_error "python3-venv is not installed!"
    print_info "Install it with: sudo apt install python3-venv"
    exit 1
fi

# Step 3: Check if we're in the Leaffliction directory
print_info "Checking current directory..."
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" = "Leaffliction" ]; then
    print_success "You are in the Leaffliction directory"
else
    print_warning "You may not be in the Leaffliction directory"
    print_info "Current directory: $PWD"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi
fi

# Step 4: Create virtual environment
VENV_DIR="venv"
print_info "Creating virtual environment in '$VENV_DIR'..."

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists!"
    read -p "Delete and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Step 5: Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

if [ $? -eq 0 ]; then
    print_success "Virtual environment activated"
    print_info "Python location: $(which python)"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Step 6: Upgrade pip
print_info "Upgrading pip to latest version..."
pip install --upgrade pip --quiet
if [ $? -eq 0 ]; then
    print_success "pip upgraded successfully"
else
    print_warning "Failed to upgrade pip, continuing anyway..."
fi

# Step 7: Install required packages
print_info "Installing required packages..."
echo ""
echo "This will install:"
echo "  - plantcv (for plant image analysis)"
echo "  - opencv-python (for computer vision)"
echo "  - numpy (for numerical operations)"
echo "  - matplotlib (for plotting)"
echo ""

pip install plantcv opencv-python numpy matplotlib

if [ $? -eq 0 ]; then
    print_success "All packages installed successfully!"
else
    print_error "Some packages failed to install"
    print_info "Check the error messages above"
    exit 1
fi

# Step 8: Verify installations
print_info "Verifying installations..."
python3 << EOF
import sys
packages = {
    'plantcv': 'plantcv',
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib'
}

all_ok = True
for module, package in packages.items():
    try:
        __import__(module)
        print(f"✓ {package:20s} - OK")
    except ImportError:
        print(f"✗ {package:20s} - FAILED")
        all_ok = False

sys.exit(0 if all_ok else 1)
EOF

if [ $? -eq 0 ]; then
    print_success "All packages verified successfully!"
else
    print_error "Some packages failed verification"
    exit 1
fi

# Step 9: Create requirements.txt
print_info "Creating requirements.txt file..."
pip freeze > requirements.txt
print_success "requirements.txt created"

# Step 10: Display summary and usage instructions
echo ""
echo "============================================================================"
echo "  Setup Complete!"
echo "============================================================================"
echo ""
print_success "Virtual environment is ready to use!"
echo ""
echo "To use the virtual environment:"
echo ""
echo "  1. Activate it (do this every time you open a new terminal):"
echo -e "     ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "  2. Run your Distribution.py script:"
echo -e "     ${GREEN}python Distribution.py ./leaves${NC}"
echo ""
echo "  3. When you're done, deactivate the virtual environment:"
echo -e "     ${GREEN}deactivate${NC}"
echo ""
echo "Quick reference:"
echo "  - Your virtual environment: $PWD/$VENV_DIR"
echo "  - Python executable: $PWD/$VENV_DIR/bin/python"
echo "  - Pip executable: $PWD/$VENV_DIR/bin/pip"
echo ""
echo "============================================================================"

# Create a quick activation script
print_info "Creating activation helper script..."
cat > activate_venv.sh << 'ACTIVATION_SCRIPT'
#!/bin/bash
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "To deactivate, type: deactivate"
ACTIVATION_SCRIPT

chmod +x activate_venv.sh
print_success "Created activation helper: ./activate_venv.sh"
echo ""
print_info "You can also activate quickly with: ${GREEN}source activate_venv.sh${NC}"
echo ""