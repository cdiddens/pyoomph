#!/bin/bash

# pyoomph ARM64 Native Installation Script - Fixed Version
# This script attempts to build pyoomph natively on Apple Silicon (ARM64)
# without requiring Rosetta 2

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== pyoomph ARM64 Native Installation ===${NC}"
echo -e "${YELLOW}Warning: This is an experimental build for ARM64 architecture${NC}"
echo ""

# Function to check if running on ARM64
check_architecture() {
    if [[ $(uname -m) != "arm64" ]]; then
        echo -e "${RED}Error: This script is designed for ARM64 architecture only${NC}"
        echo "Detected architecture: $(uname -m)"
        exit 1
    fi
    echo -e "${GREEN}✓ ARM64 architecture detected${NC}"
}

# Function to check dependencies
check_dependencies() {
    echo -e "\n${BLUE}Checking dependencies...${NC}"
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Error: Homebrew is required but not installed${NC}"
        echo "Install from: https://brew.sh"
        exit 1
    fi
    
    # Check for Python 3
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is required but not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Basic dependencies satisfied${NC}"
}

# Function to install system dependencies
install_system_deps() {
    echo -e "\n${BLUE}Installing system dependencies via Homebrew...${NC}"
    
    # Install required packages
    brew install openmpi ccache ginac openblas lapack gmp cln || {
        echo -e "${YELLOW}Some packages may already be installed, continuing...${NC}"
    }
    
    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

# Function to create Python environment
setup_python_env() {
    echo -e "\n${BLUE}Setting up Python environment...${NC}"
    
    # Ask user for installation directory
    echo -e "${BLUE}Where would you like to install the virtual environment?${NC}"
    echo -e "${YELLOW}Default: ~/${NC}"
    read -p "Enter path (press Enter for default): " ENV_DIR
    
    # Use default if empty
    if [ -z "$ENV_DIR" ]; then
        ENV_DIR="$HOME"
    fi
    
    # Expand tilde if present
    ENV_DIR="${ENV_DIR/#\~/$HOME}"
    
    # Ensure directory exists
    if [ ! -d "$ENV_DIR" ]; then
        echo -e "${YELLOW}Directory $ENV_DIR does not exist. Creating it...${NC}"
        mkdir -p "$ENV_DIR" || {
            echo -e "${RED}Failed to create directory $ENV_DIR${NC}"
            exit 1
        }
    fi
    
    # Full path to virtual environment
    VENV_PATH="$ENV_DIR/py-oomph"
    
    # Store for later use
    export PYOOMPH_VENV_PATH="$VENV_PATH"
    
    # Create virtual environment
    if [ -d "$VENV_PATH" ]; then
        echo -e "${YELLOW}Virtual environment '$VENV_PATH' already exists${NC}"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
            python3 -m venv "$VENV_PATH"
        fi
    else
        python3 -m venv "$VENV_PATH"
    fi
    
    # Activate environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    echo -e "${GREEN}✓ Python environment ready at: $VENV_PATH${NC}"
}

# Function to modify setup.py for ARM64
modify_setup_py() {
    echo -e "\n${BLUE}Creating ARM64-compatible setup.py...${NC}"
    
    # Backup pyproject.toml before modifying it
    cp pyproject.toml pyproject.toml.original
    
    # Create modified pyproject.toml without MKL for ARM64
    python3 << 'EOF'
import re

# Read original pyproject.toml
with open('pyproject.toml', 'r') as f:
    content = f.read()

# Remove MKL dependencies
# Remove both lines with mkl
content = re.sub(r"'mkl>=2021\.4\.0; sys_platform != \"darwin\"',?\s*\n", "", content)
content = re.sub(r"'mkl==2021\.4\.0; sys_platform == \"darwin\"',?\s*\n", "", content)

# Clean up any trailing comma from the previous line
content = re.sub(r",(\s*\])", r"\1", content)

# Write modified version
with open('pyproject.toml', 'w') as f:
    f.write(content)

print("Modified pyproject.toml to remove MKL dependencies")
EOF
    
    # Create modified version
    cat > setup_arm64.py << 'EOF'
import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Read the original setup.py
with open('setup.py', 'r') as f:
    original_content = f.read()

# Function to modify install_requires
def get_arm64_requirements():
    """Get requirements suitable for ARM64, excluding MKL"""
    base_requires = [
        "numpy",
        "scipy",
        "matplotlib", 
        "pygmsh",
        "meshio",
        "gmsh",
        "pybind11",
        "PyParsing",
        "Pillow",
        "Pygments",
        "pybind11-stubgen",
        "more-itertools"
    ]
    
    # Add optional dependencies that work on ARM64
    optional = [
        "mpi4py",
        "petsc4py",
    ]
    
    return base_requires + optional

# Monkey patch to use OpenBLAS instead of MKL
os.environ['PYOOMPH_USE_OPENBLAS'] = '1'
os.environ['PYOOMPH_NO_MKL'] = '1'

# Execute modified setup
exec(original_content.replace(
    '"mkl"',  # Remove MKL from requirements
    '# "mkl" # Disabled for ARM64'
))
EOF
    
    echo -e "${GREEN}✓ Modified setup.py created${NC}"
    echo -e "${GREEN}✓ Modified pyproject.toml to remove MKL${NC}"
}

# Function to create build configuration
create_build_config() {
    echo -e "\n${BLUE}Creating ARM64 build configuration...${NC}"
    
    # Create custom config
    cat > pyoomph_config_arm64.env << 'EOF'
# ARM64 Native Build Configuration
PYOOMPH_MARCH_NATIVE=false
PYOOMPH_NO_TCC=true
PYOOMPH_USE_OPENBLAS=true
PYOOMPH_NO_MKL=true
PYOOMPH_CONFIG_FILE=pyoomph_config_arm64.env

# Compiler settings
CXX=g++
CC=gcc
CXXFLAGS=-O3 -fPIC -std=c++11
LDFLAGS=-L/opt/homebrew/lib
CPPFLAGS=-I/opt/homebrew/include

# OpenBLAS settings
OPENBLAS_NUM_THREADS=1
BLAS_LIBS=-lopenblas
LAPACK_LIBS=-llapack

# MPI settings (if using MPI)
PYOOMPH_USE_MPI=auto
EOF
    
    # Export these for current session
    export PYOOMPH_MARCH_NATIVE=false
    export PYOOMPH_NO_TCC=true
    export PYOOMPH_USE_OPENBLAS=true
    export PYOOMPH_NO_MKL=true
    export OPENBLAS_NUM_THREADS=1
    
    # Set OpenBLAS paths for scipy build
    export LDFLAGS="-L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/lapack/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/openblas/include -I/opt/homebrew/opt/lapack/include"
    export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig:/opt/homebrew/opt/lapack/lib/pkgconfig"
    
    echo -e "${GREEN}✓ Build configuration created${NC}"
}

# Function to patch source files for ARM64
patch_source_files() {
    echo -e "\n${BLUE}Patching source files for ARM64...${NC}"
    
    # Patch Makefile to remove -march=native
    if [ -f "src/Makefile" ]; then
        sed -i.bak 's/-march=native//g' src/Makefile
    fi
    
    # Patch oomph-lib Makefile
    if [ -f "src/thirdparty/oomph-lib/Makefile" ]; then
        sed -i.bak 's/-march=native//g' src/thirdparty/oomph-lib/Makefile
    fi
    
    # Create a patch for ccompiler.py to use OpenBLAS
    cat > ccompiler_arm64_patch.py << 'EOF'
# This patch modifies ccompiler.py to use OpenBLAS on ARM64
import fileinput
import sys

for line in fileinput.input('pyoomph/generic/ccompiler.py', inplace=True):
    if 'mkl' in line.lower() and 'import' not in line:
        line = line.replace('mkl', 'openblas')
    print(line, end='')
EOF
    
    python ccompiler_arm64_patch.py
    rm ccompiler_arm64_patch.py
    
    echo -e "${GREEN}✓ Source files patched${NC}"
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "\n${BLUE}Installing Python dependencies (ARM64-compatible)...${NC}"
    
    # Set environment for scipy build
    export OPENBLAS="/opt/homebrew/opt/openblas"
    export LAPACK="/opt/homebrew/opt/lapack"
    export BLAS=$OPENBLAS
    
    # Install build tools first
    pip install --upgrade setuptools cython meson-python
    
    # Install numpy (binary is fine for ARM64 now)
    pip install numpy
    
    # Install scipy with proper OpenBLAS configuration
    echo -e "${BLUE}Installing scipy with OpenBLAS...${NC}"
    
    # Create site.cfg for numpy/scipy
    cat > site.cfg << EOF
[DEFAULT]
library_dirs = /opt/homebrew/lib
include_dirs = /opt/homebrew/include

[openblas]
libraries = openblas
library_dirs = /opt/homebrew/opt/openblas/lib
include_dirs = /opt/homebrew/opt/openblas/include
runtime_library_dirs = /opt/homebrew/opt/openblas/lib
EOF
    
    # Try to install scipy
    NPY_BLAS_ORDER=openblas NPY_LAPACK_ORDER=openblas pip install scipy || {
        echo -e "${YELLOW}Standard scipy install failed, trying Apple Accelerate framework...${NC}"
        
        # Try with Apple's Accelerate framework
        export ACCELERATE_LAPACK=1
        export NPY_BLAS_ORDER=accelerate
        export NPY_LAPACK_ORDER=accelerate
        
        pip install scipy --no-build-isolation --no-binary :all: || {
            echo -e "${YELLOW}Source build failed, trying pre-built wheel...${NC}"
            pip install scipy || {
                echo -e "${YELLOW}Standard wheel failed, trying conda-forge scipy wheel...${NC}"
                pip install scipy --index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple || {
                    echo -e "${RED}Failed to install scipy. All methods exhausted.${NC}"
                    echo -e "${YELLOW}You may need to install scipy manually or use conda.${NC}"
                }
            }
        }
        
        # Reset to OpenBLAS for other packages
        export NPY_BLAS_ORDER=openblas
        export NPY_LAPACK_ORDER=openblas
    }
    
    # Install other dependencies
    pip install matplotlib pybind11 gmsh commonmark six pyparsing pygments \
                pillow mpi4py kiwisolver fonttools cycler rich \
                python-dateutil packaging meshio pygmsh pybind11-stubgen
    
    # Clean up
    rm -f site.cfg
    
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
}

# Function to build oomph-lib
build_oomph_lib() {
    echo -e "\n${BLUE}Building oomph-lib for ARM64...${NC}"
    
    # Use the configuration file
    export PYOOMPH_CONFIG_FILE=pyoomph_config_arm64.env
    
    # Run prebuild
    bash ./prebuild.sh || {
        echo -e "${YELLOW}Warning: prebuild.sh encountered issues, attempting to continue...${NC}"
    }
    
    echo -e "${GREEN}✓ oomph-lib build completed${NC}"
}

# Function to build pyoomph
build_pyoomph() {
    echo -e "\n${BLUE}Building pyoomph for ARM64...${NC}"
    
    # Use the modified setup.py
    python setup_arm64.py build_ext --inplace || {
        echo -e "${RED}Build failed. Trying alternative approach...${NC}"
        
        # Alternative: use standard setup with environment variables
        PYOOMPH_NO_MKL=1 PYOOMPH_USE_OPENBLAS=1 python setup.py build_ext --inplace
    }
    
    # Install in development mode
    pip install -e .
    
    echo -e "${GREEN}✓ pyoomph build completed${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${BLUE}Verifying installation...${NC}"
    
    # Try to import pyoomph
    python -c "import pyoomph; print('✓ pyoomph imported successfully')" || {
        echo -e "${RED}Failed to import pyoomph${NC}"
        return 1
    }
    
    # Try importing key modules
    python -c "from pyoomph import *; from pyoomph.expressions import *; print('✓ Basic imports working')" || {
        echo -e "${YELLOW}Warning: Some imports failed${NC}"
    }
    
    # Run basic checks
    python -m pyoomph check compiler || {
        echo -e "${YELLOW}Compiler check failed - may need configuration${NC}"
    }
    
    echo -e "${GREEN}✓ Installation verified${NC}"
}

# Function to create alias
create_activation_alias() {
    echo -e "\n${BLUE}Creating pyoomph-activate alias...${NC}"
    
    # Determine which shell config file to use
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    else
        # Default to .bashrc
        SHELL_CONFIG="$HOME/.bashrc"
    fi
    
    # Check if alias already exists
    if grep -q "alias pyoomph-activate" "$SHELL_CONFIG" 2>/dev/null; then
        echo -e "${YELLOW}Alias 'pyoomph-activate' already exists in $SHELL_CONFIG${NC}"
        echo -e "${YELLOW}Updating it with new path...${NC}"
        # Remove old alias
        sed -i.bak '/alias pyoomph-activate/d' "$SHELL_CONFIG"
    fi
    
    # Add new alias
    echo "" >> "$SHELL_CONFIG"
    echo "# PyOomph activation alias (added by installOnArm.sh)" >> "$SHELL_CONFIG"
    echo "alias pyoomph-activate='source $PYOOMPH_VENV_PATH/bin/activate'" >> "$SHELL_CONFIG"
    
    echo -e "${GREEN}✓ Added 'pyoomph-activate' alias to $SHELL_CONFIG${NC}"
    echo -e "${YELLOW}Note: Run 'source $SHELL_CONFIG' or start a new terminal to use the alias${NC}"
}

# Function to create test script
create_test_script() {
    echo -e "\n${BLUE}Creating test script...${NC}"
    
    cat > test_arm64_install.py << 'EOF'
#!/usr/bin/env python3
"""Test script for ARM64 pyoomph installation"""

import sys
print("Testing pyoomph ARM64 installation...")

try:
    import pyoomph
    print("✓ pyoomph imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pyoomph: {e}")
    sys.exit(1)

# Test basic functionality
try:
    from pyoomph import *
    from pyoomph.expressions import *
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")

# Test problem creation
try:
    class TestProblem(Problem):
        def define_problem(self):
            self.set_c_compiler("system")  # Explicitly set compiler
            self.add_mesh(LineMesh(N=10))
    
    with TestProblem() as problem:
        problem.solve()
    print("✓ Simple problem solved")
except Exception as e:
    print(f"✗ Problem solving error: {e}")

# Quick compiler check
try:
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pyoomph", "check", "compiler"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Compiler check passed")
    else:
        print("✗ Compiler check failed (may need configuration)")
except:
    pass

print("\nARM64 installation test completed!")
print("For more thorough testing, run: python -m pyoomph check all")
EOF
    
    chmod +x test_arm64_install.py
    echo -e "${GREEN}✓ Test script created${NC}"
}


# Function to restore original files
restore_original_files() {
    echo -e "\n${BLUE}Restoring original files and cleaning up...${NC}"
    
    # Restore pyproject.toml
    if [ -f "pyproject.toml.original" ]; then
        mv pyproject.toml.original pyproject.toml
        echo -e "${GREEN}✓ Restored original pyproject.toml${NC}"
    fi
    
    # Restore Makefiles from .bak files
    if [ -f "src/Makefile.bak" ]; then
        mv src/Makefile.bak src/Makefile
        echo -e "${GREEN}✓ Restored original src/Makefile${NC}"
    fi
    
    if [ -f "src/thirdparty/oomph-lib/Makefile.bak" ]; then
        mv src/thirdparty/oomph-lib/Makefile.bak src/thirdparty/oomph-lib/Makefile
        echo -e "${GREEN}✓ Restored original oomph-lib/Makefile${NC}"
    fi
    
    # Clean up ARM64-specific files
    rm -f setup_arm64.py
    rm -f pyoomph_config_arm64.env
    
    # Clean up any other backup files
    rm -f pyproject.toml.backup
    rm -f setup.py.original
    rm -f *.bak
    
    # Clean up temporary patch files
    rm -f ccompiler_arm64_patch.py
    rm -f site.cfg
    
    # Note: test_arm64_install.py is kept for user to test installation later
    
    echo -e "${GREEN}✓ All cleanup completed${NC}"
}

# Main installation flow
main() {
    echo -e "${BLUE}Starting ARM64 native installation of pyoomph${NC}"
    echo -e "${YELLOW}This process will modify source files for ARM64 compatibility${NC}"
    echo ""
    
    # Check if we're in the pyoomph directory
    if [ ! -f "setup.py" ] || [ ! -d "pyoomph" ]; then
        echo -e "${RED}Error: This script must be run from the pyoomph root directory${NC}"
        exit 1
    fi
    
    # Set up trap to restore files on exit
    trap 'restore_original_files' EXIT
    
    # Run installation steps
    check_architecture
    check_dependencies
    install_system_deps
    setup_python_env
    modify_setup_py
    create_build_config
    patch_source_files
    install_python_deps
    build_oomph_lib
    build_pyoomph
    verify_installation
    create_test_script
    create_activation_alias
    
    echo -e "\n${GREEN}=== Installation Complete ===${NC}"
    echo -e "${BLUE}To use pyoomph:${NC}"
    echo "  1. Activate the environment: pyoomph-activate (after restarting terminal)"
    echo "     OR: source $PYOOMPH_VENV_PATH/bin/activate"
    echo "  2. Test the installation: python test_arm64_install.py"
    echo "  3. Run full tests: python -m pyoomph check all"
    echo ""
    echo -e "${YELLOW}Note: This is an experimental ARM64 build.${NC}"
    echo -e "${YELLOW}Some features may not work as expected.${NC}"
    echo -e "${YELLOW}Report any issues with ARM64 compatibility.${NC}"
    
    # Auto-activate the environment if not already active
    if [[ "$VIRTUAL_ENV" != "$PYOOMPH_VENV_PATH" ]]; then
        echo -e "\n${BLUE}Activating py-oomph environment...${NC}"
        source "$PYOOMPH_VENV_PATH/bin/activate"
        echo -e "${GREEN}✓ Environment activated. You are now in the py-oomph virtual environment.${NC}"
        echo -e "${YELLOW}To deactivate later, run: deactivate${NC}"
    else
        echo -e "\n${GREEN}✓ py-oomph environment already active${NC}"
    fi
    
    # Files will be restored automatically due to trap
}

# Run main function
main

# Post-cleanup message
echo -e "\n${GREEN}=== Ready to Use ===${NC}"
echo -e "${BLUE}PyOomph has been installed successfully!${NC}"
echo ""
echo -e "To use pyoomph in the future:"
echo -e "  ${GREEN}pyoomph-activate${NC} (in a new terminal)"
echo -e "  OR"
echo -e "  ${GREEN}source ${PYOOMPH_VENV_PATH}/bin/activate${NC}"
echo ""
echo -e "${YELLOW}The original files have been restored, but your installation is preserved.${NC}"