#!/bin/bash

# Function to print error and exit
error_exit() {
    echo "Error: $1"
    exit 1
}

# Default value for TIRAMISU_INSTALL_DIR
TIRAMISU_INSTALL_DIR=""

# Parse command-line options
while getopts "o:" opt; do
    case "$opt" in
        o) TIRAMISU_INSTALL_DIR="$OPTARG" ;;
        *) error_exit "Invalid option: -$OPTARG" ;;
    esac
done

# Set default TIRAMISU_INSTALL_DIR if not provided
if [ -z "$TIRAMISU_INSTALL_DIR" ]; then
    TIRAMISU_INSTALL_DIR="$PWD/install"
fi

# Detect the Linux distribution by checking package manager
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release

        if command -v apt &> /dev/null; then
            DISTRO="ubuntu"  # Ubuntu/Debian based
        elif command -v pacman &> /dev/null; then
            DISTRO="arch"    # Arch-based
        elif command -v dnf &> /dev/null; then
            DISTRO="fedora"  # Fedora-based
        else
            error_exit "Unsupported distribution: $ID"
        fi
    else
        error_exit "Unable to detect distribution."
    fi
}

# Install dependencies based on the detected distribution
install_dependencies() {
    echo "Installing dependencies for $DISTRO..."
    case "$DISTRO" in
        ubuntu)
            sudo apt install -y cmake libisl-dev zlib1g-dev || error_exit "Failed to install dependencies"
            ;;
        arch)
            sudo pacman -Sy --noconfirm cmake isl zlib || error_exit "Failed to install dependencies"
            ;;
        fedora)
            sudo dnf install -y cmake isl zlib-devel || error_exit "Failed to install dependencies"
            ;;
    esac
    echo "Dependencies installed successfully."
}

# Download and extract Halide binaries
download_halide() {
    link="https://github.com/halide/Halide/releases/download/v14.0.0/Halide-14.0.0-x86-64-linux-6b9ed2afd1d6d0badf04986602c943e287d44e46.tar.gz"
    filename="Halide-14.0.0-x86-64-linux.tar.gz"

    echo "Downloading Halide binaries..."
    mkdir -p 3rdParty/Halide-bin || error_exit "Failed to create directory for Halide binaries"
    cd 3rdParty/Halide-bin || error_exit "Failed to enter Halide-bin directory"

    if [ ! -f $filename ]; then
        wget $link -O $filename || error_exit "Failed to download Halide binaries"
    fi

    # Clean up existing directories if they exist
    if [ -d "bin" ] || [ -d "include" ] || [ -d "lib" ] || [ -d "share" ]; then
        echo "Cleaning up existing directories..."
        rm -rf bin include lib share || error_exit "Failed to clean up existing directories"
    fi

    echo "Extracting Halide binaries..."
    tar -xzf $filename || error_exit "Failed to extract Halide binaries"

    echo "Moving Halide files to desired structure..."
    mv Halide-14.0.0-x86-64-linux/* . || error_exit "Failed to move Halide files"
    rm -rf Halide-14.0.0-x86-64-linux || error_exit "Failed to clean up extracted folder"

    cd ../../ || error_exit "Failed to return to root directory"
    echo "Halide binaries downloaded, extracted, and reorganized successfully."
}

# Set environment variables
set_environment() {
    export TIRAMISU_ROOT="$PWD"
    export LD_LIBRARY_PATH=${TIRAMISU_ROOT}/3rdParty/Halide-bin/lib:$LD_LIBRARY_PATH
    export CMAKE_PREFIX_PATH=${TIRAMISU_ROOT}/3rdParty/Halide-bin/:$CMAKE_PREFIX_PATH
    echo "Environment variables set."
}

# Save environment variables to .bashrc and .zshrc
save_environment_variables() {
    local RC_FILES=("$HOME/.bashrc" "$HOME/.zshrc")

    for rc_file in "${RC_FILES[@]}"; do
        if [ -f "$rc_file" ]; then
            echo "Saving environment variables to $rc_file..."

            # Add LD_LIBRARY_PATH if not already set
            if ! grep -q "LD_LIBRARY_PATH=.*${TIRAMISU_ROOT}/3rdParty/Halide-bin/lib" "$rc_file"; then
                echo "export LD_LIBRARY_PATH=${TIRAMISU_ROOT}/3rdParty/Halide-bin/lib:\$LD_LIBRARY_PATH" >> "$rc_file"
            fi

            # Add CMAKE_PREFIX_PATH if not already set
            if ! grep -q "CMAKE_PREFIX_PATH=.*${TIRAMISU_ROOT}/3rdParty/Halide-bin/" "$rc_file"; then
                echo "export CMAKE_PREFIX_PATH=${TIRAMISU_ROOT}/3rdParty/Halide-bin/:\$CMAKE_PREFIX_PATH" >> "$rc_file"
            fi

            # Add TIRAMISU_ROOT if not already set
            if ! grep -q "TIRAMISU_ROOT=.*${TIRAMISU_ROOT}" "$rc_file"; then
                echo "export TIRAMISU_ROOT=${TIRAMISU_ROOT}" >> "$rc_file"
            fi

        fi
    	echo "Environment variables saved to $rc_file."
    done
}

# Build Tiramisu
build_tiramisu() {
    echo "Building Tiramisu..."

    # Clean up the build and install directories just in case, to prevent some CMake errors
    rm -rf build

    cmake . -B build -DBIN_HALIDE_INSTALL_METHOD=1 -DCMAKE_INSTALL_PREFIX=$TIRAMISU_INSTALL_DIR -DWITH_PYTHON_BINDINGS=false || error_exit "CMake configuration failed"
    cmake --build build --config Release -- -j$(nproc) tiramisu tiramisu_auto_scheduler || error_exit "Build failed"

    mkdir -p $TIRAMISU_INSTALL_DIR
    rm -rf install/*  # clean up if it existed before already
    cmake --install build || error_exit "Installation failed"
    echo "Tiramisu built and installed successfully at $TIRAMISU_INSTALL_DIR."
}

# Main script execution
detect_distro
install_dependencies
download_halide
set_environment
build_tiramisu
save_environment_variables

echo ""
echo ""
echo "Tiramisu build and install process completed successfully."
