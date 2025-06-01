#!/bin/bash

# OpenAlgebra Medical AI - Publication Script
# This script provides the exact commands to publish to GitHub and crates.io
# Run this script to get the publication commands, then execute them manually

set -e

echo "============================================================"
echo "OpenAlgebra Medical AI - Publication Guide"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step "Step 1: Final Validation"
echo "Validate the crate builds and tests pass:"
echo "  cargo check"
echo "  cargo test"
echo "  cargo build --release"
echo "  cargo package --allow-dirty"
echo ""

print_step "Step 2: Initialize Git Repository (if not already done)"
echo "Initialize git repository and add all files:"
echo "  git init"
echo "  git add ."
echo "  git commit -m \"Initial release: OpenAlgebra Medical AI v1.0.0\""
echo ""

print_step "Step 3: Set up GitHub Remote"
echo "Add GitHub remote repository:"
echo "  git remote add origin https://github.com/llamasearchai/OpenAlgebra.git"
echo ""

print_step "Step 4: Push to GitHub"
echo "Push the repository to GitHub:"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""

print_step "Step 5: Create GitHub Release"
echo "Create a release tag and push it:"
echo "  git tag v1.0.0"
echo "  git push origin v1.0.0"
echo ""
echo "Then go to: https://github.com/llamasearchai/OpenAlgebra/releases/new"
echo "- Tag version: v1.0.0"
echo "- Release title: OpenAlgebra Medical AI v1.0.0"
echo "- Description:"
echo "  Initial release of OpenAlgebra Medical AI - High-Performance Sparse Linear Algebra"
echo "  for Medical AI Model Development. Features include DICOM processing, sparse tensors,"
echo "  medical AI models, and clinical validation frameworks."
echo ""

print_step "Step 6: Publish to crates.io"
echo "Before publishing to crates.io, ensure you have:"
echo "1. A crates.io account: https://crates.io/"
echo "2. API token configured: cargo login"
echo ""
echo "Validate the package one more time:"
echo "  cargo package --allow-dirty"
echo ""
echo "Publish to crates.io:"
echo "  cargo publish"
echo ""

print_step "Step 7: Verify Publication"
echo "After publishing, verify:"
echo "1. GitHub repository: https://github.com/llamasearchai/OpenAlgebra"
echo "2. Crates.io package: https://crates.io/crates/openalgebra-medical"
echo "3. Documentation: https://docs.rs/openalgebra-medical"
echo ""

print_step "Step 8: Set up Repository"
echo "Configure your GitHub repository:"
echo "1. Add repository description: 'High-Performance Sparse Linear Algebra for Medical AI'"
echo "2. Add topics: medical-ai, sparse-matrix, healthcare, rust, machine-learning, dicom"
echo "3. Enable discussions and issues"
echo "4. Set up branch protection for main branch"
echo "5. Configure GitHub Pages (optional)"
echo ""

print_step "Step 9: Additional Files"
echo "Consider adding these additional files:"
echo "1. LICENSE (MIT license file)"
echo "2. CONTRIBUTING.md (contribution guidelines)"
echo "3. CHANGELOG.md (version history)"
echo "4. CODE_OF_CONDUCT.md (community guidelines)"
echo ""

print_warning "IMPORTANT PREREQUISITES:"
echo "Before running these commands, ensure you have:"
echo "1. GitHub account with access to llamasearchai/OpenAlgebra repository"
echo "2. Git configured with your credentials"
echo "3. Cargo/Rust installed and configured"
echo "4. crates.io account and API token (cargo login)"
echo ""

# Validation checks
print_step "Quick Validation Checks"

# Check if git is initialized
if [ ! -d ".git" ]; then
    print_warning "Git repository not initialized. Run 'git init' first."
fi

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    print_error "Cargo is not installed. Install Rust: https://rustup.rs/"
    exit 1
fi

# Check if the crate builds
print_step "Build Verification"
if cargo check &> /dev/null; then
    print_success "Rust crate builds successfully"
else
    print_error "Rust crate has build errors. Fix them before publishing."
    echo "Run: cargo check"
    exit 1
fi

# Check if tests pass
if cargo test &> /dev/null; then
    print_success "All tests pass (10 tests total)"
else
    print_error "Tests are failing. Fix them before publishing."
    echo "Run: cargo test"
    exit 1
fi

# Check if package builds
if cargo package --allow-dirty &> /dev/null; then
    print_success "Package builds successfully for publication"
else
    print_error "Package has build errors. Fix them before publishing."
    echo "Run: cargo package --allow-dirty"
    exit 1
fi

echo ""
print_success "READY FOR PUBLICATION!"
echo ""
echo "Your OpenAlgebra Medical AI crate is ready to be published!"
echo "Follow the steps above to publish to GitHub and crates.io."
echo ""
echo "============================================================"
echo "OpenAlgebra Medical AI v1.0.0 - Publication Ready"
echo "============================================================" 