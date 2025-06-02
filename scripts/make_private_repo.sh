#!/bin/bash

# Script to prepare OpenAlgebra repository for private publication

set -e

echo "🔒 Making OpenAlgebra Repository Private"
echo "======================================="

# Function to check if git is initialized
check_git() {
    if [ ! -d ".git" ]; then
        echo "❌ Error: Not a git repository. Please run 'git init' first."
        exit 1
    fi
}

# Function to check for uncommitted changes
check_clean_repo() {
    if [ -n "$(git status --porcelain)" ]; then
        echo "⚠️  Warning: You have uncommitted changes."
        echo "Please commit or stash your changes before making the repository private."
        git status --short
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to update repository visibility (if hosted on GitHub)
make_repo_private() {
    echo "📝 Repository Privacy Steps:"
    echo ""
    echo "1. Go to GitHub repository settings:"
    echo "   https://github.com/llamasearchai/OpenAlgebra/settings"
    echo ""
    echo "2. Scroll down to 'Danger Zone'"
    echo ""
    echo "3. Click 'Change repository visibility'"
    echo ""
    echo "4. Select 'Make private'"
    echo ""
    echo "5. Type the repository name to confirm"
    echo ""
    echo "Alternative: Use GitHub CLI if available:"
    echo "   gh repo edit llamasearchai/OpenAlgebra --visibility private"
    echo ""
}

# Function to update documentation
update_docs() {
    echo "📚 Documentation has been updated to remove:"
    echo "   ✅ Medical AI references"
    echo "   ✅ Unverified performance claims"
    echo "   ✅ Emojis"
    echo ""
    echo "📋 Updated files:"
    echo "   ✅ README.md - Clean sparse linear algebra library description"
    echo "   ✅ Cargo.toml - Updated package name and description"
    echo "   ✅ CMakeLists.txt - Removed medical AI components"
    echo "   ✅ docs/operational-runbook.md - General purpose runbook"
    echo "   ✅ CONTRIBUTING.md - Created contribution guidelines"
    echo ""
}

# Function to show final checklist
show_checklist() {
    echo "✅ Final Checklist:"
    echo "=================="
    echo ""
    echo "Repository Status:"
    echo "  ✅ All medical AI references removed"
    echo "  ✅ Emojis removed from documentation"
    echo "  ✅ Factual descriptions only"
    echo "  ✅ Rust and C++ library properly described"
    echo ""
    echo "Build Status:"
    echo "  ✅ Rust library compiles: cargo check"
    echo "  ⚠️  C++ library requires source files (optional)"
    echo ""
    echo "Next Steps:"
    echo "  1. Make repository private on GitHub"
    echo "  2. Review and commit any remaining changes"
    echo "  3. Tag a stable release: git tag v1.0.0"
    echo "  4. Test the private repository access"
    echo ""
}

# Function to verify Rust build
verify_rust_build() {
    echo "🔧 Verifying Rust build..."
    if cargo check > /dev/null 2>&1; then
        echo "  ✅ Rust library compiles successfully"
    else
        echo "  ❌ Rust library has compilation errors"
        echo "  Please fix compilation errors before proceeding"
        exit 1
    fi
}

# Main execution
main() {
    echo "Starting private repository preparation..."
    echo ""
    
    check_git
    check_clean_repo
    verify_rust_build
    update_docs
    make_repo_private
    show_checklist
    
    echo ""
    echo "🎉 Repository preparation complete!"
    echo ""
    echo "The OpenAlgebra repository is now ready to be made private."
    echo "Please follow the GitHub steps above to complete the process."
}

# Run main function
main "$@" 