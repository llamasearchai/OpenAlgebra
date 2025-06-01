fn main() {
    // Simple build script for OpenAlgebra Medical AI
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=OPENALGEBRA_VERSION=1.0.0");
    
    // Platform-specific configurations
    configure_platform_specific();
}

fn configure_platform_specific() {
    let target = std::env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=TARGET_PLATFORM={}", target);
    
    // Basic platform-specific linking
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=m");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=kernel32");
    }
} 