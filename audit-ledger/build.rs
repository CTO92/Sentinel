/// Build script for the SENTINEL Audit Ledger.
///
/// In a full build environment this would compile `.proto` files via tonic-build.
/// Since we define our gRPC types manually (prost structs + tonic trait impls)
/// to keep the repository self-contained, this build script is a no-op but is
/// retained so that the proto compilation step can be re-enabled when the
/// project moves to a shared proto repository.
fn main() {
    // If you have proto files, uncomment the following:
    // tonic_build::configure()
    //     .build_server(true)
    //     .build_client(true)
    //     .compile(&["proto/audit.proto"], &["proto/"])
    //     .expect("Failed to compile proto files");

    // Trigger rebuild when migrations change.
    println!("cargo:rerun-if-changed=src/storage/migrations/");
}
