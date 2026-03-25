//! Build script for compiling protobuf definitions into Rust types using tonic-build.
//!
//! This reads proto files from the project-root `proto/sentinel/v1/` directory
//! and generates Rust server and client stubs consumed by the gRPC layer.
//!
//! The generated code is written to `OUT_DIR` (the standard cargo build output
//! directory) and included via `include!` in `src/lib.rs`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("correlation-engine must reside inside the Sentinel workspace")
        .join("proto");

    let proto_files = &[
        proto_root.join("sentinel/v1/common.proto"),
        proto_root.join("sentinel/v1/probe.proto"),
        proto_root.join("sentinel/v1/anomaly.proto"),
        proto_root.join("sentinel/v1/health.proto"),
        proto_root.join("sentinel/v1/quarantine.proto"),
        proto_root.join("sentinel/v1/correlation.proto"),
        proto_root.join("sentinel/v1/trust.proto"),
        proto_root.join("sentinel/v1/audit.proto"),
        proto_root.join("sentinel/v1/config.proto"),
        proto_root.join("sentinel/v1/telemetry.proto"),
    ];

    // Verify all proto files exist before invoking tonic-build.
    for path in proto_files {
        if !path.exists() {
            panic!("Proto file not found: {}", path.display());
        }
    }

    // Use the default OUT_DIR so that `include!(concat!(env!("OUT_DIR"), ...))` works.
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(proto_files, &[&proto_root])?;

    // Re-run if any proto file changes.
    for path in proto_files {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    Ok(())
}
