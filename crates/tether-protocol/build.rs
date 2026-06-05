fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc available");
    std::env::set_var("PROTOC", protoc);
    prost_build::Config::new()
        .compile_protos(&["proto/tether/v1.proto"], &["proto"])
        .expect("compile tether v1 proto schema");
    println!("cargo:rerun-if-changed=proto/tether/v1.proto");
}
