use burn_import::onnx::ModelGen;

fn main() {
    // Generate Rust code from the ONNX model file
    ModelGen::new()
        .input("models/speedyspeech.onnx")
        .out_dir("models/")
        .run_from_script();
}
