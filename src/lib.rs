use std::env;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::{Layer, Registry};

pub mod cmu_dict;
pub mod infer;
pub mod phonemes;
pub mod tacotron2;
pub mod text_normaliser;
pub mod training;

pub use cmu_dict::CmuDictionary;

// so speedyspeech largely failed because of 2 reasons:
//
// 1. Incomplete ONNX support in the ecosystem
// 2. The graph wasn't designed with inference outside of python in mind
//
// I gave up on it before I discovered ORT as a ONNX runtime library in Rust but tacotron2 already
// works in tract and will likely work in more runtimes than speedyspeech because the repos being
// used (nvidia) are designed with exporting to run in more optimised runtimes. So there's things
// like a working ONNX export script that takes into account issues in ONNX support in torch and
// renders the graph in a friendlier form.
//
// Additionally, the use of dynamic sized tensors at graph input and internals makes it problematic
// for JIT tracing - which I believe I've mentioned elsewhere in the repo. When exporting from
// torch JIT tracing will make some variable sized axes fixed size. This could result in a model
// being exported in a way that prevents it working for inferences other than the one used for the
// export. Speedyspeech is on an old version of torch and the design make it a difficult thing to
// export out now in the current year (2023/2024).
//pub mod speedyspeech;

pub fn setup_logging() {
    let filter = match env::var("RUST_LOG") {
        Ok(_) => EnvFilter::from_env("RUST_LOG"),
        _ => EnvFilter::new("xd_tts=info,app=info,trainer=info"),
    };

    let fmt = tracing_subscriber::fmt::Layer::default();

    let subscriber = filter.and_then(fmt).with_subscriber(Registry::default());

    tracing::subscriber::set_global_default(subscriber).unwrap();
}
