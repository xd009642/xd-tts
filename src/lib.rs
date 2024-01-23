use std::env;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::{Layer, Registry};

pub mod infer;
pub mod phonemes;
pub mod speedyspeech;
pub mod tacotron2;
pub mod text_normaliser;
pub mod training;

pub fn setup_logging() {
    let filter = match env::var("RUST_LOG") {
        Ok(_) => EnvFilter::from_env("RUST_LOG"),
        _ => EnvFilter::new("xd_tts=debug,app=info,trainer=info"),
    };

    let fmt = tracing_subscriber::fmt::Layer::default();

    let subscriber = filter.and_then(fmt).with_subscriber(Registry::default());

    tracing::subscriber::set_global_default(subscriber).unwrap();
}
