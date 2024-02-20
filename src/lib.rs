#![doc = include_str!("../README.md")]
use std::env;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::{Layer, Registry};

pub mod cmu_dict;
pub mod phonemes;
// This failed for various reasons. Look in the module so see the pains of ML.
//pub mod speedyspeech;
pub mod tacotron2;
pub mod text_normaliser;
pub mod training;

pub use cmu_dict::CmuDictionary;

/// Convenience function to setup logging for any binaries I create. Automatically sets all
/// binaries and the tts library crate to `info` logging by default.
pub fn setup_logging() {
    let filter = match env::var("RUST_LOG") {
        Ok(_) => EnvFilter::from_env("RUST_LOG"),
        _ => EnvFilter::new("xd_tts=info,app=info,trainer=info"),
    };

    let fmt = tracing_subscriber::fmt::Layer::default();

    let subscriber = filter.and_then(fmt).with_subscriber(Registry::default());

    tracing::subscriber::set_global_default(subscriber).unwrap();
}
