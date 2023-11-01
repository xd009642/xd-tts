pub mod infer;
pub mod phonemes;
pub mod text_normaliser;
pub mod training;

#[cfg(features="candle")]
pub mod speedy_candle;
