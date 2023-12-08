//! So what went wrong here? Tracing and torch version mismatches! 
//!
//! The reference speedyspeech implementation is on a really old version of torch (compared to the
//! Rust bindings). This means that we can't load the models into Rust easily. Torch files seem to
//! contain symbol references which then don't resolve and you get a linker error at runtime. So
//! after trying coqui's TTS package to get a pth file I attempted to load it and it failed as
//! well because it wasn't a JITed model. Attempting to JIT the model in coqui was a bit painful
//! and I gave up when faced with all the indirection between the classes they provide and running
//! a model. 
//!
//! What is JITing a Torch model? If you save a Torch model by default it contains no structure of
//! the network instead you load it in Python and the saved file is just the weights. Changing the
//! Python code showing structure of the model will make the previous file unloadable. Instead if
//! you want a fully self-contained way of saving the model with weights and structure you JIT
//! trace the model into TorchScript. This will cause things like loops to be unrolled, dynamic
//! sized inputs to become statically sized and can be a bit finickity as a result. Some amount of
//! work is needed to make sure you provide an input that doesn't result in functionality of the
//! model being broken. All in all it's a bit painful that you can't guarantee and exact
//! representation of your Python inference code outside of Python!
use super::*;
use tch::{CModule, Tensor};
use ndarray::Array2;
use std::path::Path;
use tracing::info;

pub struct SpeedyTorch {
    model: CModule,
    phoneme_ids: Vec<Unit>,
}

impl SpeedyTorch {
    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let model = CModule::load(path)?;
        Ok(Self {
            model,
            phoneme_ids: generate_id_list(),
        })
    }
    
    pub fn infer(&self, units: &[Unit]) -> anyhow::Result<Array2<f32>> {
        let phonemes = units
            .iter()
            .map(|x| best_match_for_unit(x, &self.phoneme_ids))
            .collect::<Vec<_>>();

        let plen = Tensor::f_from_slice(&[phonemes.len() as i64])?.unsqueeze(0);
        let phonemes = Tensor::f_from_slice(phonemes.as_slice())?.unsqueeze(0);

        let prediction = self.model.forward_ts(&[phonemes, plen])?;

        info!("Prediction shape: {:?}", prediction.size());

        todo!()
    }
}
